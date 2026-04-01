#!/usr/bin/env python3
"""
Wan 2.2 VAE reconstruction using torch_neuronx.trace() on Trainium2.

This script traces the encoder and decoder as single-pass functions (no chunked
loop, no feat_cache) and compiles them to Neuron executables (NEFFs).  Each
traced module is a fixed-shape, statically-compiled graph that fits within
Trainium2's instruction limit.

Usage:
    cd examples/training/Wan2.2
    PATH="~/wan_vae/bin:$PATH" \\
    python scripts/test_vae2_2_torch_compile.py \\
        --vae-path ~/Wan2.2-VAE/Wan2.2_VAE.pth \\
        --video ~/walrus.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Load vae2_2 directly (bypass wan/__init__.py)
# ---------------------------------------------------------------------------
_WAN_ROOT = Path(__file__).resolve().parent.parent
_VAE2_2_PATH = _WAN_ROOT / "wan" / "modules" / "vae2_2.py"
_spec = importlib.util.spec_from_file_location("wan_vae2_2_standalone", _VAE2_2_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load VAE module from {_VAE2_2_PATH}")
_vae2_2 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _vae2_2
_spec.loader.exec_module(_vae2_2)

Wan2_2_VAE = _vae2_2.Wan2_2_VAE
patchify = _vae2_2.patchify
unpatchify = _vae2_2.unpatchify

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin wrappers for tracing (no cache, single-pass, pure tensor I/O)
# ---------------------------------------------------------------------------

class EncodeWrapper(nn.Module):
    """Single-pass encoder: patchified video → latent mu."""

    def __init__(self, encoder, conv1, z_dim, scale_mean, scale_inv_std):
        super().__init__()
        self.encoder = encoder
        self.conv1 = conv1
        self.z_dim = z_dim
        self.register_buffer("scale_mean", scale_mean)
        self.register_buffer("scale_inv_std", scale_inv_std)

    def forward(self, x):
        # x: [B, 12, T, H/2, W/2]  (already patchified)
        out = self.encoder(x)  # no feat_cache → full single-pass
        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        mu = (mu - self.scale_mean.view(1, self.z_dim, 1, 1, 1)) * \
             self.scale_inv_std.view(1, self.z_dim, 1, 1, 1)
        return mu


class DecodeWrapper(nn.Module):
    """Single-pass decoder: latent → reconstructed patchified video."""

    def __init__(self, decoder, conv2, z_dim, scale_mean, scale_inv_std):
        super().__init__()
        self.decoder = decoder
        self.conv2 = conv2
        self.z_dim = z_dim
        self.register_buffer("scale_mean", scale_mean)
        self.register_buffer("scale_inv_std", scale_inv_std)

    def forward(self, z):
        # z: [B, z_dim, T_lat, H_lat, W_lat]
        z = z / self.scale_inv_std.view(1, self.z_dim, 1, 1, 1) + \
            self.scale_mean.view(1, self.z_dim, 1, 1, 1)
        x = self.conv2(z)
        out = self.decoder(x, first_chunk=True)  # single-pass, trim leading temporal pad
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _largest_valid_temporal(t: int) -> int:
    if t < 1:
        raise ValueError("need at least one frame")
    return 1 + ((t - 1) // 4) * 4


def _ensure_even_hw(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 != h or w2 != w:
        x = x[:, :, :h2, :w2]
    return x


def _trim_temporal(x: torch.Tensor) -> torch.Tensor:
    t = x.shape[1]
    t_good = _largest_valid_temporal(t)
    if t_good < t:
        logger.info("trimming time: %s -> %s frames", t, t_good)
        x = x[:, :t_good]
    return x


def load_video_tensor(path, max_frames, resize_hw):
    from torchvision.io import read_video
    video, _, _ = read_video(path, output_format="TCHW")
    if video.numel() == 0:
        raise RuntimeError(f"no frames read from {path}")
    x = video.float() / 255.0 * 2.0 - 1.0
    x = x.permute(1, 0, 2, 3).contiguous()
    if max_frames is not None and x.shape[1] > max_frames:
        x = x[:, :max_frames]
    if resize_hw is not None:
        th, tw = resize_hw
        y = x.unsqueeze(0)
        y = F.interpolate(y, size=(x.shape[1], th, tw), mode="trilinear", align_corners=False)
        x = y.squeeze(0)
    return _ensure_even_hw(_trim_temporal(x))


def psnr_db(x, y):
    mse = F.mse_loss(x, y).clamp(min=1e-10)
    return float((10.0 * torch.log10(torch.tensor(4.0, device=mse.device) / mse)).item())


def save_triptych(inp, rec, out_path, frame_idx, diff_gain=4.0):
    inp = inp[:, frame_idx].clamp(-1, 1).detach().cpu()
    rec = rec[:, frame_idx].clamp(-1, 1).detach().cpu()

    def to_rgb(t):
        return ((t + 1.0) * 0.5 * 255.0).byte().permute(1, 2, 0).numpy()

    a, b = to_rgb(inp), to_rgb(rec)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    d_vis = np.clip(diff * diff_gain, 0, 255).astype(np.uint8)
    h, w, _ = a.shape
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(Image.fromarray(a), (0, 0))
    canvas.paste(Image.fromarray(b), (w, 0))
    canvas.paste(Image.fromarray(d_vis), (2 * w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    logger.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Wan 2.2 VAE torch_neuronx.trace reconstruction test")
    p.add_argument("--vae-path", type=str, required=True)
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--max-frames", type=int, default=17)
    p.add_argument("--resize-height", type=int, default=480)
    p.add_argument("--resize-width", type=int, default=848)
    p.add_argument("--output", type=str, default="/home/ubuntu/walrus_recon_traced.png")
    p.add_argument("--compiler-args", type=str, default="--optlevel 1 --model-type generic")
    p.add_argument("--cpu-only", action="store_true", help="run CPU baseline without tracing")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Load video (CPU)
    # ------------------------------------------------------------------
    resize = (args.resize_height, args.resize_width)
    x_cpu = load_video_tensor(args.video, args.max_frames, resize)
    logger.info("Video tensor: %s  range=[%.2f, %.2f]", tuple(x_cpu.shape), x_cpu.min(), x_cpu.max())

    # ------------------------------------------------------------------
    # Step 2: Load VAE on CPU
    # ------------------------------------------------------------------
    logger.info("Loading VAE checkpoint...")
    t0 = time.time()
    vae = Wan2_2_VAE(vae_pth=args.vae_path, device="cpu", dtype=torch.float32)
    logger.info("  loaded in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 3: CPU baseline encode/decode (validates correctness)
    # ------------------------------------------------------------------
    logger.info("CPU baseline encode/decode...")
    x_in = x_cpu.unsqueeze(0)  # [1, C, T, H, W]
    x_patchified = patchify(x_in, patch_size=2)
    logger.info("  patchified: %s", tuple(x_patchified.shape))

    enc_wrapper = EncodeWrapper(
        vae.model.encoder, vae.model.conv1, vae.model.z_dim,
        vae.scale[0], vae.scale[1],
    ).eval()
    dec_wrapper = DecodeWrapper(
        vae.model.decoder, vae.model.conv2, vae.model.z_dim,
        vae.scale[0], vae.scale[1],
    ).eval()

    with torch.no_grad():
        z_cpu = enc_wrapper(x_patchified)
        logger.info("  latent: %s", tuple(z_cpu.shape))
        recon_patchified = dec_wrapper(z_cpu)
        logger.info("  recon patchified: %s", tuple(recon_patchified.shape))
        recon_cpu = unpatchify(recon_patchified, patch_size=2).clamp(-1, 1)
        logger.info("  recon: %s", tuple(recon_cpu.shape))

    # Metrics
    T_cmp = min(x_in.shape[2], recon_cpu.shape[2])
    mse_cpu = F.mse_loss(x_in[:, :, :T_cmp], recon_cpu[:, :, :T_cmp]).item()
    psnr_cpu = psnr_db(x_in[:, :, :T_cmp], recon_cpu[:, :, :T_cmp])
    logger.info("  CPU MSE=%.6f  PSNR=%.2f dB  (frames compared: %d)", mse_cpu, psnr_cpu, T_cmp)

    if args.cpu_only:
        save_triptych(x_in.squeeze(0)[:, :T_cmp], recon_cpu.squeeze(0)[:, :T_cmp],
                      Path(args.output), T_cmp // 2)
        return

    # ------------------------------------------------------------------
    # Step 4: Trace with torch_neuronx.trace()
    # ------------------------------------------------------------------
    import torch_neuronx

    compiler_args = args.compiler_args.split()

    logger.info("Tracing encoder with torch_neuronx.trace()...")
    logger.info("  example input: %s", tuple(x_patchified.shape))
    t0 = time.time()
    traced_encoder = torch_neuronx.trace(
        enc_wrapper,
        x_patchified,
        compiler_args=compiler_args,
        compiler_workdir="/tmp/neuron_encoder_workdir",
    )
    t_enc_trace = time.time() - t0
    logger.info("  encoder traced in %.1fs", t_enc_trace)

    logger.info("Tracing decoder with torch_neuronx.trace()...")
    logger.info("  example input: %s", tuple(z_cpu.shape))
    t0 = time.time()
    traced_decoder = torch_neuronx.trace(
        dec_wrapper,
        z_cpu,
        compiler_args=compiler_args,
        compiler_workdir="/tmp/neuron_decoder_workdir",
    )
    t_dec_trace = time.time() - t0
    logger.info("  decoder traced in %.1fs", t_dec_trace)

    # ------------------------------------------------------------------
    # Step 5: Run traced models (first call = load NEFF, subsequent = fast)
    # ------------------------------------------------------------------
    logger.info("Running traced encode (cold)...")
    t0 = time.time()
    z_neuron = traced_encoder(x_patchified)
    t_enc_cold = time.time() - t0
    logger.info("  latent: %s  (%.1fs)", tuple(z_neuron.shape), t_enc_cold)

    logger.info("Running traced decode (cold)...")
    t0 = time.time()
    recon_neuron_p = traced_decoder(z_neuron)
    t_dec_cold = time.time() - t0
    recon_neuron = unpatchify(recon_neuron_p, patch_size=2).clamp(-1, 1)
    logger.info("  recon: %s  (%.1fs)", tuple(recon_neuron.shape), t_dec_cold)

    # Warm pass
    logger.info("Running traced encode+decode (warm)...")
    t0 = time.time()
    z2 = traced_encoder(x_patchified)
    recon2_p = traced_decoder(z2)
    t_warm = time.time() - t0
    logger.info("  warm round-trip: %.1fs", t_warm)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    T_cmp = min(x_in.shape[2], recon_neuron.shape[2])
    x_cmp = x_in[:, :, :T_cmp]
    r_cmp = recon_neuron[:, :, :T_cmp]

    mse = F.mse_loss(x_cmp, r_cmp).item()
    psnr = psnr_db(x_cmp, r_cmp)

    save_triptych(x_cmp.squeeze(0), r_cmp.squeeze(0), Path(args.output), T_cmp // 2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TORCH_NEURONX.TRACE RECONSTRUCTION SUMMARY")
    print("=" * 60)
    print(f"  Video            : {args.video}")
    print(f"  Resolution       : {args.resize_width}x{args.resize_height}")
    print(f"  Frames compared  : {T_cmp}")
    print(f"  Latent shape     : {tuple(z_neuron.shape)}")
    print(f"  ---")
    print(f"  CPU MSE          : {mse_cpu:.6f}")
    print(f"  CPU PSNR         : {psnr_cpu:.2f} dB")
    print(f"  Neuron MSE       : {mse:.6f}")
    print(f"  Neuron PSNR      : {psnr:.2f} dB")
    print(f"  ---")
    print(f"  Encoder trace    : {t_enc_trace:.1f}s")
    print(f"  Decoder trace    : {t_dec_trace:.1f}s")
    print(f"  Encode (cold)    : {t_enc_cold:.1f}s")
    print(f"  Decode (cold)    : {t_dec_cold:.1f}s")
    print(f"  Round-trip (warm): {t_warm:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
