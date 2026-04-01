# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Encode → decode smoke test for the frozen Wan 2.2 VAE (vae2_2).

Run from anywhere; the script loads ``wan/modules/vae2_2.py`` directly (no full ``wan`` import):

    cd examples/training/Wan2.2
    python scripts/test_vae2_2_reconstruction.py \\
        --vae-path /path/to/Wan2.2_VAE.pth --synthetic

Or pass a checkpoint directory that contains ``Wan2.2_VAE.pth`` (same layout as ``generate.py``):

    python scripts/test_vae2_2_reconstruction.py --checkpoint-dir /path/to/models

Constraints (see ``wan/modules/vae2_2.py``): spatial ``H`` and ``W`` must be even;
temporal length must satisfy ``T = 1 + 4*k`` (the script trims excess frames).
Pixels are expected in ``[-1, 1]``.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Wan2.2 tree root (parent of ``wan/``)
_WAN_ROOT = Path(__file__).resolve().parent.parent

# Load ``vae2_2`` by file path so we never import ``wan`` (``wan/__init__.py`` pulls in
# T5 and other modules that call ``torch.cuda`` at import time — breaks CPU/XLA hosts).
_VAE2_2_PATH = _WAN_ROOT / "wan" / "modules" / "vae2_2.py"
_spec = importlib.util.spec_from_file_location("wan_vae2_2_standalone", _VAE2_2_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load VAE module from {_VAE2_2_PATH}")
_vae2_2 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _vae2_2
_spec.loader.exec_module(_vae2_2)
Wan2_2_VAE = _vae2_2.Wan2_2_VAE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VAE_NAME = "Wan2.2_VAE.pth"


def _largest_valid_temporal(t: int) -> int:
    """Largest T' <= t with (T' - 1) % 4 == 0 (and T' >= 1)."""
    if t < 1:
        raise ValueError("need at least one frame")
    return 1 + ((t - 1) // 4) * 4


def _ensure_even_hw(x: torch.Tensor) -> torch.Tensor:
    """Crop (C, T, H, W) so H, W are even."""
    _, _, h, w = x.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 != h or w2 != w:
        x = x[:, :, :h2, :w2]
        logger.info("cropped spatial size to even H=%s W=%s", h2, w2)
    return x


def _trim_temporal(x: torch.Tensor) -> torch.Tensor:
    t = x.shape[1]
    t_good = _largest_valid_temporal(t)
    if t_good < t:
        logger.info("trimming time: %s -> %s frames (VAE temporal grouping)", t, t_good)
        x = x[:, :t_good]
    return x


def synthetic_video(c: int, t: int, h: int, w: int, device: str, seed: int) -> torch.Tensor:
    # torch.Generator only supports CPU/CUDA/MPS — not XLA. Sample on CPU, then move.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(c, t, h, w, generator=g, dtype=torch.float32) * 0.2
    if device != "cpu":
        x = x.to(device)
    return x


def load_video_tensor(
    path: str,
    device: str,
    max_frames: int | None,
    resize_hw: tuple[int, int] | None,
) -> torch.Tensor:
    try:
        from torchvision.io import read_video
    except ImportError as e:
        raise RuntimeError("torchvision is required for --video (see requirements.txt)") from e

    video, _, _ = read_video(path, output_format="TCHW")
    if video.numel() == 0:
        raise RuntimeError(f"no frames read from {path}")
    # uint8 TCHW -> float [-1, 1] CTHW
    x = video.float() / 255.0
    x = x * 2.0 - 1.0
    x = x.permute(1, 0, 2, 3).contiguous()
    if max_frames is not None and x.shape[1] > max_frames:
        x = x[:, :max_frames]
    if resize_hw is not None:
        th, tw = resize_hw
        # (1, C, T, H, W) for 3D interpolate
        y = x.unsqueeze(0)
        y = F.interpolate(y, size=(x.shape[1], th, tw), mode="trilinear", align_corners=False)
        x = y.squeeze(0)
    x = x.to(device)
    return _ensure_even_hw(_trim_temporal(x))


def psnr_db(x: torch.Tensor, y: torch.Tensor) -> float:
    """PSNR assuming values in [-1, 1] (peak squared error 4)."""
    mse = F.mse_loss(x, y).clamp(min=1e-10)
    return float((10.0 * torch.log10(torch.tensor(4.0, device=mse.device) / mse)).item())


def save_triptych(
    inp: torch.Tensor,
    rec: torch.Tensor,
    out_path: Path,
    frame_idx: int,
    diff_gain: float = 4.0,
) -> None:
    """Save input | reconstruction | amplified abs diff for one frame."""
    inp = inp[:, frame_idx].clamp(-1, 1).detach().cpu()
    rec = rec[:, frame_idx].clamp(-1, 1).detach().cpu()

    def to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
        u8 = ((t + 1.0) * 0.5 * 255.0).byte().permute(1, 2, 0).numpy()
        return u8

    a = to_rgb_uint8(inp)
    b = to_rgb_uint8(rec)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wan 2.2 VAE encode/decode reconstruction test")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--vae-path",
        type=str,
        help=f"path to {DEFAULT_VAE_NAME} (or other compatible 2.2 VAE weights)",
    )
    src.add_argument(
        "--checkpoint-dir",
        type=str,
        help=f"directory containing {DEFAULT_VAE_NAME} (same as generate.py model folder)",
    )

    p.add_argument(
        "--synthetic",
        action="store_true",
        help="use random noise (C,T,H,W) instead of a video file",
    )
    p.add_argument("--video", type=str, default=None, help="video file path (requires torchvision)")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames when loading --video")
    p.add_argument("--resize-height", type=int, default=None, help="resize loaded video to this H")
    p.add_argument("--resize-width", type=int, default=None, help="resize loaded video to this W")

    p.add_argument("--channels", type=int, default=3, help="channels for --synthetic (use 3 for RGB)")
    p.add_argument("--frames", type=int, default=17, help="T for --synthetic (should be 1+4k)")
    p.add_argument("--height", type=int, default=256, help="H for --synthetic (even)")
    p.add_argument("--width", type=int, default=256, help="W for --synthetic (even)")

    p.add_argument("--output", type=str, default="vae2_2_recon_triptych.png", help="output image path")
    p.add_argument("--frame-index", type=int, default=-1, help="which frame to visualize (default: middle)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --synthetic")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu | xla (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=("bfloat16", "float32"),
        help="autocast dtype for VAE (default: bfloat16 on CUDA/XLA, float32 on CPU)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_dir:
        vae_path = Path(args.checkpoint_dir) / DEFAULT_VAE_NAME
    else:
        vae_path = Path(args.vae_path)
    if not vae_path.is_file():
        raise FileNotFoundError(f"VAE weights not found: {vae_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Import torch_xla early — before any tensor is placed on an XLA device.
    # torch_xla registers the "xla" backend with PyTorch; without this import
    # calls like tensor.to("xla") or torch.tensor(..., device="xla") will fail.
    if str(device).split(":")[0] == "xla":
        import torch_xla
        import torch_xla.core.xla_model as xm

        # Resolve the canonical XLA device string (e.g. "xla:0").
        device = str(xm.xla_device())
        logger.info("XLA device: %s", device)

    if args.dtype is None:
        vae_dtype = (
            torch.bfloat16 if str(device).split(":")[0] in ("cuda", "xla") else torch.float32
        )
    else:
        vae_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    if args.synthetic == bool(args.video):
        raise SystemExit("specify exactly one of --synthetic or --video")

    if args.synthetic:
        t, h, w = args.frames, args.height, args.width
        if h % 2 or w % 2:
            raise SystemExit("--height and --width must be even (patchify)")
        t_good = _largest_valid_temporal(t)
        if t_good != t:
            logger.info("adjusting --frames %s -> %s", t, t_good)
            t = t_good
        x = synthetic_video(args.channels, t, h, w, device, args.seed)
    else:
        resize = None
        if args.resize_height is not None or args.resize_width is not None:
            if args.resize_height is None or args.resize_width is None:
                raise SystemExit("use both --resize-height and --resize-width or neither")
            resize = (args.resize_height, args.resize_width)
            if resize[0] % 2 or resize[1] % 2:
                raise SystemExit("resize height/width must be even")
        x = load_video_tensor(args.video, device, args.max_frames, resize)
        if x.shape[0] != 3:
            raise SystemExit(f"expected 3 RGB channels, got C={x.shape[0]}")

    logger.info("input tensor shape (C,T,H,W) = %s", tuple(x.shape))

    vae = Wan2_2_VAE(vae_pth=str(vae_path), device=device, dtype=vae_dtype)

    # On XLA/Trainium the entire encode+decode loop accumulates into one monolithic
    # lazy graph before the compiler sees it.  For the full VAE that graph exceeds
    # Trainium2's per-executable instruction limit (NCC_EBVF030: 22.6M > 5M).
    #
    # Fix: pass xm.mark_step as step_fn to WanVAE_.encode/decode.  It is called
    # after every encoder/decoder chunk call and after every torch.cat, flushing
    # each chunk as a small, independently-compiled graph (~constant size regardless
    # of video length).
    #
    # inference_mode is too strict for the Wan2.2 VAE: the internal feat_cache
    # lists are mutated via clone() inside the encode/decode loops, which
    # requires version-counter updates that inference_mode disallows.
    # no_grad() gives the same memory/speed benefit without that restriction.
    is_xla = str(device).split(":")[0] == "xla"
    if is_xla:
        import torch_xla.core.xla_model as xm
        step_fn = xm.mark_step
        logger.info("XLA mode: using xm.mark_step() between encode/decode chunks")
    else:
        step_fn = None

    with torch.no_grad():
        z = vae.model.encode(x.unsqueeze(0), vae.scale, step_fn=step_fn)
        z = z.squeeze(0)
        if is_xla:
            xm.mark_step()
        x_hat = vae.model.decode(z.unsqueeze(0), vae.scale, step_fn=step_fn)
        x_hat = x_hat.squeeze(0).clamp(-1, 1)

    if is_xla:
        xm.mark_step()
        # Materialize on CPU for scalar metrics and PIL (avoids extra lazy graphs on XLA).
        x = x.cpu()
        x_hat = x_hat.cpu()

    mse = F.mse_loss(x_hat, x).item()
    psnr = psnr_db(x_hat, x)
    logger.info("latent shape (C,T,H,W) = %s", tuple(z.shape))
    logger.info("reconstruction MSE = %.6f  PSNR ≈ %.2f dB", mse, psnr)

    t = x.shape[1]
    fidx = args.frame_index if args.frame_index >= 0 else t // 2
    fidx = max(0, min(fidx, t - 1))
    out = Path(args.output)
    save_triptych(x, x_hat, out, fidx)


if __name__ == "__main__":
    main()
