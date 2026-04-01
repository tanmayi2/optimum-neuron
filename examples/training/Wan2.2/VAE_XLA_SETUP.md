# Wan 2.2 VAE — XLA / Trainium2 Setup Guide

This document covers installing the `wan_vae` environment, downloading the VAE checkpoint,
and running the encode → decode reconstruction smoke test on AWS Trainium2 via `torch_xla`.

---

## Hardware

The setup targets **trn2.3xlarge** (or larger trn2/trn2n instances):

| Property | Value |
|---|---|
| NeuronDevice | Trainium2 |
| NeuronCores | 4 (IDs 0–3) |
| HBM | 96 GB |
| Host RAM | 124 GB |
| Neuron SDK | 2.x (`neuronx-cc 2.23+`, `torch-neuronx 2.9+`) |

---

## 1. Virtual Environment

A Python 3.12 venv named `wan_vae` is used so the Wan2.2 dependencies are isolated
from the system Neuron venvs in `/opt/`.

```bash
python3 -m venv ~/wan_vae
source ~/wan_vae/bin/activate
```

Install PyTorch + torch_xla from the Neuron package index, then the remaining deps:

```bash
# Neuron-flavoured torch/xla (matches the compiler version on the instance)
pip install \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  torch-neuronx==2.9.0.* torchvision torch-xla

# VAE-specific extras (no flash_attn / full model deps needed)
pip install einops imageio imageio-ffmpeg opencv-python-headless \
            huggingface_hub av easydict Pillow
```

> **Note**: `av` (PyAV) is required by `torchvision.io.read_video`.
> The `wan` package is **not** pip-installed — `vae2_2.py` is loaded directly by
> the test script to avoid `wan/__init__.py` pulling in T5/diffusers at import time.

---

## 2. Checkpoint

Download just the VAE weights from the Wan2.2-TI2V-5B HuggingFace repo (~2.7 GB):

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    filename="Wan2.2_VAE.pth",
    local_dir="/home/ubuntu/Wan2.2-VAE",
)
EOF
```

The file will be saved to `/home/ubuntu/Wan2.2-VAE/Wan2.2_VAE.pth`.

---

## 3. Running the reconstruction test

The test script lives at `scripts/test_vae2_2_reconstruction.py`.
Run from the `examples/training/Wan2.2/` directory with the venv active **and**
`wan_vae/bin` on `PATH` (so `libneuronpjrt-path` is found by the torch_xla init):

```bash
cd examples/training/Wan2.2

# --- CPU baseline (always works, gives reference metrics) ---
PATH="~/wan_vae/bin:$PATH" \
python scripts/test_vae2_2_reconstruction.py \
  --vae-path ~/Wan2.2-VAE/Wan2.2_VAE.pth \
  --video /path/to/input.mp4 \
  --max-frames 17 \
  --resize-height 480 \
  --resize-width  848 \
  --device cpu \
  --output recon_cpu.png

# --- XLA / Trainium2 (small input compiles successfully) ---
PATH="~/wan_vae/bin:$PATH" \
NEURON_CC_FLAGS="--optlevel 1 --model-type generic" \
python scripts/test_vae2_2_reconstruction.py \
  --vae-path ~/Wan2.2-VAE/Wan2.2_VAE.pth \
  --synthetic \
  --frames 1 \
  --height 64 \
  --width  64 \
  --device xla \
  --output recon_xla_synthetic.png
```

> For long-running compilations use `systemd-run --user` (see §5) to decouple the
> process from the terminal session.

### Key CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--device xla` | — | Selects Trainium2 via torch_xla |
| `--dtype bfloat16` | bf16 on XLA | Halves graph size vs float32 |
| `--max-frames N` | — | Must satisfy `1 + 4k` (VAE temporal grouping) |
| `--resize-height H` | — | Must be even; spatial stride is 16× total |
| `--resize-width W` | — | Must be even |
| `--synthetic` | — | Random noise input, no video file needed |

---

## 4. XLA-specific patches applied to `vae2_2.py`

Two changes were required to make the VAE forward pass XLA-compatible:

### 4a. Negative slice index clamping (`_cache_slice`)

XLA enforces strict bounds on negative slice indices.
`x[:, :, -CACHE_T:, :, :]` on a length-1 time axis raises:

```
RuntimeError: Value out of range (expected to be in range of [-1, 0], but got -2)
```

CPU/CUDA silently clamp out-of-range negative starts to 0; XLA does not.
Fix: a helper function that computes a non-negative start index before slicing:

```python
def _cache_slice(x: torch.Tensor) -> torch.Tensor:
    t = x.shape[2]
    start = t - min(CACHE_T, t)   # always >= 0
    return x[:, :, start:, :, :]
```

All six occurrences of `x[:, :, -CACHE_T:, :, :].clone()` in `vae2_2.py` are
replaced with `_cache_slice(x).clone()`.

### 4b. `inference_mode` → `no_grad` in the test script

`torch.inference_mode()` disallows version-counter updates.
The VAE's internal `feat_cache` lists are mutated (via `.clone()`) inside the
encode/decode loops, which requires version tracking.
Wrapping the encode/decode in `torch.no_grad()` instead gives the same
memory/speed benefit without that restriction.

---

## 5. Surviving terminal restarts

First-time compilation takes 10–30 minutes and will be killed if the terminal
session (tmux/SSH) dies.  Use a systemd transient service:

```bash
systemd-run --user \
  --unit=wan-vae-xla \
  --working-directory=/path/to/Wan2.2 \
  --setenv=PATH=/home/ubuntu/wan_vae/bin:/usr/bin:/bin \
  "--setenv=NEURON_CC_FLAGS=--optlevel 1 --model-type generic" \
  --property=StandardOutput=file:/tmp/vae_xla.log \
  --property=StandardError=file:/tmp/vae_xla.log \
  /home/ubuntu/wan_vae/bin/python scripts/test_vae2_2_reconstruction.py \
    --vae-path ~/Wan2.2-VAE/Wan2.2_VAE.pth ...

# Monitor
journalctl --user -u wan-vae-xla -f
```

---

## 6. Known compilation limits (Trainium2 trn2.3xlarge)

| Input size | Frames | Compile result | Notes |
|---|---|---|---|
| 64×64 | 1 | **PASS** | ~few minutes, NEFF ~small |
| 480×848 | 5 | **FAIL** NCC_EBVF030 | 22.6M instrs > 5M limit |
| 480×848 | 17 | **FAIL** OOM | 11 workers × 33 GB = 363 GB |

The root cause is XLA lazy evaluation: all encode/decode loop iterations accumulate
into **one monolithic computation graph** before the compiler sees it.
Trn2.3xlarge enforces a 5M-instruction limit per executable.

See §7 for the roadmap to fix this.

---

## 7. Next optimization steps

### 7a. Graph chunking with `xm.mark_step()` (highest priority)

Insert `xm.mark_step()` after each iteration of the encode and decode loops.
This forces XLA to compile and dispatch each chunk as a small, independently-compilable
graph rather than one giant one:

```python
# encode loop — after each encoder call
for i in range(iter_):
    out_chunk = self.encoder(x_chunk, ...)
    xm.mark_step()   # <-- compile & execute this chunk
    out = torch.cat([out, out_chunk], 2)
```

Each chunk's graph is then a single encoder forward pass (~constant size regardless
of video length), well within the 5M instruction limit.

### 7b. `bfloat16` dtype

Using `--dtype bfloat16` halves the number of constants in the HLO graph and
typically reduces compiled instruction count by 30–50%.  The VAE's `Upsample`
layer already has a `x.float()` guard for nearest-neighbour interpolation, so
bfloat16 is architecturally safe.

### 7c. Static-shape tracing with `torch_neuronx.trace`

Pre-trace the encoder and decoder with fixed shapes using `torch_neuronx.trace`.
Traced NEFFs are cached and reused; no runtime compilation overhead:

```python
import torch_neuronx
encoder_traced = torch_neuronx.trace(vae.model.encoder, example_input)
```

This requires the encoder/decoder to be called with static shapes at every invocation
(no dynamic temporal concatenation).

### 7d. Tensor parallelism across NeuronCores

Use `neuronx_distributed` to shard the weight matrices across the 4 NeuronCores.
This reduces per-core instruction count and memory pressure proportionally.
Refer to the [NxD training guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/).

### 7e. Smaller spatial resolution

The 16× spatial compression ratio means 480×848 → 30×53 latent.
Halving to 240×424 (still divisible by 16) reduces spatial ops by 4×, which
would drop the instruction count below the 5M limit for the full video.

### 7f. Separate first-chunk and subsequent-chunk graphs

The encode loop has two structurally different paths:
- `i == 0`: processes 1 frame (no cache)
- `i > 0`: processes 4 frames (with cache)

Compiling these as two separate static-shape NEFFs eliminates the dynamic branching
that forces a large unified graph.
