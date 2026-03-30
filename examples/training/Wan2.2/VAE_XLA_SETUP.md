# Wan 2.2 VAE Reconstruction Test — XLA / Trainium Setup Guide

End-to-end steps to run `scripts/test_vae2_2_reconstruction.py` on an AWS
Trainium instance (TRN2) using PyTorch/XLA via `torch-neuronx`.

---

## 1. Instance requirements

| Item | Requirement |
|---|---|
| Instance type | `trn2.48xlarge` (or any `trn1`/`trn2` family) |
| OS | Ubuntu 22.04 / 24.04 (Neuron-enabled AMI) |
| Neuron driver | ≥ 2.26 (pre-installed on Neuron AMI) |
| Python | 3.10 – 3.12 |

---

## 2. Create a virtual environment

```bash
python3 -m venv ~/wan_vae
source ~/wan_vae/bin/activate
```

> **Always activate the venv before any command below.**

---

## 3. Install Neuron packages

Install `torch-neuronx` and the matching `torch-xla` from the AWS Neuron PyPI
index.  The versions below were tested and confirmed working:

| Package | Version |
|---|---|
| `torch` | 2.9.1 |
| `torch-neuronx` | 2.9.0.2.12.22436 |
| `torch-xla` | 2.9.0 |
| `libneuronxla` | 2.2.15515.0 |
| `neuronx-cc` | 2.23.6484.0 |

```bash
pip install --upgrade pip

# Neuron index (adjust URL for your Neuron SDK release)
pip install torch-neuronx==2.9.* \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
```

---

## 4. Install VAE-only Python dependencies

The full `requirements.txt` pulls in flash-attn and other heavy packages that
are not needed for the VAE test.  Install only what the test actually uses:

```bash
pip install einops pillow numpy torch torchvision
```

---

## 5. Get the VAE weights

The weights file is `Wan2.2_VAE.pth` (~800 MB), hosted on the Hugging Face Hub
under `Wan-AI/Wan2.2-TI2V-5B`.

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    filename="Wan2.2_VAE.pth",
    local_dir="/home/ubuntu/wan_weights",
)
print("saved to", path)
EOF
```

Or with the CLI:
```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir /home/ubuntu/wan_weights
```

---

## 6. Run the test

```bash
cd /home/ubuntu/optimum-neuron/examples/training/Wan2.2

python scripts/test_vae2_2_reconstruction.py \
    --vae-path /home/ubuntu/wan_weights/Wan2.2_VAE.pth \
    --synthetic \
    --frames 17 --height 256 --width 256 \
    --device xla \
    --dtype bfloat16
```

**First run** triggers Neuron compilation (~5–10 min depending on instance).
Subsequent runs load from cache (`/var/tmp/neuron-compile-cache/`) in seconds.

**Expected output (synthetic noise):**

```
INFO: XLA device: xla:0
INFO: input tensor shape (C,T,H,W) = (3, 17, 256, 256)
INFO: loading /home/ubuntu/wan_weights/Wan2.2_VAE.pth
INFO: latent shape (C,T,H,W) = (48, 5, 16, 16)
INFO: reconstruction MSE = 1.039734  PSNR ≈ 5.85 dB
INFO: wrote vae2_2_recon_triptych.png
```

> Low PSNR on synthetic noise is expected — the VAE is trained on natural
> video.  For a quality check pass a real video with `--video clip.mp4`;
> expect 25–35 dB PSNR on natural content.

### Input constraints

| Constraint | Detail |
|---|---|
| Spatial dims | H and W must be **even** (patchify requirement) |
| Temporal length | Must satisfy `T = 1 + 4k`; the script trims excess frames automatically |
| Channels | 3 (RGB) |
| Pixel range | `[-1, 1]` (the script normalises loaded video automatically) |

---

## 7. Debugging: known errors and fixes

### 7.1  `ModuleNotFoundError: No module named 'torch'`

The venv is not active.

```bash
source ~/wan_vae/bin/activate
```

---

### 7.2  `RuntimeError: Init: … NRT_FAILURE … Logical Neuron Core(s) not available`

Another process is holding the Neuron cores (e.g., a previous test that was
killed without releasing the device).

```bash
# Find and kill the stale process
ps aux | grep python
kill <PID>

# Also remove any stale compiler lock files
find /var/tmp/neuron-compile-cache -name "*.lock" -delete
```

---

### 7.3  `RuntimeError: Cannot set version_counter for inference tensor`

Occurs inside `torch.inference_mode()` on XLA when:

- **`torch.autocast`** is used — XLA raises this on entry.
  *Fix in `vae2_2.py`:* replaced with `contextlib.nullcontext()` for XLA.

- **`.view()` or `.to(dtype)` without `copy=True`** produces a view that
  shares storage with a regular tensor; entering inference mode then tries to
  set the version counter on the view.
  *Fix:* pre-shape scale tensors to 5-D in `__init__`; use `tensor.to(dtype, copy=True)`.

- **In-place ops (`.clamp_()`)** on tensors that may be views.
  *Fix:* use out-of-place ops (`.clamp()`).

---

### 7.4  `RuntimeError: Value out of range … got -2`

XLA rejects negative slice indices when the dimension is smaller than the
absolute value of the index (e.g., `x[:, :, -2:, :, :]` on a T=1 tensor).

*Fix in `vae2_2.py`:*
```python
# Before (breaks on XLA when x.shape[2] < 2):
cache_x = x[:, :, -CACHE_T:, :, :].clone()

# After:
cache_x = x[:, :, max(0, x.shape[2] - CACHE_T):, :, :].clone()
```

---

### 7.5  `ValueError: … [NCC_IBIR158] Access pattern out of bounds` / `[NCC_ITEN404] DramToDramTranspose`

**Root cause:** `neuronx-cc` has an internal compiler bug triggered by
convolution weights with exactly **12 input or output channels**.  The Wan 2.2
VAE's `patchify` converts 3 RGB channels → 12 channels (2×2 spatial patches),
feeding `CausalConv3d(12, 160, 3)` — the encoder's first layer.  The
compiler's SBUF tiling algorithm cannot handle this channel count on TRN2.

**Fix applied in `vae2_2.py`** (`_pad_12_to_16_channels`):

After loading weights, zero-pad the two boundary convolutions from 12 → 16
channels (a power-of-two that aligns with Neuron's SBUF):

```python
# encoder.conv1  [160, 12, kT, kH, kW] → [160, 16, kT, kH, kW]
# decoder.head[-1]  [12, dim, ...] → [16, dim, ...]  (weight + bias)
```

Correspondingly the encode input is padded from C=3 to C=4 (zero channel),
so patchify produces 16 channels; the decode output is clipped back to C=3.
Mathematical equivalence is preserved — the extra channels are all zeros.

---

### 7.6  Long compilation hang (35+ min, process appears stuck)

All encode/decode iterations accumulate into a single enormous XLA lazy graph
that takes forever to compile.

*Fix:* call `xm.mark_step()` after each encode/decode chunk iteration to flush
the lazy graph incrementally.  In `vae2_2.py` this is done via a `step_fn`
argument injected from `Wan2_2_VAE.__init__` when the device is XLA.

---

### 7.7  Stale compiler lock files blocking compilation

If a previous process was killed mid-compile, lock files remain and the next
run waits indefinitely printing:

```
Another process must be compiling …, been waiting for: N.N minutes
```

```bash
find /var/tmp/neuron-compile-cache -name "*.lock" -delete
```

---

## 8. Compilation cache

Compiled `.neff` files are cached in `/var/tmp/neuron-compile-cache/`.
On a cache hit the test completes in ~30 s instead of ~10 min.

To force recompilation (e.g. after changing the model or compiler):

```bash
rm -rf /var/tmp/neuron-compile-cache/
```

---

## 9. Key files

| File | Purpose |
|---|---|
| `wan/modules/vae2_2.py` | VAE implementation (Wan2_2_VAE, WanVAE_, encoder/decoder) |
| `scripts/test_vae2_2_reconstruction.py` | Encode→decode smoke test |
| `/var/tmp/neuron-compile-cache/` | Neuron `.neff` compilation cache |
| `/home/ubuntu/wan_weights/Wan2.2_VAE.pth` | Pretrained VAE weights |
