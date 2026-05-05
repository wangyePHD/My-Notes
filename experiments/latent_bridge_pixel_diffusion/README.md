# Latent-Bridge Pixel Diffusion

Research prototype for testing the idea:

```text
noise -> low-resolution / semantic bridge -> high-resolution pixels
```

The code keeps the final image path in pixel space. The low-resolution bridge is a deterministic target derived from the same high-resolution image, not a VAE latent and not a decoder output. During sampling, the bridge is denoised first, then the high-resolution branch is denoised conditioned on the generated bridge.

## What This Prototype Implements

- High-resolution pixel patch diffusion with x-prediction.
- Deterministic low-resolution RGB bridge.
- Cascaded Latent Forcing schedule:
  - bridge denoises from noise to clean in the first phase;
  - high-resolution pixels denoise in the second phase.
- Optional residual prediction:
  - model predicts `x_hr - upsample(y_low)` instead of full high-resolution pixels.
- A small DiT-like Transformer with separate high-res and bridge output heads.
- Folder image dataset loader.

Semantic latents such as DINO/SigLIP are intentionally left as a clean extension point. Start with the low-res bridge first; it is easier to validate and isolates the high-resolution pixel/residual question.

## Install

```bash
cd experiments/latent_bridge_pixel_diffusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Example for a quick 256/64 smoke run:

```bash
python train.py \
  --data /path/to/images \
  --image-size 256 \
  --bridge-size 64 \
  --patch-size 16 \
  --batch-size 8 \
  --steps 10000 \
  --out runs/smoke_256
```

The same run can be launched from the provided config:

```bash
python train.py --config configs/smoke_256.yaml --data /path/to/images --out runs/smoke_256
```

Example for the intended 1K/256 experiment:

```bash
torchrun --nproc_per_node=8 train.py \
  --data /path/to/images \
  --image-size 1024 \
  --bridge-size 256 \
  --patch-size 16 \
  --dim 768 \
  --depth 12 \
  --heads 12 \
  --batch-size 1 \
  --grad-accum 8 \
  --amp bf16 \
  --predict-residual \
  --steps 200000 \
  --out runs/lbpd_1k_residual
```

Or with the provided config:

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/lbpd_1k_residual.yaml \
  --data /path/to/images \
  --out runs/lbpd_1k_residual
```

## Sample

```bash
python sample.py \
  --ckpt runs/lbpd_1k_residual/checkpoint_latest.pt \
  --num 16 \
  --steps 80 \
  --out samples/lbpd_1k
```

## First Ablations

Run these before scaling:

1. `--image-size 256 --bridge-size 64`, full pixel prediction.
2. `--image-size 512 --bridge-size 128`, full pixel prediction.
3. `--image-size 512 --bridge-size 128 --predict-residual`.
4. Change `--cascade-ratio` from `0.5` to `0.7`.
5. Disable bridge influence by setting `--bridge-loss-weight 0` only after adding a pure-pixel baseline.

The first signal to look for is not SOTA FID. Check whether the bridge/residual version reaches comparable FID faster and whether small text, boundaries, and high-frequency textures degrade less.
