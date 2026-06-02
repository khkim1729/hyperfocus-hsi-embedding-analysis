# Spectral Foundation Model v71 — Usage Guide (English)

> 한국어 사용 가이드는 [`README.md`](./README.md)를 참고하세요.

## Overview

This bundle is the **pretrained encoder** of the domain-independent Spectral
Foundation Model (v71), trained via a Masked Autoencoder (MAE) objective on four
hyperspectral datasets — NASA EO-1, NASA AVIRIS, NEON, and Spectral Earth — and
designed to produce general-purpose spectral representations across sensors with
70–200+ bands.

The encoder maps a 1-D spectrum `[B, band_dim]` to a fixed-size representation
`[B, 128]` that can be used as input features for downstream tasks (classification,
regression, segmentation) via linear probing or full fine-tuning.

## Files

| File | Purpose |
|------|---------|
| `checkpoint_best.pth` | Pretrained model weights (v71) |
| `config.yaml` | Pretraining hyperparameters |
| `spectral_foundation_v2.py` | Encoder architecture (`SpectralEncoderV2` + `SingleFrequencySpectralEmbedding`) |
| `load_model.py` | Loader utility (`load_encoder(...)`) |
| `README.md` | Original Korean usage guide |
| `USAGE.md` | This file |

The bundle is **self-contained** — the only third-party dependencies are
`torch` and `pyyaml`. Nothing imports from the `hyper-focus` repo.

## Model spec

| Item | Value |
|------|-------|
| Parameters | 1,586,432 (~1.59 M) |
| Embedding dim | 128 |
| Transformer layers | 8 |
| Attention heads | 16 |
| FFN hidden dim | 512 (`embed_dim × 4`) |
| Default band count | 80 |
| Positional encoding | Wavelength-aware sinusoidal |
| Output shape | `[B, 128]` |

## Requirements

```
torch >= 2.0
pyyaml
```

## Quickstart

```python
import sys
sys.path.insert(0, "/data/chlee/hyperfocus_v71")

from load_model import load_encoder

encoder = load_encoder(device="cuda")  # band_dim defaults to 80
```

The loader strips DDP / MAE-wrapper prefixes (`encoder.`, `_wl_tensor_*`)
automatically and returns the encoder in `eval()` mode.

## Loading with a custom band count

The bandwise embedding (`nn.Linear(1, 128)`) is independent of band count, so
you can re-instantiate the encoder for any spectrum length and the pretrained
weights still load.

```python
# Indian Pines (200 bands)
encoder = load_encoder(band_dim=200, device="cuda")

# Urban (162 bands)
encoder = load_encoder(band_dim=162, device="cuda")
```

`load_state_dict` is called with `strict=False`; the loader prints any
missing/unexpected keys, which is expected when `band_dim` differs from 80.

## Passing real wavelengths (recommended)

Positional encoding is **wavelength-aware**, so passing the actual wavelengths
of your sensor yields better representations than the default linear
interpolation between 400 nm and 2500 nm.

```python
import torch
from load_model import load_encoder

# Real wavelengths in nanometers, one per band
wavelengths = torch.tensor([400.0, 410.0, 420.0, ...])  # length == band_dim

encoder = load_encoder(
    band_dim=len(wavelengths),
    wavelengths=wavelengths,
    device="cuda",
)
```

## Feature extraction

```python
import torch
from load_model import load_encoder

encoder = load_encoder(device="cuda")
encoder.eval()

# x: [B, band_dim] float32, z-score-normalized (see "Input expectations" below)
x = torch.randn(256, 80, device="cuda")

with torch.no_grad():
    features = encoder(x)  # [256, 128]
```

For batched inference over a large array, iterate in chunks and accumulate on
CPU:

```python
def extract_features(encoder, X, batch_size=512, device="cuda"):
    encoder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(encoder(batch).cpu())
    return torch.cat(outs, dim=0)  # [N, 128]
```

## Linear probing (frozen encoder + linear head)

```python
import torch
import torch.nn as nn
from load_model import load_encoder

encoder = load_encoder(band_dim=200, device="cuda")
for p in encoder.parameters():
    p.requires_grad = False
encoder.eval()

classifier = nn.Linear(128, num_classes).cuda()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

# Inside training loop:
with torch.no_grad():
    feats = encoder(x)         # [B, 128]
logits = classifier(feats)     # [B, num_classes]
loss = nn.functional.cross_entropy(logits, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## Fine-tuning (unfrozen encoder)

Use a lower learning rate than linear probing — the encoder is already
pretrained, and large gradients early on can wipe out useful representations.

```python
encoder = load_encoder(band_dim=200, device="cuda")
classifier = nn.Linear(128, num_classes).cuda()

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(classifier.parameters()),
    lr=1e-4,
)

encoder.train()
classifier.train()
# usual forward/backward/step loop
```

## Input expectations

- **Shape**: `[B, band_dim]` float32 tensor.
- **Normalization**: Per-band z-score (mean ≈ 0, std ≈ 1). v71 was pretrained
  with 1st–99th percentile outlier clipping applied **before** z-score
  normalization (see `data.normalization` in `config.yaml`). For best results,
  apply the same recipe to your downstream data:
  1. Compute 1st and 99th percentile per band on the **training split only**.
  2. Clip every spectrum to those bounds.
  3. Subtract per-band mean and divide by per-band std (computed on the
     training split after clipping).
  4. Reuse those statistics on validation / test splits — do not recompute
     them on held-out data.

  Per-band stats are **not** stored in the v71 checkpoint; they are a property
  of the downstream dataset and must be computed locally.

## Verification

Confirm the bundle loads cleanly:

```bash
cd /data/chlee/hyperfocus_v71
python load_model.py
```

Expected output (Korean string is from the original loader):

```
v71 인코더 로딩 중...
인코더 로드 완료: 1,586,432 파라미터
입력:  torch.Size([4, 80])
출력: torch.Size([4, 128])
검증 통과!
```

## Architecture

```
Input spectrum [B, band_dim]
  → Linear(1, 128)           # band-wise independent embedding → [B, band_dim, 128]
  → + wavelength-aware sinusoidal positional encoding
  → TransformerEncoder (8 layers, 16 heads, FFN=512)
  → Mean pooling over bands
  → Output representation [B, 128]
```

## Pretraining provenance

- **Source repo**: `/data/chlee/hyper-focus/`
- **Original checkpoint dir**:
  `/data/chlee/hyper-focus/artifacts/refactored_experiments/checkpoints/20260116_0543_SF_v71/`
- **Release bundle (source of this copy)**:
  `/data/chlee/hyper-focus/released_models/v71/`
- **Training datasets**: NASA EO-1, NASA AVIRIS, NEON, Spectral Earth
  (10×10 patches, "nobel_score" variants; sample sizes in `config.yaml`)
- **Training objective**: MAE with `mask_ratio=0.75`, block masking,
  contrastive auxiliary loss (weight 0.1), MSE + SAM + gradient + frequency
  reconstruction losses (1.0 / 0.3 / 0.2 / 0.1)
- **Training schedule**: 600 epochs, batch size 256, LR 1e-4, AdamW
  (weight decay 0.05), CosineAnnealingWarmup (10 warmup epochs),
  gradient clipping at 1.0
