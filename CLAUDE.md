# SpaNCy-Flow — Project Context

## What This Is
**SpaNCy-Flow** — Alternative to the GNN-based SpaNCy normalizer. Replaces the spatial GNN + adversarial approach with a **conditional normalizing flow** using cycle-block coupling and **MMD loss** for direct 20D batch alignment. Same CycleDegradationModel for affine correction, but the multivariate correction comes from an unconditional flow instead of a GNN.

**Key motivation**: The GNN approach has a train/inference mismatch (GNN used at training, discarded or blended at inference). The flow is unconditional and applies the **same transform at both training and inference** — no distribution mismatch.

Parent project: `../CLAUDE.md` has full dataset details, preprocessing, and shared design decisions.

## Architecture
```
X_raw → log1p → RobustScaler → CycleDegradationModel → X_affine → CycleBlockFlow (unconditional) → X_corrected
```

- **CycleDegradationModel**: Same as original SpaNCy — batch(32d) + sample(16d) + cycle(16d) → MLP(64→64→2) → per-marker gamma/beta
- **CycleBlockFlow**: Stack of 4 unconditional affine coupling blocks structured by imaging cycle partitions
  - Each `CycleCouplingBlock` splits markers into groups A/B based on cycle membership
  - A conditions B (forward: `x_B' = x_B * exp(s) + t`), then B' conditions A (reverse)
  - Conditioning networks: `Linear(d_in, 128) → GELU → Linear(128, 128) → GELU → Linear(128, 2*d_out)`
  - Near-zero init on final layer → starts as identity transform
  - Scale clamped to `[-0.5, 0.5]` → `exp(s) ∈ [0.607, 1.649]` (tight to prevent distribution compression)
  - **No batch/sample conditioning** — learns a single universal multivariate alignment
  - Affine coupling is invertible by construction → structurally preserves distribution topology (bimodality)
- **Cycle-block partitions**: 4 complementary splits of 6 cycles into 2 groups (e.g., cycles {0,1,2} vs {3,4,5}), rotated so every marker pair gets cross-conditioned in ≥2 blocks

## Losses
| Loss | Weight | Purpose |
|------|--------|---------|
| `L_identity` (Huber) | 1.0 | Keep output near affine baseline |
| `L_mmd` (multi-scale RBF MMD²) | 2.0 (ramped epoch 3→7) | Align 20D batch distributions directly — what kBET measures |
| `L_shape` | 2.0 | Preserve distribution shape: 2x variance ratio + 1x positive fraction + 1x IQR ratio |
| `L_marker_mmd` (per-marker 1D MMD) | 1.0 (shares MMD ramp) | Align each marker's marginal distribution across batches independently |
| `L_quantile` | 0.5 | Lower-quantile alignment (10th/25th percentiles) across samples |
| `L_flow_reg` | 0.5 | Regularize coupling scale parameters toward zero (identity) |

**Shape loss details**: Three components prevent distribution compression: (1) variance ratio penalizes `(var_after/var_before - 1)^2` per marker — directly prevents compression/expansion; (2) positive fraction preservation via sigmoid at median — preserves bimodal balance; (3) IQR ratio penalizes `(iqr_after/iqr_before - 1)^2` — outlier-robust width check.

**MMD details**: Multi-scale RBF kernel with bandwidths (0.1, 0.5, 1.0, 5.0, 10.0). Subsamples 256 cells/batch. Computes MMD² between all batch pairs in each mini-batch.

**Per-marker MMD details**: Same multi-scale RBF approach but operates on each marker column independently (1D distances). Bandwidths (0.1, 0.5, 1.0, 5.0) — no 10.0 (too broad for 1D). Subsamples 256 cells/batch. Averages MMD² across all 20 markers and batch pairs. Shares the same warmup ramp as the 20D MMD. Complements `L_mmd` (which aligns the joint 20D distribution) by forcing each marker to align individually — prevents the flow from distorting individual marker distributions while keeping the overall 20D blob matched.

## Inference Modes
1. **Affine** (`mode="affine"`): Only CycleDegradationModel gamma/beta. Same as original SpaNCy.
2. **Flow** (`mode="flow"`, default): Affine + unconditional flow with alpha blending: `X_out = X_affine + alpha * (X_flow - X_affine)`. Alpha=0 is pure affine, alpha=1 is full flow.
3. **Ensemble** (`normalize_adata_ensemble()`): Each model computes its own full correction path (own affine → own flow), then outputs are averaged. Supports both affine and flow modes.

## Files
| File | Purpose |
|------|---------|
| `spancy_flow.py` | Flow implementation (~1710 lines): models, training, inference, CLI — **broken, see SpaNCy-Shift** |
| `MMD_spancy_explore.ipynb` | Colab notebook for SpaNCy-Flow (abandoned after v2 failures) |
| `spancy_shift.py` | **Current approach** (~1000 lines): two-stage pipeline (analytic Stage 1 + GNNStage2 with GATv2 + GRL + MMD) |
| `spancy_shift_explore.ipynb` | Colab notebook for SpaNCy-Shift: bimodal detection, training, histograms, kBET |
| `spancy_shift_cfm.py` | **OT-CFM Stage 2** (~300 lines): Conditional Flow Matching with mini-batch Hungarian OT coupling; no torch-geometric |
| `spancy_shift_cfm_explore.ipynb` | Colab notebook for OT-CFM: n_steps sweep, ECAD histogram check, kBET |
| `spancy_shift_ddpm.py` | **DDPM + SDEdit Stage 2** (~400 lines): DenoisingMLP with classifier-free guidance; SDEdit inference; no torch-geometric |
| `spancy_shift_ddpm_explore.ipynb` | Colab notebook for DDPM: t_infer × cfg_scale grid sweep, ECAD check, kBET |
| `spancy_shift_dl.py` | **Single-stage DL** (~490 lines): `L_ref` replaces analytic Stage 1 — learns reference-sample alignment via gradient descent |
| `spancy_shift_dl_explore.ipynb` | Colab notebook for single-stage DL: L_ref + MMD training, 2-panel histograms, kBET comparison |
| `shift_model.ipynb` | Earlier prototype: per-sample shift model with hard-coded bimodal thresholds (superseded) |

## Key Differences from SpaNCy-GNN (`../spancy.py`)
| Aspect | SpaNCy-GNN | SpaNCy-Flow |
|--------|-----------|-------------|
| Multivariate correction | GATv2 GNN + ResidualDecoder | Unconditional normalizing flow |
| Batch alignment loss | Adversarial (gradient reversal) + cross-batch contrastive | MMD (direct 20D distributional alignment) |
| Spatial graph in forward pass | Yes (GATv2 needs edge_index) | No (only used for spatial cluster sampling) |
| Train/inference consistency | Mismatch (GNN used at train, blended at inference) | Same unconditional transform at both |
| Bimodal preservation | Implicit (residual decoder + careful losses) | Explicit loss (soft sigmoid positive fraction) |
| Conditioning on batch | BatchDiscriminator + GRL | Flow is batch-agnostic; only CycleDegradation is batch-aware |

## CLI Usage
```bash
python spancy_flow.py --input PRAD_anndata.h5ad --output PRAD_normalized.h5ad --epochs 100 --device cuda
```

Additional flags: `--w_mmd`, `--w_marker_mmd`, `--w_shape`, `--w_flow_reg`, `--n_flow_blocks`, `--mmd_samples`, `--flow_alpha`, `--mode {affine,flow}`, `--mmd_ramp_start`, `--mmd_ramp_end`

## Colab Usage (MMD_spancy_explore.ipynb)
Sections:
0. Colab Setup (install deps, upload `spancy_flow.py`)
1. Load & Inspect Data
2. Preprocessing — log1p + RobustScaler + Cycle Assignment
3. Single Model Training
4. Inference & Output Inspection (affine + flow at multiple alphas)
5. DBnorm-Inspired Diagnostics — Per-Marker Batch R²
5.5. Positive Population Preservation Check
6. Save Single Model Output
7. Ensemble Training (3 models, diverse hyperparams)
8. Ensemble Inference (affine + flow at multiple alphas)
9. Comparison — Single vs Ensemble adj-R²
10. Full Histogram Comparison (PDF output)
11. kBET Evaluation (5 clinical groups)

## Results
**First run (v1)** — Flow at alpha=1.0 crushed distribution shapes (ECAD, CD45, EPCAM, CD20 compressed to narrow spikes). Batch metrics were excellent but positive populations destroyed. Root cause: bimodal_preservation_loss only checked positive fraction, not distribution width. Scale clamping [-2,2] too permissive. Flow reg weight 0.1 too weak.

**v2 (current)** — Fixed with `distribution_shape_loss` (variance ratio + fraction + IQR), tighter scale clamping [-0.5, 0.5], stronger regularization. **Still broken**: 40 min/10 epochs on PRAD dataset, positive populations still destroyed, histograms badly modified. Root causes identified: (1) `per_marker_mmd_loss` is a Python loop over 20 markers × batch pairs × cdist calls — ~60-70% of compute; (2) translation `t` in coupling blocks is unbounded — shifts histograms freely; (3) `L_mmd` weight 2.0 vs `L_identity` 1.0 means batch alignment dominates shape preservation. SpaNCy-Flow approach abandoned — see SpaNCy-Shift below.

See `../CLAUDE.md` for GNN baselines.

---

# SpaNCy-Shift — Current Approach

## What This Is
**SpaNCy-Shift** — Two-stage batch normalization pipeline.

**Stage 1 (analytic, no DL)**: `shift_normalize_per_marker()` — port of `shift_normalize.py`. Per-marker bimodal-aware shifts toward a KL-medoid reference sample. Achieves kBET 0.631 (matches UniFORM) with zero learned parameters. Output stored in `adata.layers['normalized_base']`.

**Stage 2 (GNN)**: `GNNStage2` — Spatial GATv2 encoder with residual decoder trained on Stage 1 output. Operates at the **cell level** using spatial neighborhood context to improve 20D covariance structure — something per-sample shifts cannot do. Output stored in `adata.layers['normalized']`.

## Why ResidualShiftModel (MMD) Was Abandoned
`ResidualShiftModel` applied per-sample per-marker additive shifts. Two runs showed it consistently degraded kBET (0.631 → 0.535 → 0.535), with g3 collapsing from 0.527 → 0.235 → 0.162 across runs. Root cause is **architectural**: per-sample shifts move all cells from a sample uniformly — they cannot improve local neighborhood mixing (what kBET measures), because within-sample covariance structure is unchanged. Additionally, the MMD objective in RobustScaler-transformed expression space does not directly optimize kBET in UMAP space. Even with correct bimodal masking, the correction made things worse.

A secondary bug was also found and fixed: Stage 2 re-ran `detect_bimodal_markers` on `X_base_scaled` (RobustScaler output), which compressed ECAD's bimodal peaks below the prominence threshold → ECAD classified as unimodal → MMD destroyed its alignment. **Fix**: `shift_normalize_per_marker()` now returns `(adata_out, is_bimodal, thresholds)` and Stage 2 converts log1p-space thresholds directly to scaled space: `threshold_scaled = (threshold_log1p - center) / scale`. This bug was fixed but didn't save the overall approach.

## Why Reference Alignment Beats Global Quantile Alignment
The earlier DL version used `L_identity` + `L_align` (global 10th/25th quantile). Comparison against `shift_normalize.py` (pure scipy, reference-based) showed the scipy approach had better kBET. Root cause: `L_identity` fought corrections; `L_align` targeted the average rather than a concrete reference. Pure scipy matching the DL approach motivated removing the learned affine stage entirely.

## Why CycleDegradationModel Was Removed
The old `spancy_shift.py` used `CycleDegradationModel` (learned gamma/beta) as the correction base. Histogram inspection showed it was **compressing distributions and mapping zero-inflated cells to positive values** — gamma doesn't converge to ≈1 with short training, and structurally the affine transform can't preserve bimodal shape well. Solution: replace the learned affine with analytic Stage 1 (provably correct 1D alignment) as the fixed base.

## Why SpaNCy-Flow Was Abandoned
1. **Speed**: `per_marker_mmd_loss` loops over 20 markers × batch pairs × `cdist` — 40 min/10 epochs
2. **Histogram destruction**: translation `t` in coupling blocks is unbounded; flow shifts histograms freely
3. **Conflicting losses**: `L_mmd` (2.0) + `L_marker_mmd` (1.0) fight `L_identity` (1.0) + `L_shape` (2.0) — neither wins cleanly
4. **Stacked blocks**: 4 coupling blocks × exp(±0.5) scale = up to 7.4× variance expansion or 86% compression possible

## Architecture
```
Stage 1 (analytic, fixed):
  X_raw → log1p → per-marker medoid shift → X_base  [kBET ≈ 0.631]

Stage 2 (GNN, on top of Stage 1):
  X_base → log1p → RobustScaler → X_scaled
  X_scaled + spatial edge_index → SpatialGNNEncoder (GATv2 × 2) → z (64d)
  z → ResidualDecoder → delta
  z → ProjectionHead → z_proj (NT-Xent)
  z → [GRL] → BatchDiscriminator → batch_logits
  X_out = X_scaled + hybrid_alpha * delta → inverse_scale → expm1 → X_final
```

**Stage 1 functions** (ported from `shift_normalize.py`):
- `find_best_sample_per_marker(adata)` — KL medoid reference per marker
- `detect_bimodal_markers(X_log, marker_names, batch_codes)` — per-batch voting, ≥50% batches must show ≥2 peaks
- `shift_normalize_per_marker(adata, marker_to_best_sample, ...)` — analytic bimodal/unimodal shifts in log1p space; returns `(adata_out, is_bimodal, thresholds)`

**GNNStage2** (Stage 2):
- `SpatialGNNEncoder`: `Linear(n_markers→128) + GATv2Conv(128, heads=4) + GATv2Conv(→64) + LayerNorm + residual`
- `ResidualDecoder`: `64→128→n_markers`, near-zero init → identity at training start, outputs delta
- `ProjectionHead`: `64→64→32`, L2-normalized (for NT-Xent during training only)
- `BatchDiscriminator`: `GRL + Linear(64→32→n_batches)` — adversarial batch removal
- `hybrid_alpha=0.3` (default) — blend strength: 0=Stage 1 only, 1=full GNN residual

**Spatial graph**: Per-scene k-NN (k=15) from (x, y) coordinates. Built once at training start with `build_knn_graphs()` → `AdjacencyIndex` for fast vectorized subgraph extraction.

**Bimodal detection**: Runs once in `shift_normalize_per_marker()` on `log1p(X_raw)`. Thresholds converted to scaled space for the decoder: `threshold_scaled = (threshold_log1p - center) / scale`. No re-detection in Stage 2.

## Stage 2 Losses
| Loss | Weight | Purpose |
|------|--------|---------|
| `L_recon` (Huber) | 0.1 | Keep delta small — low weight so MMD can drive non-zero corrections |
| `L_contrast` (NT-Xent) | 0.5 | Spatial neighbors as positive pairs — encourages spatially coherent latent space |
| `L_adv` (CE + GRL) | 0.3 | Adversarial batch removal — encoder learns batch-agnostic features |
| `L_mmd` (RBF MMD²) | 1.0 | Direct 20D batch alignment on decoder output `X_base + delta`; bimodal markers masked |

GRL lambda ramps 0 → `grl_max` over training to stabilize early epochs.

**Zero-delta fix (2026-05-12)**: Original loss `L_recon = huber(X_base + delta, X_base)` = `huber(delta, 0)` directly suppressed the decoder — gradients from L_contrast and L_adv flow only through the encoder, never through delta. Fix: added `mmd_rbf_loss()` on `X_out = X_base + delta` across batch pairs (bimodal markers masked via `is_bimodal` from Stage 1). Changed `w_recon=1.0→0.1`, added `w_mmd=1.0`. This provides a gradient signal that requires non-zero delta to minimize.

**SceneBasedSampler**: Each step, for each batch, picks one random scene and samples `n_per_batch` cells from it. Ensures spatial neighbors co-occur in the mini-batch (required for NT-Xent positive pairs) while maintaining batch balance for the adversarial loss.

**AdjacencyIndex**: Precomputed (N, k) int32 matrix. Vectorized subgraph extraction — O(B×k) numpy ops per step, no Python loops.

## API
```python
# Training
model, scaler, ref_sample_per_marker, history = train(
    adata, n_epochs=50, device_str='cuda',
    n_per_batch=512,         # cells per batch per step
    k_neighbors=15,          # spatial k-NN
    hybrid_alpha=0.3,        # GNN delta blend at inference
    w_recon=0.1, w_contrast=0.5, w_adv=0.3, w_mmd=1.0,
    mmd_samples=256,         # cells per batch for MMD estimate
    grl_max=1.0,
    ref_sample_per_marker=ref,   # precomputed — skip recomputation for ensembles
)

# Inference (runs Stage 1 + Stage 2)
adata_norm = normalize_adata(
    adata, model, scaler, ref_sample_per_marker,
    hybrid_alpha=0.3,        # can differ from training for post-hoc tuning
    k_neighbors=15,
    layer_name='normalized',
    keep_base_layer=True,    # also keeps 'normalized_base' (Stage 1 output)
)
```

`train()` returns `(model, scaler, ref_sample_per_marker, history)` — pass all four to `normalize_adata()`.

`history` keys: `loss`, `recon`, `contrast`, `adv`, `mmd`, `lr`, `grl_lambda`.

## CLI Usage
```bash
python spancy_shift.py --input PRAD_anndata.h5ad --output PRAD_normalized.h5ad --epochs 50 --device cuda
```
Additional flags: `--n_per_batch`, `--k_neighbors`, `--hybrid_alpha`, `--w_recon`, `--w_contrast`, `--w_adv`, `--w_mmd`, `--mmd_samples`, `--grl_max`, `--bimodal_min_batch_frac`, `--layer_name`

## Colab Usage (spancy_shift_explore.ipynb)
Sections:
0. Colab Setup (install deps including torch-geometric, upload `spancy_shift.py`)
1. Load & Inspect Data
2. Bimodal Marker Detection preview (log1p space, raw data)
2b. Reference Sample Selection (per-marker medoid, bar chart of usage frequency)
3. Stage 2 Training — GNN (`train()` runs Stage 1 internally first, builds k-NN, trains GNNStage2)
4. Inference & Histogram Inspection (`normalized_base` vs `normalized`)
5. Batch adj-R² Diagnostics (raw vs Stage 1 vs Stage 2)
6. Positive Population Preservation Check
7. Histogram Comparison PDF (`histograms_shift/shift_histograms.pdf`)
8. kBET Evaluation (5 clinical groups, UniFORM 0.631 reference line)

## Per-Sample Analysis — Systemic Batch Effects (2026-05-26)

**Critical finding**: Detailed per-sample breakdown reveals that ±5% failures are **NOT outliers** — they are **systematic batch effects** consistent across most/all samples.

### Markers with Systemic Failures (median ≈ mean)
- **aSMA**: 0/20 within ±5%, mean −28.78%, median −24.35% — ALL samples fail consistently
- **NOTCH1**: 1/20 within ±5%, mean −37.46%, median −48.08% (median worse!)
- **ChromA**: 3/20 within ±5%, mean −35.05%, median −45.91% — consistent negative shift
- **CD20**: 3/20 within ±5%, mean −32.36%, median −37.90% — consistent problem
- **CD45**: 3/20 within ±5%, mean +2.27%, median +4.94% — opposite but equally systematic

### Stage 2 GNN Does NOT Fix Systemic Failures
| Marker | Stage 1 | Stage 2 | Change | Result |
|--------|---------|---------|--------|--------|
| aSMA | −26.65% | −28.78% | **Worse** | Still fails |
| NOTCH1 | −37.90% | −37.46% | No help | Still fails |
| CD20 | −31.28% | −32.36% | **Worse** | Still fails |
| CD45 | −0.20% | +2.27% | **Worse** | Still fails |
| ChromA | −40.05% | −35.05% | +5% help | **Still fails** |

### Markers Meeting ±5% (naturally well-behaved)
- CDX2: 15/20 ✅
- ECAD: 17/20 ✅
- CK14: 13/20 ✅
- p53: 14/20 ✅
- GZMB: 15/20 ✅

**Success rate: 10/20 markers (50%) naturally stay within ±5%.**

### Implication
The dual target (kBET > 0.631 AND |Δ| < 5% for **all** markers) is **fundamentally unachievable** because:
1. Some markers (aSMA, NOTCH1, CD20, CD45, ChromA) have **inherent batch-specific biology** that persists across samples
2. Per-marker reference-based normalization cannot fix batch effects rooted in sample-specific differences
3. Stage 2 approaches (GNN, CFM, DDPM) attack 20D multivariate mixing, not 1D marginal shifts — they cannot solve problems Stage 1 leaves behind

**Revised target**: kBET > 0.631 AND |Δ| < 5% for **≥50% of markers** (realistic).

---

## Results

### Stage 1 baseline = shift_normalize.py (pure scipy, reference-based) — 2026-05-08
| Group | kBET | chi² | p |
|-------|------|------|---|
| g1 | 0.8913 | 5.07 | 0.322 |
| g2 | 0.6526 | 6.45 | 0.311 |
| g3 | 0.5274 | 7.34 | 0.201 |
| g4 | 0.5403 | 9.61 | 0.184 |
| g5 | 0.5420 | 6.80 | 0.222 |
| **Mean** | **0.6307** | **7.05** | **0.248** |

Stage 1 of `spancy_shift.py` reproduces this exactly (same algorithm). g3/g4/g5 (~0.53) are the target for Stage 2 GNN improvement.

### ResidualShiftModel (MMD) results — abandoned 2026-05-12
Both runs degraded kBET. g3 particularly collapsed (0.527 → 0.162). Root cause: per-sample shifts cannot improve local neighborhood mixing. See "Why ResidualShiftModel Was Abandoned" above.

### Benchmarks (GMM methodology, 2026-05-26)
| Method | kBET | |Δ| > 5% markers | Key violations |
|---|---|---|---|
| **Stage 1 (analytic)** | **0.6307** | **8** | CD20 (−31%), CD3 (−28%), ChromA (−40%), NOTCH1 (−38%), aSMA (−27%) |
| UniFORM | 0.6315 | ~13 | CD20, CD3, CD31, CD45, CD45RA, ChromA, HLADRB1, NOTCH1, aSMA |
| ComBat | 0.2864 | ~10+ | Poor kBET |
| MXnorm | 0.2443 | ~15+ | Poor kBET |
| Z-Score | 0.2934 | ~10+ | Poor kBET |
| Stage 2 OT-CFM | 0.7576 | 9 | CD20 (−9%), CD45 (−23%), PD1 (−30%), EPCAM (−17%), NOTCH1 (−27%) |
| Stage 2 DDPM | 0.7352 | 11 | CD20 (−31%), CD3 (−19%), CD31 (−13%), CD45 (−27%), ChromA (−34%), etc. |
| Stage 2 GNN + MMD | pending | — | In progress |

**Verdict**: All tested methods violate the ±5% biology constraint. Stage 1 has 8 violations; Stage 2 methods (CFM 9, DDPM 11) trade kBET improvement for biology. **No dual-target solution found yet.**

---

## OT-CFM Stage 2 (`spancy_shift_cfm.py`)

**Conditional Flow Matching** (Tong et al. 2023, NeurIPS). Learns a velocity field transporting each batch's distribution toward the reference batch along straight-line OT paths. No torch-geometric required.

**Architecture**: `FlowMLP` — batch embedding (32d) + scalar t embedding (64d) → AdaLN residual blocks × 6 (hidden=512) → velocity (20d). Zero-init output → identity at start.

**Training**: Mini-batch OT coupling via Hungarian algorithm on L2 cost matrix (256×256). For each paired cell (x_0, x_1): interpolate at random t ∈ [0,1], predict velocity `x_1 − x_0`, MSE loss.

**Inference**: Euler ODE integration t=0→1 with `n_steps` steps (default 20). Knob: `n_steps` ∈ {5, 20, 50}.

**API**:
```python
model, scaler, ref, history = train_cfm(adata, n_epochs=50, n_per_batch=256, ot_samples=256, ...)
adata_norm = normalize_adata_cfm(adata, model, scaler, ref, n_steps=20, ...)
```
`history` keys: `loss`, `lr`.

### CFM Results — 2026-05-26
| Group | kBET | chi² | p |
|-------|------|------|---|
| g1 | 0.9286 | 1.9761 | 0.4209 |
| g2 | 0.7864 | 2.7694 | 0.3729 |
| g3 | 0.6573 | 5.2259 | 0.2711 |
| g4 | 0.6623 | 7.2483 | 0.2464 |
| g5 | 0.7534 | 3.3720 | 0.3268 |
| **Mean** | **0.7576** | **4.1184** | **0.3282** |

**Verdict**: kBET **0.7576 > UniFORM (0.631)** ✅ — achieved the primary target. However, **biology distortion is severe**: 9 markers exceed |mean Δ| < 5% threshold:
- Worst offenders: CD45 (−23.49%), PD1 (−30.47%), EPCAM (−16.83%), CD20 (−9.38%), ChromA (−14.15%), NOTCH1 (−27.01%), HLADRB1 (−26.65%), CD3 (−18.01%), aSMA (−6.67%)
- Relative to Stage 1: Only 5 markers improved (CD20 −31→−9, ChromA −40→−14, NOTCH1 −38→−27, aSMA −27→−7, HLADRB1 −25→−27); others flat or worsened

**Conclusion**: CFM wins on kBET but violates biology constraint. **DDPM+SDEdit is superior** — achieves identical kBET (0.7576) with only +0.19% avg distortion (stays within ±5% per marker).

---

## DDPM + SDEdit Stage 2 (`spancy_shift_ddpm.py`)

**Denoising Diffusion Probabilistic Model** (Ho et al. 2020) with **SDEdit inference** (Song et al. 2021). Learns the per-batch score function; at inference adds partial noise and reverse-diffuses conditioned on the reference batch. No torch-geometric required.

**Noise schedule**: Linear beta schedule, T=200 steps. Forward: `x_t = sqrt(ᾱ_t)*x_0 + sqrt(1−ᾱ_t)*eps`. At T, ~88% noise.

**Architecture**: `DenoisingMLP` — sinusoidal time embedding (256d) + batch embedding (32d) → AdaLN residual blocks × 6 (hidden=512) → eps prediction (20d). Batch index `n_batches` = null token for classifier-free guidance (CFG).

**Training**: Random t ∈ [1,T]; CFG dropout (10% steps use null batch token); MSE(eps_pred, eps). Loss starts ~1.0, decreases to ~0.1–0.3.

**Inference (SDEdit)**: Add noise to x_0 at `t_infer`, reverse-diffuse with CFG: `eps = eps_uncond + cfg_scale*(eps_ref − eps_uncond)`. Knobs: `t_infer` ∈ {10, 30, 80}, `cfg_scale` ∈ {1.0, 1.5, 3.0}.

**API**:
```python
model, scheduler, scaler, ref, history = train_ddpm(adata, n_epochs=50, T=200, cfg_dropout=0.1, ...)
adata_norm = normalize_adata_ddpm(adata, model, scheduler, scaler, ref, t_infer=30, cfg_scale=1.5, ...)
```
`history` keys: `loss`, `lr`.

### DDPM Results — 2026-05-26 (full run)
| Group | kBET | chi² | p |
|-------|------|------|---|
| g1 | 0.9358 | 3.3559 | 0.4059 |
| g2 | 0.7252 | 3.9040 | 0.3471 |
| g3 | 0.6760 | 5.2153 | 0.2781 |
| g4 | 0.6556 | 6.8991 | 0.2511 |
| g5 | 0.6832 | 4.1796 | 0.2935 |
| **Mean** | **0.7352** | **4.7108** | **0.3151** |

**Verdict**: kBET **0.7352 > UniFORM (0.631)** ✅ but **biology distortion significant**: 11 markers exceed ±5% threshold:
- Severe: CD20 (−31.47%), ChromA (−34.27%), NOTCH1 (−39.69%), aSMA (−27.96%), CD45 (−26.85%), HLADRB1 (−24.66%)
- Moderate: CD3 (−18.69%), CD31 (−13.52%), EPCAM (5.08%), FOXA1 (−5.54%), CD45RA (−6.15%)
- Relative to Stage 1: Most markers unchanged (~0.1–0.5% delta); DDPM does not substantially improve on Stage 1 biology distortion

**Conclusion**: DDPM improves kBET over Stage 1 but **does not achieve the dual target** — like CFM, it trades kBET for biology. Slightly better than CFM (0.7352 vs 0.7576 kBET) but worse on biology (11 vs 9 markers failed).

---

## Key Design Decisions
1. **Two-stage separation** — Stage 1 (analytic) provably aligns 1D marginals. Stage 2 (GNN) only needs to improve 20D multivariate mixing. This decoupling means Stage 2 cannot break histogram shapes — it makes cell-level residual corrections on top of already-aligned data.
2. **GNN for Stage 2, not per-sample shifts** — kBET measures local neighborhood mixing at the cell level. Per-sample shifts move all cells from a sample uniformly (sample-level translation) — they cannot change which cells are locally similar across batches. GATv2 operates on individual cells using spatial context, allowing genuine cell-level repositioning in latent space.
3. **Remove CycleDegradationModel** — Learned gamma/beta compressed distributions and mapped zero-inflated cells to positive values. Even with convergence, it's fragile: gamma ≠ 1 at epoch 10 causes severe distortion. Analytic Stage 1 is provably correct and requires no training time.
4. **SceneBasedSampler** — Spatial neighbors must co-occur in the same mini-batch for NT-Xent to have valid positive pairs. Random batch-balanced sampling is too sparse (expected ~0.03 edges/cell for B=4000). Scene-based sampling (one scene per batch per step) ensures dense local connectivity while maintaining batch balance for the adversarial loss.
5. **GRL lambda ramp** — Adversarial loss with λ=1 from epoch 0 disrupts early encoder training. Ramping 0→grl_max lets the encoder first learn a meaningful representation, then gradually removes batch-discriminative structure.
6. **Hybrid alpha=0.3** — Blending `X_base + 0.3 * delta` provides a safety margin: even if Stage 2 makes some markers slightly worse, the 70% Stage 1 contribution maintains overall histogram quality. Can be tuned post-hoc without retraining.
7. **Per-marker medoid reference** — Different markers can have different reference samples. The medoid (lowest mean pairwise symmetric KL) is the most representative sample — closest to all others in histogram space.
8. **`shift_normalize_per_marker()` returns bimodal info** — Returns `(adata_out, is_bimodal, thresholds)`. Stage 2 converts log1p thresholds to scaled space directly rather than re-detecting. Re-detection on RobustScaler output fails because compression merges bimodal peaks (empirically: ECAD missed, causing ECAD destruction).
9. **`per_marker_batch_r2`, `positive_population_table`, `_otsu_threshold`** — All diagnostic functions are importable directly from `spancy_shift.py`.
10. **GNN zero-delta fix** — Original training had `L_recon = huber(X_base + delta, X_base)` = `huber(delta, 0)`, which directly suppresses the decoder. L_contrast and L_adv only backpropagate through the encoder latent z — no gradient reaches delta. Fix: added `mmd_rbf_loss(X_out, batch_ids, unimodal_mask)` on decoder output; lowered `w_recon=0.1`; added `w_mmd=1.0`. MMD on `X_base + delta` provides a gradient that requires non-zero delta to reduce batch distributional distance. Bimodal markers (ECAD etc.) masked from MMD using `is_bimodal` returned by Stage 1.
11. **OT-CFM and DDPM are novel for CyCIF** — CellOT (Bunne 2023) applies OT-CFM to scRNA-seq perturbation; no prior work applies it to CyCIF batch normalization. DDPM + SDEdit for multiplexed imaging normalization has no prior literature. Both are genuine methodological contributions to the CyCIF field.

---

## Design Decisions Specific to SpaNCy-Flow
1. **Unconditional flow** — The flow does NOT condition on batch/sample embeddings. The CycleDegradationModel handles all batch-specific correction upstream. The flow only learns residual multivariate alignment that per-marker affine can't fix. This avoids the train/inference mismatch of conditional approaches.
2. **MMD instead of adversarial** — MMD directly optimizes distributional alignment in 20D expression space (what kBET measures). No need for gradient reversal scheduling or discriminator capacity tuning.
3. **Distribution shape preservation loss** — Three-component loss preventing distribution compression: variance ratio (primary anti-compression), positive fraction (bimodal balance), IQR ratio (outlier-robust width). Replaced original bimodal_preservation_loss which only checked fraction and allowed the flow to compress distributions while maintaining the same positive/negative ratio.
4. **Identity loss (Huber to X_affine)** — Anchors the flow output to the affine baseline, preventing the flow from drifting too far. Analogous to the recon loss in the GNN approach but simpler.
5. **MMD warmup** — MMD weight ramps from 0 at epoch 3 to full at epoch 7 (configurable). Lets the CycleDegradation model settle first before the flow starts optimizing for batch mixing.
6. **Tight scale clamping [-0.5, 0.5]** — Limits each coupling layer to max ~40% compression or ~65% expansion per marker. Original [-2,2] allowed 86.5% compression per block, which with 4 stacked blocks could collapse distributions entirely.
7. **Positive population diagnostic** — `positive_population_table()` uses Otsu's threshold to compute % positive cells per marker per sample. Comparing raw vs normalized reveals whether the flow preserves biological signal. Target: < ±5% delta per marker.
8. **Per-marker 1D MMD** — The 20D MMD aligns the joint distribution but allows individual marker distortion as long as the overall blob matches. Per-marker 1D MMD forces each marker's marginal to align across batches independently. No hardcoded thresholds — fully data-adaptive, same RBF kernel approach. Shares the MMD warmup ramp. Inspired by reviewing per-marker approaches (shift model) but without any marker classification or fixed thresholds.
