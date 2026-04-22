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
| `spancy_shift.py` | **Current approach** (~1100 lines): CycleDegradationModel + AdaptiveShiftModel + 20D MMD |
| `spancy_shift_explore.ipynb` | Colab notebook for SpaNCy-Shift: training, inference, bimodal detection, diagnostics, kBET |
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
**SpaNCy-Shift** — Replaces the broken normalizing flow with explicit per-sample shifts. Combines the insight from `shift_model.ipynb` (per-sample bimodal-aware shifts work well for adj-R²) with the 20D MMD loss from SpaNCy-Flow (for kBET). Fully adaptive bimodal detection — no hard-coded thresholds.

## Why SpaNCy-Flow Was Abandoned
1. **Speed**: `per_marker_mmd_loss` loops over 20 markers × batch pairs × `cdist` — 40 min/10 epochs
2. **Histogram destruction**: translation `t` in coupling blocks is unbounded; flow shifts histograms freely
3. **Conflicting losses**: `L_mmd` (2.0) + `L_marker_mmd` (1.0) fight `L_identity` (1.0) + `L_shape` (2.0) — neither wins cleanly
4. **Stacked blocks**: 4 coupling blocks × exp(±0.5) scale = up to 7.4× variance expansion or 86% compression possible

## Architecture
```
X_raw → log1p → RobustScaler → CycleDegradationModel → X_affine → AdaptiveShiftModel → X_shifted
```

**CycleDegradationModel** (same as SpaNCy-Flow, unchanged):
- batch(32d) + sample(16d) + cycle(16d) → MLP(64→64→2) → per-marker gamma/beta
- `X_affine = (X - beta) / gamma`

**AdaptiveShiftModel**:
- `nn.Embedding(n_samples, n_markers)` for neg-peak shift + separate for pos-peak shift
- Bimodal markers: smooth blend `(1-w_pos)*shift_neg + w_pos*shift_pos` where `w_pos = sigmoid((X_affine - threshold) * sharpness)`
- Unimodal markers: single `shift_neg` applied uniformly
- All shifts zero-initialized → identity at start
- **Shifts preserve per-marker variance by construction** (additive only, no scaling)

**Bimodal detection** (run once at training start on full scaled dataset):
- `detect_bimodal_markers()` runs `_find_peaks()` **per batch** and uses majority voting
- A marker is bimodal only if ≥50% of batches independently show ≥2 peaks (`bimodal_min_batch_frac=0.5`)
- Threshold = median midpoint across bimodal batches (not global midpoint)
- Prevents batch-separated unimodal distributions (e.g. ChromA) from being falsely classified as bimodal from the pooled global histogram
- Replaces all 4 hard-coded thresholds from `shift_model.ipynb`: `bc_threshold=0.35`, `separation_threshold=1.0`, `BIC>5000`, `min_weight=0.05`

## Losses
| Loss | Weight | Purpose |
|------|--------|---------|
| `L_recon` (Huber) | 1.0 | **Critical anchor**: `huber(X_affine, X_scaled)` — prevents gamma/beta collapse |
| `L_identity` (Huber) | 0.5 | Keep X_shifted near X_affine — anchor shifts |
| `L_align` (quantile 10th/25th) | 0.5 | Align neg-population quantiles across samples |
| `L_mmd` (20D RBF MMD²) | 0.5 (warmed epoch 3→7) | Multivariate batch alignment for kBET |

No `L_shape` (shifts can't change variance), no `L_marker_mmd` (slow, redundant after affine), no `L_flow_reg`.

**MMD implementation**: vectorized over bandwidths (tensor ops, no Python loop per bandwidth).

## Inference Modes
1. **Affine** (`mode="affine"`): CycleDegradationModel only. Shape-preserving.
2. **Shift** (`mode="shift"`, default): Affine + adaptive shifts with `shift_alpha` blending (0=pure affine, 1=full shift).

## CLI Usage
```bash
python spancy_shift.py --input PRAD_anndata.h5ad --output PRAD_normalized.h5ad --epochs 10 --device cuda
```
Additional flags: `--w_mmd`, `--w_recon`, `--w_identity`, `--w_align`, `--mmd_ramp_start`, `--mmd_ramp_end`, `--shift_alpha`, `--mode {affine,shift}`, `--align_samples`, `--bimodal_min_batch_frac`

## Colab Usage (spancy_shift_explore.ipynb)
Sections:
0. Colab Setup (install deps, upload `spancy_shift.py`)
1. Load & Inspect Data
2. Bimodal Marker Detection (adaptive, visualized)
3. Single Model Training (default 10 epochs)
4. Inference & Output Inspection (affine + shift at multiple alphas)
5. Batch adj-R² Diagnostics (`per_marker_batch_r2` — now in `spancy_shift.py`)
6. Positive Population Preservation Check
7. Histogram Comparison (PDF output to `histograms_shift/`)
8. kBET Evaluation (5 clinical groups)

## Results
*Pending first Colab run. Will update after 10-epoch test.*

Baselines from SpaNCy-GNN ensemble (for comparison):
- Mean adj-R²: 0.044 (ensemble affine, 3 models)
- Mean kBET: 0.418 (ensemble affine)
- Mean kBET: 0.574 (ensemble hybrid alpha=0.2, earlier broken run)

## Key Design Decisions
1. **Shifts instead of flow** — Additive shifts cannot change per-marker variance by construction. No shape preservation loss needed. Histogram shapes preserved automatically.
2. **`L_recon` anchors CycleDegradationModel (critical)** — `huber(X_affine, X_scaled)` with w_recon=1.0 prevents gamma/beta collapse. Without it, quantile loss (10-40) dominates over identity loss (0.0001), driving gamma/beta to diverge and destroying all distributions. The recon loss forces gamma/beta to stay close to identity so all other losses make only small corrections.
3. **Per-batch bimodal voting** — `detect_bimodal_markers()` runs `_find_peaks()` per batch and requires ≥50% of batches to independently show ≥2 peaks. Prevents batch-separated unimodal distributions (e.g. ChromA: different batches sit at different positions, creating two peaks in the global histogram but none within each batch) from being falsely classified as bimodal. Threshold = median midpoint across bimodal batches.
4. **Remove `L_marker_mmd`** — CycleDegradationModel already achieves 0.044 mean adj-R² (per-marker alignment done). 1D MMD on top is redundant and was the primary speed bottleneck (~60% of training time).
5. **Weak MMD (0.5 vs old 2.0)** — MMD is now a regularizer for kBET, not the primary objective. Recon + identity losses dominate → corrections stay small → histograms preserved.
6. **MMD ramp epoch 3→7** — Lets CycleDegradation settle first before MMD starts pushing batch mixing. Original 5→15 ramp with 10 epochs meant MMD never reached full weight.
7. **Ensemble deferred** — Single model first. Ensemble (`normalize_adata_ensemble`) to be added after single model results are validated.
8. **`per_marker_batch_r2` promoted to module** — Previously only existed inline in `MMD_spancy_explore.ipynb`. Now a proper function in `spancy_shift.py`.

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
