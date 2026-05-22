# SpaNCy-Shift — Methods Reference

All methods share the same **Stage 1** analytic baseline. Stage 2 is where the approaches differ.

---

## Overall Pipeline

```
Raw CyCIF data (1.76M cells × 20 markers)
        │
        ▼
┌───────────────────────────────────────┐
│  STAGE 1  —  Analytic Shift           │
│  (no learning, always runs first)     │
│  Output: normalized_base  kBET≈0.631  │
└───────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2  —  Deep Learning (choose one)                     │
│                                                             │
│   ┌──────────┐   ┌──────────┐   ┌────────────────────────┐ │
│   │   GNN    │   │  OT-CFM  │   │   DDPM + SDEdit        │ │
│   │ (GATv2)  │   │ FlowMLP  │   │   DenoisingMLP         │ │
│   └──────────┘   └──────────┘   └────────────────────────┘ │
│  Output: normalized   Target kBET > 0.631                   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Diagnostics: kBET · positive population Δ · batch adj-R²
```

---

## Stage 1 — Analytic Shift Normalization
**File:** `spancy_shift.py` → `shift_normalize_per_marker()`
**Reference:** `shift/shift_normalize.py` (pure scipy source of truth)

```
X_raw (counts)
    │
    ▼  log1p
X_log
    │
    ├─► detect_bimodal_markers()
    │       Per-batch histogram peak voting.
    │       Marker = bimodal if ≥50% of batches show ≥2 peaks.
    │       Output: is_bimodal[20], thresholds[20]
    │
    ├─► find_best_sample_per_marker()
    │       For each marker: compute pairwise symmetric KL
    │       between all sample histograms.
    │       Reference = medoid (lowest mean KL to all others).
    │       Output: ref_sample_per_marker {marker → sample_id}
    │
    ▼
Per-marker, per-sample shift in log1p space:
    │
    ├── UNIMODAL marker:
    │       shift = median(ref) - median(sample)
    │       X_shifted = X_sample + shift
    │
    └── BIMODAL marker:
            Find neg peak and pos peak in both sample & ref.
            neg_shift = ref_neg_peak - sample_neg_peak
            pos_shift = ref_pos_peak - sample_pos_peak
            Piecewise linear blend between the two shifts.
            Tail beyond pos peak: pure shift (slope=1 cap).
            ┌──────────────────────────────────────────┐
            │ neg region │  blend zone  │  pos region  │
            │ +neg_shift │  lerp(n→p)   │  +pos_shift  │
            └──────────────────────────────────────────┘
    │
    ▼
X_base = expm1(X_shifted)   [stored in adata.layers['normalized_base']]

Result: kBET ≈ 0.631  (matches UniFORM with zero learned parameters)
```

---

## Stage 2a — GNN (Spatial GATv2)
**File:** `spancy_shift.py` → `GNNStage2`, `train()`, `normalize_adata()`

The GNN operates at the **cell level** using spatial context. A cell's correction depends on its neighbours — something per-sample shifts cannot do.

```
X_base
    │
    ▼  log1p → RobustScaler
X_scaled (N × 20)
    │
    │  Build spatial k-NN graph from (x, y) per scene (k=15)
    │  AdjacencyIndex: precomputed (N,k) int32 matrix
    │
    ▼
┌─────────────────────────────────────────────────┐
│  SpatialGNNEncoder                              │
│                                                 │
│  Linear(20→128)                                 │
│       │                                         │
│  GATv2Conv(128, heads=4)  ← edge_index          │
│       │ + residual                              │
│  GATv2Conv(128→64, heads=4)  ← edge_index       │
│       │ + LayerNorm                             │
│       ▼                                         │
│    z  (N × 64)                                  │
└─────────────────────────────────────────────────┘
    │              │                  │
    ▼              ▼                  ▼
ResidualDecoder  ProjectionHead   BatchDiscriminator
64→128→20        64→64→32         64→32→n_batches
outputs delta    L2-norm          + GRL (gradient reversal)
    │            z_proj           batch_logits
    │             │                  │
    │          NT-Xent            Cross-entropy
    │          L_contrast         L_adv
    │
    ▼  Huber(delta, 0) → L_recon   MMD(X_out, batch) → L_mmd
    │                               (bimodal markers masked)
    │
    ▼
X_out = X_scaled + hybrid_alpha * delta   (alpha=0.3 default)
    │
    ▼  inverse_scale → expm1
X_normalized   [stored in adata.layers['normalized']]

Losses: L_recon(0.1) + L_contrast(0.5) + L_adv(0.3) + L_mmd(1.0)
Sampler: SceneBasedSampler — one scene per batch per step
         ensures spatial neighbours co-occur in mini-batch
```

**Key knob:** `hybrid_alpha` — 0 = pure Stage 1, 1 = full GNN delta.

---

## Stage 2b — OT-CFM (Conditional Flow Matching)
**File:** `spancy_shift_cfm.py` → `FlowMLP`, `train_cfm()`, `normalize_adata_cfm()`

Learns a **velocity field** that transports cells from each batch toward the reference batch along straight-line optimal transport paths. No spatial graph, no adversarial training.

```
X_base → log1p → RobustScaler → X_scaled
    │
    ▼
Identify reference batch (majority vote over per-marker ref samples)

TRAINING (one step):
    │
    ├─ Sample source cells x_0 from all non-ref batches (batch-balanced)
    ├─ Sample target cells x_1 from reference batch
    │
    ▼
OT Coupling (Hungarian algorithm on 256×256 L2 cost matrix)
    │   Pairs phenotypically similar cells across batches
    │   → (x_0, x_1) matched pairs
    │
    ▼
Interpolate at random t ~ U(0,1):
    x_t = (1-t)·x_0 + t·x_1 + σ·noise   (σ=0.01)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  FlowMLP                                        │
│                                                 │
│  batch_emb(32d) ──┐                             │
│  t_emb(64d) ──────┤                             │
│  x_t(20d) ────────┴→ AdaLN residual × 6        │
│                        hidden=512               │
│                        ▼                        │
│                   velocity (20d)                │
│                   zero-init → identity at start │
└─────────────────────────────────────────────────┘
    │
    ▼
Loss = MSE(velocity_pred, x_1 - x_0)

INFERENCE:
    │
    ▼
Euler ODE integration t=0 → t=1 with n_steps steps:
    x_{t+dt} = x_t + dt · FlowMLP(x_t, t, source_batch)
    │
    ▼  inverse_scale → expm1
X_normalized   [stored in adata.layers['normalized']]

Key knob: n_steps — fewer = conservative, more = stronger correction
Sweep: n_steps ∈ {5, 20, 50}
```

---

## Stage 2c — DDPM + SDEdit
**File:** `spancy_shift_ddpm.py` → `DenoisingMLP`, `DDPMScheduler`, `train_ddpm()`, `normalize_adata_ddpm()`

Learns the **score function** (noise predictor) per batch. At inference, uses SDEdit: add partial noise then reverse-diffuse toward the reference batch using classifier-free guidance (CFG).

```
X_base → log1p → RobustScaler → X_scaled
    │
    ▼
Linear beta schedule: β_1=1e-4 → β_T=0.02, T=200 steps
Cumulative: ᾱ_t = ∏β_i

TRAINING:
    │
    ├─ Sample random t ~ U(1, T)
    ├─ Sample noise ε ~ N(0, I)
    ▼
Forward process:
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    │
    │  CFG dropout: 10% of steps → replace batch label with null token
    │
    ▼
┌─────────────────────────────────────────────────┐
│  DenoisingMLP                                   │
│                                                 │
│  sin/cos time_emb(256d) ──┐                     │
│  batch_emb(32d) ──────────┤                     │
│  x_t(20d) ────────────────┴→ AdaLN residual × 6 │
│                                hidden=512        │
│                                ▼                 │
│                           ε_pred (20d)           │
│                           zero-init output       │
└─────────────────────────────────────────────────┘
    │
    ▼
Loss = MSE(ε_pred, ε)

INFERENCE (SDEdit):
    │
    ├─ Start from x_0 = X_scaled (Stage 1 output)
    ▼
Add partial noise at t_infer:
    x_{t_infer} = √ᾱ_{t_infer} · x_0 + √(1-ᾱ_{t_infer}) · ε
    │
    ▼
Reverse diffusion t_infer → 0 with CFG:
    ε_uncond = DenoisingMLP(x_t, t, null_batch)
    ε_ref    = DenoisingMLP(x_t, t, ref_batch)
    ε_guided = ε_uncond + cfg_scale · (ε_ref - ε_uncond)
    │
    ▼  DDPM reverse step (subtract predicted noise)
    │  repeat until t=0
    ▼  inverse_scale → expm1
X_normalized   [stored in adata.layers['normalized']]

Key knobs:
  t_infer   — noise level: low=conservative, high=aggressive
              sweep: {10, 30, 80}
  cfg_scale — guidance strength toward ref batch
              sweep: {1.0, 1.5, 3.0}

IMPORTANT: Bimodal markers (ECAD, CD45 etc.) pass through Stage 1
           unchanged — DDPM does NOT modify them.
```

---

## Diagnostics

### Positive Population Preservation
**Function:** `positive_population_table()` in `spancy_shift.py`

Methodology matches UniFORM paper (Chen et al.) exactly.

```
For each marker:
    GLOBAL threshold (normalized data):
        Fit GMM (2 components) on ALL normalized log1p cells combined
        (subsample 50k for fit; predict on all cells; fallback to Otsu)
        threshold = max(hard-assigned negative class)
        │
        ▼
    For each sample:
        LOCAL threshold (raw data):
            Fit GMM on this sample's raw log1p cells only
            threshold = max(hard-assigned negative class)

        pct_pos_raw  = % raw cells   > local threshold
        pct_pos_norm = % norm cells  > global threshold
        delta        = pct_pos_norm - pct_pos_raw

Target: |delta| < 5% per marker
UniFORM reference: mean delta ≈ -3.4% but large per-marker distortions
```

### Batch adj-R²
**Function:** `per_marker_batch_r2()` in `spancy_shift.py`

```
For each marker:
    Regress log1p(expression) on batch one-hot labels
    Report adjusted R²

Lower adj-R² = less residual batch effect
Target: adj-R² < 0.05 per marker
```

### kBET
Computed externally via `pegasus.calc_kBET()`.

```
For each clinical group (5 groups, 2 samples from different batches):
    Subset cells → UMAP embedding
    kBET: test whether local neighbourhoods are batch-mixed
    acceptance rate = fraction of cells where batch composition
                      matches global expectation (chi-square test)

Higher kBET = better batch mixing
Baselines: UniFORM = 0.631, SpaNCy-GNN ensemble hybrid = 0.574
Target:    Stage 2 > 0.631
```

---

## Abandoned Approaches

### SpaNCy-GNN (`../spancy.py`)
Full GNN with `CycleDegradationModel` (learned gamma/beta) + adversarial + cross-batch contrastive loss. Ensemble hybrid reached kBET 0.574 — below UniFORM. Problem: GNN used at training but blended at inference (train/inference mismatch); CycleDegradation model needs many epochs to converge.

### SpaNCy-Flow (`spancy_flow.py`)
Normalizing flow (cycle-block coupling) + MMD. Destroyed distribution shapes (ECAD compressed to spike). Too slow (40 min/10 epochs). Conflicting losses couldn't be balanced.

### ResidualShiftModel
Per-sample additive shifts with MMD loss. Consistently degraded kBET (0.631 → 0.535). Root cause: per-sample shifts move all cells from a sample uniformly → cannot improve local neighbourhood mixing (what kBET measures).

---

## Results Summary

| Method | kBET | Biology |
|--------|------|---------|
| Raw | — | baseline |
| UniFORM | 0.631 | destroys ChromA / CD45 / PD1 |
| Stage 1 analytic | 0.631 | excellent preservation |
| SpaNCy-GNN ensemble hybrid | 0.574 | better biology than UniFORM |
| **Stage 2 DDPM** (best so far) | **0.757** | +0.19% delta over Stage 1 |
| Stage 2 GNN | pending | target > 0.631 |
| Stage 2 OT-CFM | pending | target > 0.631 |

**Dual target:** kBET > 0.631 AND positive population |Δ| < 5% per marker.
