# SpaNCy-Shift — Methods Reference

> **Status (updated 2026-06-07):** verified against `spancy_shift.py` source and CLAUDE.md.
> Supersedes the pre-2026-06-04 version (which described piecewise-linear bimodal shifts,
> a global+local positive-population threshold, and the now-flipped "trade-off unavoidable"
> conclusion — all stale).
>
> **2026-06-11 change (production path):** Stage 1 bimodal markers now use a single **negative
> (leftmost) peak shift** (pure translation) instead of the sigmoid neg/pos blend — fixes ECAD shape
> compression (var_ratio 0.856 → 0.997) at a small fraction cost (ECAD pos-pop +1.12% → +2.44%, still
> passing). **kBET re-measured (negative-peak run, 10 epochs, α=0.6): Stage 2 = 0.708** (was 0.7117,
> −0.003 ≈ noise, still ≫ UniFORM 0.6315); **Stage 1 = 0.620** (was 0.6322 — the ECAD positive-pop
> alignment it gave up costs ~0.012, now just below UniFORM; the GNN recovers it, +0.088 lift vs
> +0.080). **Silhouette re-measured (19 samples, deterministic ≥50-cell guard): Raw = 0.367, Stage 1
> = 0.367 (= raw; the old −0.017 gap was entirely ECAD's bimodal sigmoid shift), Stage 2 α=0.6 = 0.365
> (≈ raw, was 0.333).** Net: a three-way win — kBET ≫ UniFORM, 1D shape clean (ECAD now too),
> silhouette ≈ raw. Tables below updated for the operating point (α=0.6) and Stage 1; α=0.3/1.0 rows
> are from the prior sigmoid run (not re-swept).

All methods share the same **Stage 1** analytic baseline. Stage 2 is where the approaches differ.
**For the article**, only Stage 1 + Stage 2a (GNN) and the statistical baselines (UniFORM, ComBat,
Z-Score, MXnorm) are in scope; Stage 2b/2c (OT-CFM, DDPM) are exploratory and reserved for the
thesis report.

---

## Overall Pipeline

```
Raw CyCIF data (1.76M cells × 20 markers, 20 samples, 7 batches)
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
│   │ ARTICLE  │   │ thesis   │   │   thesis               │ │
│   └──────────┘   └──────────┘   └────────────────────────┘ │
│  Output: normalized                                         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Diagnostics: kBET · per-sample silhouette · positive-population Δ
           · 1D shape preservation · batch adj-R²
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
    │       Reference = medoid (lowest mean KL to all others)
    │       = the MOST REPRESENTATIVE sample (NOT "least batch-affected").
    │       Output: ref_sample_per_marker {marker → sample_id}
    │
    ▼
Per-marker, per-sample shift in log1p space:
    │
    ├── UNIMODAL marker (_shifts_unimodal / _apply_unimodal):
    │       shift = median(ref) − median(sample)      ← MEDIAN shift, pure translation
    │       X_shifted = X_sample + shift
    │       (distance-preserving → no shape change)
    │
    └── BIMODAL marker (_shifts_bimodal → _apply_unimodal):  [updated 2026-06-11]
            Find neg (leftmost) peak in both sample & ref.
            shift_neg = ref_neg_peak − sample_neg_peak
            X_shifted = X_sample + shift_neg          ← NEGATIVE-PEAK shift, pure translation
            (distance-preserving → no shape change; var_ratio ~1.0)
            The whole sample is translated so its negative/background mode aligns to
            the reference's. The positive-peak shift is NOT applied — Stage 1 no longer
            corrects the neg→pos spacing.
            WHY (replaces the old sigmoid neg/pos blend): the blend translated the two
            populations by different amounts, which narrowed the inter-peak gap and
            compressed bimodal width (ECAD var_ratio 0.856). A single negative-peak shift
            is a pure translation → var_ratio 0.997, and ECAD's positive-population Δ stays
            in-target (+1.12% → +2.44%, still < 5%). The positive population's residual
            multivariate offset is left to Stage 2 (but the GNN excludes bimodal markers
            from its MMD loss, so in practice ECAD ≈ this Stage 1 shift).
            Old code path `_apply_bimodal` (sigmoid blend) is retained but unused.
    │
    ▼
X_base = clip(expm1(X_shifted), 0, None)   [stored in adata.layers['normalized_base']]

Result: kBET ≈ 0.631  (matches UniFORM with zero learned parameters)
Returns: (adata_out, is_bimodal, thresholds)
```

---

## Stage 2a — GNN (Spatial GATv2)  ← ARTICLE METHOD
**File:** `spancy_shift.py` → `GNNStage2`, `train()`, `normalize_adata()`

The GNN operates at the **cell level** using spatial context. A cell's correction depends on its
neighbours — something per-sample shifts cannot do.

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
near-zero init   L2-norm          + GRL (gradient reversal)
outputs delta    z_proj           batch_logits
    │             │                  │
    │          NT-Xent            Cross-entropy
    │          L_contrast         L_adv
    │
    ▼  Huber(delta, 0) → L_recon   MMD(X_out, batch) → L_mmd
    │                               (bimodal markers excluded from MMD only)
    │
    ▼
X_out = X_scaled + hybrid_alpha · delta
    │
    ▼  inverse_scale → expm1 → clip(≥0)
X_normalized   [stored in adata.layers['normalized']]

Losses: L_recon(0.1) + L_contrast(0.5) + L_adv(0.3) + L_mmd(1.0)
Sampler: SceneBasedSampler — one scene per batch per step
         ensures spatial neighbours co-occur in mini-batch
GRL lambda ramps 0 → grl_max over training.
Training: N_EPOCHS = 10 (canonical — see "Epoch count" below).
```

**Key knob:** `hybrid_alpha` — 0 = pure Stage 1, 1 = full GNN delta. Inference-only; the same
trained model is re-evaluated at any α with no retraining. **Operating point: α = 0.6.**

**IMPORTANT — bimodal handling is asymmetric between train and inference:**
- **Training:** bimodal markers are excluded from the **MMD loss** (`unimodal_mask = ~is_bimodal`),
  so no batch-alignment gradient reaches their decoder output.
- **Inference (`normalize_adata`):** `is_bimodal` is **discarded** (`adata_out, _, _ = ...`) and
  `hybrid_alpha · delta` is applied to **all 20 markers**, bimodal included.
- Bimodal markers end up ≈ Stage 1 **only because** their learned delta is ~0 (no MMD gradient +
  Huber recon pushes it toward 0) — NOT because they are masked at inference.
- **Paper wording:** "bimodal markers are excluded from the second stage's MMD objective, so the
  learned residual leaves their shape essentially unchanged." Do NOT write "masked out at inference"
  or "passed through unchanged."

**Epoch count:** 10 epochs is canonical. Retraining at 50 epochs was tested (2026-06-06) and is
marginally *worse* (kBET 0.717→0.707, silhouette 0.333→0.330, HLADRB1 worse, g4 regresses). Stage 2
is a near-zero-init residual under an MMD loss; short training keeps deltas small and clean. Report
the epoch count in Methods, and make sure kBET and silhouette in any table come from the same run/α.

---

## Stage 2b — OT-CFM (Conditional Flow Matching)  [thesis only]
**File:** `spancy_shift_cfm.py` → `FlowMLP`, `train_cfm()`, `normalize_adata_cfm()`

Learns a **velocity field** that transports cells from each batch toward the reference batch along
straight-line optimal-transport paths. No spatial graph, no adversarial training.

```
X_base → log1p → RobustScaler → X_scaled
    │
    ▼
Identify reference batch (majority vote over per-marker ref samples)

TRAINING (one step):
    ├─ Sample source cells x_0 from non-ref batches (batch-balanced)
    ├─ Sample target cells x_1 from reference batch
    ▼
OT Coupling (Hungarian on 256×256 L2 cost matrix) → matched (x_0, x_1) pairs
    │   Bimodal dims are masked from the OT cost and velocity.
    ▼
Interpolate at random t ~ U(0,1):  x_t = (1−t)·x_0 + t·x_1 + σ·noise   (σ=0.01)
    ▼
┌─────────────────────────────────────────────────┐
│  FlowMLP                                        │
│  batch_emb(32d) + t_emb(64d) + x_t(20d)         │
│      → AdaLN residual × 6 (hidden=512)          │
│      → velocity (20d), zero-init → identity     │
└─────────────────────────────────────────────────┘
    ▼  Loss = MSE(velocity_pred, x_1 − x_0)

INFERENCE: Euler ODE integration t=0→1 with n_steps steps → inverse_scale → expm1
Key knob: n_steps ∈ {5, 20, 50}
```

---

## Stage 2c — DDPM + SDEdit  [thesis only]
**File:** `spancy_shift_ddpm.py` → `DenoisingMLP`, `DDPMScheduler`, `train_ddpm()`, `normalize_adata_ddpm()`

Learns the per-batch **score function**; at inference uses SDEdit — add partial noise, then
reverse-diffuse toward the reference batch with classifier-free guidance (CFG).

```
X_base → log1p → RobustScaler → X_scaled
    ▼
Linear beta schedule β_1=1e-4 → β_T=0.02, T=200;  ᾱ_t = ∏(1−β_i)

TRAINING:  x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε ;  CFG dropout 10% (null batch token)
    ▼
┌─────────────────────────────────────────────────┐
│  DenoisingMLP                                   │
│  sin/cos time_emb(256d) + batch_emb(32d) + x_t  │
│      → AdaLN residual × 6 (hidden=512)          │
│      → ε_pred (20d), zero-init output           │
└─────────────────────────────────────────────────┘
    ▼  Loss = MSE(ε_pred, ε)

INFERENCE (SDEdit): noise to t_infer, reverse-diffuse with CFG
    ε_guided = ε_uncond + cfg_scale·(ε_ref − ε_uncond)
    → inverse_scale → expm1
Key knobs: t_infer ∈ {10, 30, 80}, cfg_scale ∈ {1.0, 1.5, 3.0}
```

---

## Diagnostics

### Positive-Population Preservation (per-sample GMM)
**Function:** `positive_population_table()` in `spancy_shift.py`

```
log10(x + 1) transform on both raw and normalized.

For each marker, for each sample (≥10 cells):
    thr_local = 2-component GMM threshold fitted on THIS SAMPLE's RAW cells
                (_gmm_threshold; same threshold applied to raw AND normalized)
    pct_pos_raw  = % raw cells  > thr_local
    pct_pos_norm = % norm cells > thr_local
    delta        = pct_pos_norm − pct_pos_raw          (target |Δ| < 5%)

    density_ratio = density at threshold bin / density at peak bin   (0=valley, 1=peak)
    reliable      = density_ratio < 0.3
```

The per-sample local threshold (not a global one) makes Δ robust to inter-sample intensity
variation. **HEADLINE metric:** the full per-marker **mean Δ ± SD over all 20 markers** (with a
`pass_5pct` flag and within-±5% count). Read mean Δ *with* its SD — a high SD (CD45 52%, CDX2 30%,
DAPI_R1 28%) means a unimodal marker whose GMM threshold sits near the histogram peak, so the large
|Δ| is threshold instability, not real distortion.

`summarize_positive_population(min_reliable_frac=0.5)` applies the reliability filter
(`density_ratio < 0.3`) but is **DEMOTED to secondary/reference only** — it drops most unimodal
markers (often leaving ~3), too few to compare methods. Do not use it as the primary pass/fail gate.

> **Methodology note:** earlier versions used a single global GMM (or global-on-normalized +
> local-on-raw) threshold. Those produced artefactual deltas (CD20 −31%, ChromA −40%, NOTCH1 −38%)
> from inter-sample pooling. The per-sample GMM (commit 6df17c9, reliability filter 2026-06-01)
> gives the correct values below (CD20 −3.77% PASSES, NOTCH1 +3.58% PASSES). Any table showing the
> old global-GMM numbers is wrong.

### Per-sample Silhouette (biology preservation, 20D)
3 cell types matching the UniFORM paper (Wang et al. 2025): tumor epithelial (ECAD+), immune
(CD45+), non-immune stromal (aSMA+). Labels fixed once on raw data via per-anchor 2-component GMM
threshold (priority ECAD > CD45 > aSMA; none-positive = unassigned/excluded) and reused for every
method. `silhouette_score(X_log_20D, labels, metric='euclidean')` is computed **within each sample,
then averaged** (NOT per clinical group — the per-group version was batch-confounded and inflated
the raw baseline). Guards: skip sample with <15 labeled cells; require ≥2 cell types each with ≥5
cells; subsample 3000 cells/sample. Higher = better.

### 1D Shape-Preservation Diagnostic
**Notebook cell 6b (explore) / 5g (benchmark).** Per (marker, sample), location-invariant ratios vs
raw in log1p space: `peak_ratio` (mode height), `var_ratio`, `iqr_ratio`; 1.0 = preserved. Catches
1D marginal reshaping that positive-pop (fractions) and silhouette (20D clusters) are both blind to.
Stage 1 (pure shift) is the built-in control (≈1.0 everywhere — including the lone bimodal marker
ECAD since the 2026-06-11 negative-peak change: var_ratio 0.856 → 0.997). Distortion flag = peak<0.8
or var outside [0.8, 1.25]. **Known gap:** the flag uses var, not iqr → it over-counts tail-driven var
inflation (e.g. MXnorm). Report iqr_ratio alongside var_ratio.

### Batch adj-R²
**Function:** `per_marker_batch_r2()`. Regress each marker on batch one-hot labels; report adjusted
R². Lower = less residual batch effect (target < 0.05). Raw 0.254 → Stage 1 0.0061 → GNN 0.0059.

### Histogram figures (per-sample overlays)
**Notebook cell:** sec 7 (`spancy_shift_explore`) and the matching cell in `mxnorm_benchmark`.
One colored line per sample (tab20), plotted in log1p space, grid = markers (rows) × methods (cols).

- **Raw column** via a `RAW = '__RAW__'` sentinel in `layers_to_plot` + a `_get_X(key)` helper that
  returns `adata.X` for the sentinel (raw is not a layer) and `adata.layers[key]` otherwise.
- **Cross-PDF axis matching (updated 2026-06-09):** bin edges (`_marker_edges`) and the y-limit
  (`_marker_ymax` → explicit `ax.set_ylim`) are computed from **RAW only**, per marker. Raw is
  identical in both notebooks, so the shift PDF and the benchmark PDF share identical x-bins and
  y-axis per marker. Replaces the old `sharey='row'` (which only shared within one figure). Valid
  because each curve is a single sample's histogram and Stage-1/2 normalization is mostly a per-sample
  shift, so peak heights are preserved and raw-derived limits don't clip normalized curves. Switch to
  `density=True` if the two notebooks ever use different N per sample.
- **Article = one combined side-by-side figure**, not two PDFs. The GNN `normalized` layer and the
  benchmark layers are produced in separate Colab runtimes; bridge them by saving the GNN array to a
  shared Google Drive (`np.save`/`np.load`) — or download/upload — then `adata.layers['gnn'] = …`
  (guard `assert gnn.shape == adata.shape`), add `('gnn', 'SpaNCy-Shift (GNN α=0.6)')` to
  `layers_to_plot`, and re-run the single benchmark histogram cell. Never crop/paste rendered panels
  (axis misalignment + rasterization). Column order: Raw | UniFORM | ComBat | Z-Score | MXnorm | GNN.

### kBET
Computed via `pegasus.calc_kBET()` per clinical group (5 groups, each pairing samples from different
batches; rep="umap"). Acceptance rate = fraction of cells whose local neighbourhood batch
composition matches the global expectation. Higher = better.

---

## Results

### kBET — GNN hybrid_alpha sweep (single 10-epoch model, no retraining)

Negative-peak run (2026-06-11), α=0.6 and Stage 1 (α=0.0) re-measured; α=0.3/1.0 carried from the
prior sigmoid run (not re-swept):

| α | kBET | shape distorted | silhouette | pos-pop pass |
|---|------|-----------------|------------|--------------|
| 0.0 (Stage 1) | **0.620** | 0/20 | **0.367** (= raw) | 11/20 |
| 0.3 (sigmoid run) | 0.672 | 0/20 | ~0.341† | 11/20 |
| **0.6 (operating point)** | **0.708** | **0/20** | **0.365** | **11/20** |
| 1.0 (sigmoid run, max-kBET) | 0.777 | ~1–5 mild | 0.311† | 11/20 (≈noise) |

† silhouette: Stage 1 and α=0.6 are the negative-peak run with the deterministic 19-sample guard
(Raw = 0.367); α=0.3/1.0 are the prior sigmoid run (18-sample, not re-swept) and are not directly
comparable in absolute value — read them as "0/20 and ~1–5 distorted," not as a like-for-like level.
Per-group kBET at α=0.6 (negative-peak run): g1 0.916, g2 0.698, g3 0.632, g4 0.597, g5 0.699.

**α=0.6 is the operating point** — kBET 0.708 (> UniFORM 0.6315, +0.077) with biology preserved on all
three axes (silhouette 0.365 ≈ raw 0.367; 1D shape clean — now including ECAD; positive-pop = Stage 1).
This is the **first method to clear the revised dual target** (kBET > 0.631 AND biology preserved). The GNN
lifts kBET +0.088 over its own Stage 1 (0.620 → 0.708), recovering the ECAD positive-population mixing
that the shape-preserving Stage 1 gives up. α=1.0 is the max-kBET variant (best kBET in the project,
at a measurable silhouette cost).

### Positive-Population Δ — full per-marker table (per-sample GMM, mean ± SD over 20 samples)

| Marker | Stage 1 mean Δ | Stage 1 SD | Stage 2 GNN mean Δ | Stage 2 GNN SD | Pass (<5%) |
|--------|----------------|------------|---------------------|----------------|------------|
| ECAD   | +2.44% | 11.98% | +2.44% | 11.98% | ✅ |
| FOXA1  | +0.28% | 24.71% | −0.10% | 25.19% | ✅ |
| p53    | −0.34% | 24.43% | +0.17% | 24.71% | ✅ |
| CD3    | −1.12% | 34.12% | −0.91% | 34.46% | ✅ |
| CK14   | −2.98% | 19.23% | −2.21% | 19.77% | ✅ |
| CD31   | −3.02% | 11.00% | −3.01% | 11.98% | ✅ |
| CD56   | −3.70% | 23.64% | −3.76% | 24.35% | ✅ |
| CD20   | −3.77% | 30.82% | −4.53% | 30.85% | ✅ |
| PD1    | +2.21% | 35.57% | +2.01% | 35.74% | ✅ |
| NOTCH1 | +3.58% | 15.30% | +3.05% | 15.63% | ✅ |
| Ki67   | +3.22% | 23.97% | +0.89% | 24.22% | ✅ |
| EPCAM  | −5.97% | 24.49% | −6.07% | 25.27% | ❌ |
| GZMB   | −6.69% | 23.08% | −6.37% | 23.60% | ❌ |
| CD45RA | −7.81% | 20.54% | −7.23% | 20.72% | ❌ |
| HLADRB1| +8.97% | 20.52% | +9.21% | 21.73% | ❌ |
| DAPI_R1| −12.25%| 28.18% | −13.00%| 28.91% | ❌ |
| CDX2   | −13.32%| 30.33% | −13.42%| 31.04% | ❌ |
| ChromA | −13.36%| 24.35% | −12.53%| 24.65% | ❌ |
| CD45   | +13.84%| 52.65% | +13.33%| 52.73% | ❌ |
| aSMA   | +13.98%| 29.62% | +13.69%| 29.88% | ❌ |

**Stage 1: 11/20 pass, 9/20 fail. Stage 2 GNN: same 11/20 pass — the GNN is essentially neutral on
1D marginals** (it corrects the 20D joint, not the marginals). The 9 failing markers are dominated by
high inter-sample SD (threshold instability on unimodal markers), not systematic distortion: e.g.
CD45 mean +13.8% but SD 52.7% with sign-inconsistent per-sample deltas. UniFORM exceeds ±5% on a
comparable-or-larger set by the same measure.

### Per-sample Silhouette (20D, 3 cell types)

Two raw baselines (different guard-passing sample sets — compare WITHIN each block only):

**shift-repo run (19 samples, negative-peak, deterministic ≥50-cell guard; raw = 0.3669):**
| Method | Silhouette | Δ vs raw |
|--------|------------|----------|
| Raw | 0.3669 | — |
| Stage 1 (analytic) | **0.3670** | +0.0001 (= raw) |
| **Two-stage GNN (α=0.6)** | **0.3649** | −0.0020 |

**mxnorm_benchmark run (raw = 0.364; re-run pending with the matching ≥50-cell guard → expect raw ≈ 0.367):**
| Method | Silhouette | Δ vs raw |
|--------|------------|----------|
| Raw | 0.364 | — |
| UniFORM | 0.364 | ≈0.000 |
| ComBat | 0.356 | −0.008 |
| Z-Score | 0.347 | −0.017 |
| MXnorm | −0.028 | −0.392 |

After the negative-peak change, **Stage 1 matches raw exactly and GNN α=0.6 is essentially raw-neutral**
(−0.002) — the old −0.017 Stage 1 gap was entirely ECAD's bimodal sigmoid shift. The GNN now joins
UniFORM in the raw-neutral silhouette cluster while beating it on kBET; MXnorm alone collapses cluster
structure. **Once the benchmark cell re-runs with the same deterministic guard, both blocks share one
19-sample set and one raw (≈0.367) → the two blocks collapse into a single table.**

### 1D Shape preservation (cell 5g / 6b)

| Method | distorted | peak (all) | var (all) | note |
|--------|-----------|-----------|-----------|------|
| UniFORM | **0/20** | 1.002 | 1.002 | pure translation, distance-preserving |
| **GNN α=0.6** | **0/20** | ~1.02 | ~0.96 | matches UniFORM on shape |
| GNN α=1.0 | ~1–5 mild | ~1.0 | 0.970 | mild, far below CFM |
| ComBat | 15/20 | 0.915 | 1.664 | mildest of the three baselines |
| MXnorm | 15/20 | 1.051 | 2.784 | var inflation is TAIL-driven (iqr mild) |
| Z-Score | 19/20 | 0.800 | 2.735 | flattens mode AND broadens bulk — worst |

Two camps: **shape-clean {UniFORM, GNN α=0.6}** vs **shape-distorting {ComBat, MXnorm, Z-Score}**.
GNN α=0.6 matches UniFORM on shape AND beats it on kBET (0.717 vs 0.631) — the headline comparison.
(The distortion flag uses var not iqr, so it over-counts MXnorm's tail-driven inflation; state this.)

### Stage 2 alternatives (thesis only — NOT in the article)

| Method | kBET | 1D shape | silhouette | verdict |
|--------|------|----------|------------|---------|
| OT-CFM | 0.7576 | 16/20 wrecked | 0.350 (preserved) | best kBET but reshapes marginals |
| DDPM + SDEdit | 0.7352 | (re-measure under per-sample GMM) | pending | trades shape/biology for kBET |

CFM and the GNN distort **opposite** biology: CFM transports whole distributions → keeps the 3
clusters separated (silhouette ok) but reshapes every marginal (1D shape wrecked); the GNN moves
cells individually → keeps marginals (1D shape clean) but mildly erodes cluster boundaries at high α.
Which matters depends on downstream use (phenotyping ↔ silhouette; gating ↔ 1D shape). The earlier
"CFM destroys biology (CD45 −23%, PD1 −30%)" claim was a **global-threshold artifact** — under
per-sample GMM, CFM's positive-pop ≈ Stage 1. Its real cost is 1D shape.

---

## Summary table (article scope)

| Method | kBET | Pos-pop pass | 1D shape | Silhouette | One-line verdict |
|--------|------|--------------|----------|------------|------------------|
| **GNN α=0.6** | **0.708** | 11/20 (=S1) | clean 0/20 | 0.365 | clears dual target; clean on all axes incl. silhouette ≈ raw, beats UniFORM kBET |
| GNN α=1.0 | 0.777 | 11/20 (≈noise) | ~1–5 mild | 0.311† | max-kBET variant; silhouette cost (†sigmoid run) |
| Stage 1 (analytic) | 0.620 | 11/20 | clean 0/20 | 0.367 | provably-correct 1D baseline, no learning; silhouette = raw |
| UniFORM | 0.631 | comparable/worse | clean 0/20 | 0.364 | best linear baseline; shape-clean but kBET capped |
| ComBat | 0.286 | — | 15/20 | 0.356 | poor kBET, reshapes markers |
| Z-Score | 0.293 | — | 19/20 | 0.347 | poor kBET, worst shape |
| MXnorm | 0.244 | — | 15/20 (tail) | −0.028 | poor kBET, collapses cluster structure |

**Conclusion (flipped from the old "trade-off unavoidable"):** a spatial per-cell GNN residual at
α=0.6 improves batch mixing beyond UniFORM with biology essentially untouched on all three axes.
Per-cell deltas rearrange the 20D joint (what kBET reads) *within* each marginal's envelope, instead
of transporting and reshaping marginals as the linear baselines and CFM do.

---

## Abandoned Approaches

- **SpaNCy-GNN (`../spancy.py`)** — full GNN with learned `CycleDegradationModel` (gamma/beta) +
  adversarial + cross-batch contrastive. Ensemble hybrid reached kBET 0.574 (below UniFORM).
  Train/inference mismatch; CycleDegradation needs many epochs to converge.
- **SpaNCy-Flow (`spancy_flow.py`)** — cycle-block normalizing flow + MMD. Crushed distribution
  shapes (ECAD → spike), 40 min/10 epochs, conflicting losses.
- **ResidualShiftModel** — per-sample additive shifts with MMD. Consistently degraded kBET
  (0.631 → 0.535); per-sample shifts move all cells uniformly → cannot improve local neighbourhood
  mixing (what kBET measures).
