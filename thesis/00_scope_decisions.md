# Scope decisions — what goes in, and why

## The problem with "put everything in"

The project has produced roughly ten approaches (SpaNCy-GNN, ensemble, SpaNCy-Flow,
ResidualShiftModel, single-stage DL, Stage 1, Stage 2 GNN, OT-CFM, DDPM, MMD-ResNet, graph-fix,
neighbour-mean, …). Written up chronologically, that is a lab notebook, not a thesis.

**But the thesis plan already contains the fix.** Objective 2 *commits* to comparing graph-, flow- and
diffusion-based paradigms. The exploration was planned. Present it as a **designed comparison**:
same input (Stage 1), same metrics, same seeds, criteria fixed in advance. Do not present it as a
sequence of attempts. The rule for everything else stays:

- an experiment is in the main text if it **answers an objective** or **justifies a design choice**;
- everything else is one sentence in "Design rationale" (3.6), an appendix table, or nothing.

---

## Thesis-plan objectives (as approved) → how the thesis answers them

| # | Objective (thesis plan) | What answers it | Chapter |
|---|---|---|---|
| O1 | Study and benchmark existing MTI normalization methods | Raw / Z-score / ComBat / MXnorm / UniFORM on PRAD-CyCIF, all evaluation axes; Stage 1 as analytic reference-alignment baseline | 2.3, 5.1 |
| O2 | Develop a data-driven normalization method, investigating **graph-, flow- and diffusion-based** paradigms to identify the most effective | Two-stage design (Stage 1 + learned Stage 2); controlled comparison of Stage 2 paradigms: graph (GATv2 residual), flow (OT-CFM), diffusion (DDPM + SDEdit), plus per-cell no-graph and MMD-ResNet as reference arms | 3, 5.2 |
| O3 | Evaluate the proposed method against existing techniques, focusing on **batch-correction metrics** | kBET (primary) vs benchmarks, **plus** biology-preservation axes; matched-noise control shows why kBET alone is insufficient | 4, 5.3 |
| O4 | Detailed experimental analysis vs existing techniques | Seed repeats, matched-noise control, safeguard ladder, graph ablation, α trade-off | 5.4, 5.5 |

### Two things to check with the supervisor

1. **O3 wording: "focusing on batch-correction metrics".** Your strongest finding is that
   batch-correction metrics alone can be gamed: matched random noise scores kBET 0.692, above UniFORM.
   Do not narrow the evaluation to fit the old wording. Answer O3 as written (kBET is the primary
   batch metric) and *extend* it with biology-preservation axes, explaining why. Changing a plan's
   wording is usually fine if it is motivated, but tell the supervisor.
2. **O2: "identify the most effective solution".** "Most effective" needs a criterion **stated in
   Methods before the results**. Otherwise it looks like the winner was picked first and the
   criterion afterwards (the same problem as the moved ±5% target). Proposed criterion:
   - **Primary:** kBET (batch mixing).
   - **Constraints (must hold to be eligible):** silhouette ≥ Stage 1 − 0.01; 0 shape-distorted
     markers; positive population not worse than Stage 1; numerically stable across seeds.
   - **Ranking:** highest kBET among eligible methods. Ineligible methods are still reported, as
     points on the trade-off frontier.

   Be honest that these thresholds were set during the project (the silhouette bar was set before
   the graph-fix runs; the shape flag before CFM was measured), not at the thesis-plan stage.

---

## The paradigm comparison (core of O2) — evidence status

| Paradigm | Representative | Evidence now | Needed for a fair comparison |
|---|---|---|---|
| Graph-based | Stage 2 GATv2 residual corrector, α=0.6 | ✅ kBET 0.709 ± 0.006 (n=6), silhouette 0.360 ± 0.002, 0 distorted, ladder + noise control | done |
| (control) Per-cell, no graph | same model, `graph_mode='none'` | ✅ 3 seeds | done; tells you *what in the graph paradigm works* |
| Flow-based | OT-CFM (continuous normalizing flow) | ⚠️ single run: kBET 0.758, silhouette 0.350, **16/20 markers shape-distorted** | **3 seeds**; silhouette with the same 19-sample guard; per-sample pos-pop |
| Flow-based (abandoned) | Coupling flow (SpaNCy-Flow) | unstable, destroyed marginals, no final numbers | one paragraph: why discrete coupling flows were replaced by OT-CFM |
| Diffusion-based | DDPM + SDEdit | ⚠️ single run: kBET 0.735; pos-pop numbers were global-threshold artefacts; silhouette/shape pending | **3 seeds**; re-measure pos-pop (per-sample GMM), shape, silhouette |
| Prior art | MMD-ResNet (raw and on Stage 1) | ✅ 3 seeds | done |
| Optional, strong | Matched-noise control for CFM and DDPM | — | Tells you how much of *their* kBET is perturbation size. Without it, the comparison favours whichever method moves cells most |

**Likely conclusion** (only after the reruns): flow and diffusion reach higher kBET, but violate the
shape constraint (CFM) or remain unverified (DDPM). The graph-based corrector is the most effective
*eligible* method. Within it, the ablation shows the benefit comes from the per-cell residual and
safeguards, not from the graph. That is a legitimate answer to O2, provided it is stated plainly.

---

## Include / appendix / exclude

### IN — main text

| Item | Objective | Chapter |
|---|---|---|
| Benchmarks: raw, Z-score, ComBat, MXnorm, UniFORM | O1 | 5.1 |
| Stage 1 (analytic, KL-medoid ref, median / negative-peak shift) | O1/O2 (base of the method) | 3.2, 5.1 |
| Paradigm comparison: graph, flow (OT-CFM), diffusion (DDPM), no-graph, MMD-ResNet | O2 | 3.3, 5.2 |
| Multi-axis evaluation: kBET, per-sample silhouette, 1D shape, positive population | O3 | 4, 5.3 |
| Matched-noise control | O3/O4 | 5.3 |
| Safeguard ladder (masking, Huber, α, NT-Xent) | O4 | 5.4 |
| α trade-off (0 / 0.6 / 1.0) | O2 (operating point), O4 | 5.4 |
| Seed repeats (n ≥ 3) | all numbers | 4.5 |

### APPENDIX

| Item | Why only appendix |
|---|---|
| Graph ablation tables (old-graph / no-graph / graph-fix) and neighbour-mean | Summarised in one paragraph in 5.2/5.4 ("spatial inputs gave no structured gain"); tables here |
| CFM n_steps sweep, DDPM t_infer × cfg grid | Hyperparameter selection for O2 arms |
| Per-marker positive-population table (20 rows) | Summary in 5.3 |
| Per-group kBET tables | Summary in text |
| w_adv seed repeat | One sentence in 4.5 as the "why seeds" example |

### OUT — at most one sentence in "Design rationale" (3.6)

| Item | The one sentence it earns |
|---|---|
| SpaNCy-GNN (learned γ/β CycleDegradationModel, ensemble) | "A learned per-sample affine stage compressed distributions and mapped zero-inflated cells to positive values; it was replaced by an analytic alignment." |
| ResidualShiftModel | "Per-sample additive shifts cannot change local neighbourhood composition and reduced kBET, motivating a per-cell correction." |
| Single-stage DL (`spancy_shift_dl`) | Omit, or one clause: learning the alignment end-to-end did not beat the analytic Stage 1 |
| Sigmoid neg/pos bimodal blend | Why the negative-peak shift was chosen (ECAD var 0.856 vs 0.997) |
| 50 epochs | "Longer training (50 epochs) did not improve any metric." |
| Per-group silhouette, global-GMM pos-pop | Methods note on why the metrics are per-sample; old numbers not reported |
| Trend, both-peaks notebooks | Out unless they end up supporting a claim. Decide explicitly |

---

## Red flags to fix before writing

1. **Moved target.** "All markers |Δ| < 5%" → "≥ 50% of markers" after results. Do not present
   that as a pass/fail target, and drop "first method to clear the dual target". Use the O2 criterion
   above, stated up front.
2. **Positive-population definition ≠ UniFORM's.** Define ours precisely and do not compare with
   values printed in the UniFORM paper (our best reproduction attempt: MAE 16.7 pp).
3. **"Spatial" is not supported.** no-graph ≈ graph; `scene_id` is reused across patients, so the
   published graph mixed patients; neighbour-mean input was also rejected. Call Stage 2 the
   "graph-based paradigm" (fits O2), and report that its graph adds no measurable kBET. Avoid
   "spatial" in the title; consider whether the SpaNCy name needs a footnote.
4. **Most of the kBET gain is perturbation size.** About +0.018 of the +0.09 is structured. Report both.
   The same must be checked for CFM/DDPM (matched-noise control above).
5. **Single runs.** CFM and DDPM are single runs while the graph arm has 6 seeds. That asymmetry
   would bias O2. Rerun with 3 seeds, or label them clearly and weaken the conclusion.
6. **Novelty.** CellOT already uses OT on multiplexed protein imaging (perturbations); MMD-ResNet is
   prior art for a per-cell MMD corrector. Claim the *two-stage combination, the paradigm comparison
   and the evaluation protocol*, not the components.
7. **One dataset.** Main limitation. An external dataset would strengthen O3 more than any further
   ablation.
8. **Training budget** is 10 epochs, originally a "quick test". State that 50 was tested and rejected.

---

## Contributions (draft, aligned with the objectives)

1. **(O1)** A benchmark of five existing MTI normalizers on PRAD-CyCIF under a common multi-axis
   evaluation.
2. **(O2)** A two-stage normalizer, analytic per-marker reference alignment plus a learned per-cell
   residual corrector with biology safeguards, and a controlled comparison of graph-, flow- and
   diffusion-based Stage 2 paradigms on the same input.
3. **(O3)** Evidence that batch-correction metrics alone are insufficient: matched random noise beats
   UniFORM on kBET. Hence a multi-axis protocol (kBET + per-sample silhouette + 1D shape + positive
   population).
4. **(O4)** Controls and ablations showing what the learned gain consists of: matched-noise and
   shuffled-delta controls, a safeguard ladder, graph ablation, and seed repeats.
