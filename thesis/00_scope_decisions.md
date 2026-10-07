# Scope decisions — what goes in, and why

## The problem with "put everything in"

You are right to be uneasy. The project has produced roughly ten approaches (SpaNCy-GNN, ensemble,
SpaNCy-Flow, ResidualShiftModel, single-stage DL, Stage 1, Stage 2 GNN, OT-CFM, DDPM, MMD-ResNet,
graph-fix, neighbour-mean, …). Writing them all up chronologically produces a lab notebook, not a
thesis. An examiner reads that as: *no hypothesis, try things until something works.*

A thesis is an **argument**. Each experiment earns its place only if it does one of two jobs:

- **supports a conclusion** (a claim in Ch. 6), or
- **justifies a design decision** in the final method (Ch. 3).

Everything else is development history. A sentence or a footnote at most, or nothing.

Exploring many options is fine and normal. What makes it look aimless is *presenting* it unordered.
Presented as "we considered A, B, C for reason X; A failed because Y, which motivated design Z",
the same history reads as method.

---

## Proposed research question

> **Can a learned per-cell correction, added on top of an analytic reference-based alignment, improve
> cross-batch mixing of CyCIF single-cell data without distorting biological signal — and how must
> such a method be evaluated so that the improvement is real?**

Sub-questions (each maps to one results chapter section):

1. **RQ1:** How far does a training-free, per-marker reference alignment (Stage 1) get, compared
   with existing normalizers (UniFORM, ComBat, Z-score, MXnorm)?
2. **RQ2:** Does a learned residual corrector (Stage 2) improve batch mixing beyond Stage 1, and
   how much of that improvement is genuine correction rather than perturbation?
3. **RQ3:** Which components of Stage 2 are necessary to keep biology intact?
4. **RQ4 (evaluation):** Why is a single batch-mixing metric (kBET) insufficient, and what
   multi-axis evaluation is needed?

RQ4 is arguably the most original, robust finding of the project: the matched-noise control
showed that random noise scores kBET 0.692, which is better than UniFORM. Do not hide it. It is
a contribution.

---

## Include / appendix / exclude

### IN — main text

| Item | Role in the argument | Chapter |
|---|---|---|
| Stage 1 (analytic, KL-medoid reference, median shift / negative-peak shift) | Baseline + base of the method; answers RQ1 | 3, 5.1 |
| Stage 2 per-cell residual corrector (α = 0.6, MMD + Huber + bimodal mask) | The method; RQ2 | 3, 5.2 |
| Benchmarks: UniFORM, ComBat, Z-score, MXnorm, raw | Context for RQ1/RQ2 | 5.1 |
| Multi-axis evaluation: kBET, per-sample silhouette, 1D shape ratios, positive population | RQ4; needed to read every result | 4, 5.5 |
| Seed repeats (n ≥ 3) for every learned arm | Credibility of all numbers | 4.5, everywhere |
| Matched-noise control (Gaussian + shuffled delta) | RQ2: how much of the lift is learned | 5.3 |
| Safeguard ladder (no-mask, no-Huber, α = 1, no-NT-Xent) | RQ3: which parts matter | 5.4 |
| MMD-ResNet (raw and on Stage 1) | Direct prior art. Shows Stage 1 is what makes a per-cell MMD corrector work and our safeguards are what protect biology | 5.2 |
| α trade-off (0 / 0.6 / 1.0) | Justifies the operating point; shows the kBET-vs-biology frontier | 5.2 |

### APPENDIX — supporting, not central

| Item | Why only appendix |
|---|---|
| Graph ablation (old-graph vs no-graph vs graph-fix) | Needed to justify *not* calling the method spatial. The result itself is negative, so state it in one paragraph in 5.4 and put the tables in the appendix |
| OT-CFM and DDPM Stage 2 | Single runs, different design family. One short "alternative Stage 2 families" section is OK: CFM's *transport reshapes marginals vs per-cell keeps marginals* contrast is interesting. Full sweeps → appendix. **If time is short, drop DDPM entirely** (its pos-pop numbers were never re-measured with per-sample GMM) |
| Per-marker positive-population table (20 rows) | Too detailed for the main text; summary in 5.5 |
| Per-group kBET tables | Summary in text, full table in appendix |
| 10 vs 50 epochs | One sentence in 4.x ("50 epochs tested, no gain") |
| w_adv (CE) seed repeat | One sentence: single-seed gain didn't replicate. Good example of why seeds matter (4.5) |

### OUT — at most one sentence in "Design rationale" (3.6)

| Item | The one sentence it earns (if any) |
|---|---|
| SpaNCy-GNN (CycleDegradationModel, learned γ/β, ensemble) | "A learned per-sample affine stage compressed distributions and mapped zero-inflated cells to positive values; we replaced it with an analytic alignment." |
| SpaNCy-Flow (coupling flows) | "Invertible flows with an MMD objective were unstable and reshaped marginals" — or omit |
| ResidualShiftModel | "Per-sample additive shifts cannot change local neighbourhood composition and reduced kBET" — this is actually a useful motivation for *per-cell* Stage 2; keep the sentence |
| Single-stage DL (`spancy_shift_dl`) | Omit |
| Sigmoid neg/pos bimodal blend | One sentence on why the negative-peak shift was chosen (sigmoid blend compressed ECAD, var ratio 0.856 vs 0.997) |
| Per-group silhouette, global-GMM positive population | Methods note on *why* the metric is per-sample (batch confound); do not report the old numbers |
| Neighbour-mean input (`nbr_confirm`) | Rejected 2026-10-07: its kBET gain is explained by a larger correction. One sentence next to the graph ablation ("spatial inputs gave no structured gain"); table in appendix D |
| Trend, both-peaks notebooks | Out unless they end up supporting a claim. Decide explicitly |

---

## Red flags to fix before writing

These would be found by an examiner. Fix or state them openly.

1. **The target was moved after seeing results.** "All markers |Δ| < 5%" became "≥ 50% of markers"
   once nothing met it, and the project then calls α = 0.6 "the first method to clear the dual
   target". That is circular. **Fix:** do not present a pass/fail target. Report per-marker positive
   population descriptively and compare methods on it. Keep "first to clear the target" out of the
   thesis.
2. **Positive-population definition ≠ UniFORM's.** Ours transfers a per-sample raw threshold.
   UniFORM uses one global threshold on pooled normalized data, and our best attempt does not
   reproduce their paper (MAE 16.7 pp). **Fix:** define our metric precisely, say it differs, and do
   not compare our numbers with numbers printed in the UniFORM paper.
3. **"Spatial GNN" is not supported.** no-graph ≈ old-graph on kBET, and `scene_id` is reused across
   patients, so the published k-NN graph mixed patients. **Fix:** describe Stage 2 as a per-cell
   residual corrector implemented with a GATv2 encoder. State that the graph adds no measurable kBET
   and that it was kept because the no-graph variant was slightly worse on silhouette at every seed.
   Do not use "spatial" in the title. Consider renaming the method (SpaNCy = *Spatial* Neighborhood…).
4. **Most of the kBET gain is perturbation size.** Of the +0.09 lift, about +0.018 is structured,
   learned correction (noise control). **Fix:** report both numbers. Frame the contribution as
   "+0.018 over matched noise at better biology", not "+0.09".
5. **Single-run numbers.** CFM 0.758, DDPM 0.735 and the old 0.708/0.365 are single runs. **Fix:**
   use 0.709 ± 0.006 / 0.360 ± 0.002. Label CFM/DDPM as single-run or rerun them with 3 seeds.
6. **Novelty claims.** "No prior OT work for CyCIF": CellOT already evaluates on multiplexed protein
   imaging (perturbations, not batch correction). MMD-ResNet is direct prior art for the corrector.
   **Fix:** claim the *combination and the evaluation*, not the components.
7. **One dataset.** All results are PRAD-CyCIF. **Fix:** state it as the main limitation. If there
   is time, one external dataset (UniFORM also used CRC-ORION and TMA-Lunaphore) would strengthen
   the thesis more than any further ablation.
8. **Training budget.** The final model is 10 epochs, originally labelled "quick test". 50 epochs was
   tested and rejected, so state that so it does not look unfinished.

---

## Contributions (draft, honest version)

1. A two-stage normalizer for CyCIF: an analytic per-marker reference alignment, plus a learned
   per-cell residual corrector with explicit biology safeguards (bimodal masking, Huber anchoring,
   blend α).
2. Evidence that the analytic stage alone matches the best existing normalizer (UniFORM) on batch
   mixing, and that it is what makes a per-cell MMD corrector work (MMD-ResNet +0.16 kBET when
   given Stage 1 input).
3. Controls showing which part of the learned gain is real: matched-noise and shuffled-delta
   controls, a safeguard ablation ladder, and seed repeats.
4. A multi-axis evaluation protocol (kBET + per-sample silhouette + 1D shape + positive population),
   motivated by the finding that kBET alone rewards random noise.
