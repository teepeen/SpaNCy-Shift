# Thesis outline — template

Generic master's-thesis structure, about 60–80 pages. Adapt to the university guideline.
Page counts are rough targets. `[ref]` keys match `references.bib` and `03_reference_map.md`.

---

## Front matter
- Title. Avoid "spatial" (see scope red flag 3). Draft: *Two-stage batch normalization of
  multiplexed tissue imaging data: analytic alignment with a safeguarded learned correction*
- Abstract (+ Finnish tiivistelmä if required). Write it LAST. Use mean ± SD numbers only.
- Abbreviations: CyCIF, MTI, kBET, MMD, GMM, KL, GATv2, NT-Xent, GRL, OT, CFM, DDPM

---

## 1 Introduction (4–6 p)
**Purpose:** problem → why it matters → gap → research question → contributions → structure.

1.1 Multiplexed tissue imaging and why cohorts need normalization. `[lin2018cycif, hickey2022primer]`
1.2 Batch effects: what they are and why they mislead. `[leek2010batch]`
1.3 Gap: existing MTI normalizers are per-marker 1D (UniFORM, ComBat, MXnorm) and cannot fix
    multivariate structure. Learned correctors (MMD-ResNet) can, but risk distorting biology.
    `[wang2025uniform, harris2022slide, shaham2017mmdresnet]`
1.4 Objectives O1–O4, worded as in the approved thesis plan (see `00_scope_decisions.md`), plus one
    sentence on how O3 is extended with biology-preservation metrics and why.
1.5 Contributions (the four from the scope doc, each tagged with its objective).
1.6 Thesis structure (one paragraph).

---

## 2 Background (12–15 p)
**Purpose:** give the reader exactly what they need for Ch. 3–5. No more.

2.1 **Multiplexed tissue imaging**
  - Cyclic immunofluorescence: stain → image → bleach cycles. `[lin2018cycif, lin2015cycif]`
  - From image to single-cell table: stitching, registration, segmentation, quantification.
    `[schapiro2022mcmicro]`
  - **Study structure: patient/sample → slide → scene → cell; batch; cycle.** Hierarchy table
    + figure. `[schapiro2022miti, hickey2022primer, harris2022slide]`, scene = scanner ROI
    (Zeiss manual, footnote).
2.2 **Sources of technical variation**
  - Fixation, antibody lot, staining run, imaging, cycle-dependent signal loss.
    `[chang2020restore, harris2022slide, intensity2023clinical]`
  - Data properties that break standard methods: zero-inflation, right skew, bimodal markers.
    `[wang2025uniform]`
2.3 **Normalization and batch correction methods**
  - Per-marker: Z-score, ComBat, MXnorm/functional registration, RESTORE, UniFORM.
    `[johnson2007combat, harris2022slide, harris2022mxnorm, chang2020restore, wang2025uniform]`
  - Cytometry: CytoNorm, cyCombine. `[vangassen2020cytonorm, pedersen2022cycombine]`
  - Embedding / integration (scRNA-seq): Harmony, scVI. `[korsunsky2019harmony, lopez2018scvi]`
  - Learned distribution matching: MMD, MMD-ResNet. `[gretton2012mmd, shaham2017mmdresnet]`
2.4 **Deep-learning paradigms compared (O2)** (3–4 p, one subsection each)
  - Graph-based: message passing, GATv2; contrastive NT-Xent; gradient reversal.
    `[brody2022gatv2, chen2020simclr, ganin2016dann]`
  - Flow-based: normalizing flows → continuous flows → (OT-)conditional flow matching.
    `[tong2024cfm, bunne2023cellot]`
  - Diffusion-based: DDPM, SDEdit partial-noising, classifier-free guidance.
    `[ho2020ddpm, meng2022sdedit]`
  - Common pieces: residual correction, MMD, Huber. `[gretton2012mmd, huber1964]`
2.5 **Evaluating batch correction**: the removal-vs-conservation trade-off; kBET; silhouette.
    `[buttner2019kbet, luecken2022scib, tran2020benchmark, rousseeuw1987silhouette]`

**Avoid:** a literature dump. Every method named here must reappear in Ch. 3 or 5.

---

## 3 Methods (12–15 p)
**Purpose:** describe the FINAL pipeline so someone could reimplement it. Present tense, no history.

3.1 Overview figure: raw → log1p → Stage 1 → Stage 2 → α-blend → output.
3.2 **Stage 1: analytic reference alignment**
  - KL-medoid reference sample per marker.
  - Bimodality detection (per-batch peak voting).
  - Unimodal: median shift. Bimodal: negative-peak shift. Both pure translations in log1p.
  - Cite as a port of `shift_normalize.py`; relate to UniFORM landmark registration `[wang2025uniform]`.
3.3 **Stage 2: per-cell residual corrector**
  - Encoder (GATv2 on per-(sample, scene) k-NN graph), residual decoder (zero-init), α blend.
  - Losses: Huber (anchor), MMD (batch alignment, bimodal markers masked), NT-Xent, adversarial CE
    with GRL ramp. Final weights table.
  - Sampler, training budget (10 epochs; state 50 tested), inference.
  - **Wording rule:** "per-cell corrector with a graph encoder". State that ablation (5.4) shows
    the graph contributes no measurable kBET.
3.4 **Alternative Stage 2 paradigms (O2)**: same Stage 1 input, same output space.
  - Flow: OT-CFM (FlowMLP, mini-batch Hungarian OT coupling, Euler integration, n_steps).
  - Diffusion: DDPM + SDEdit (DenoisingMLP, CFG, t_infer, cfg_scale).
  - Hyperparameter selection procedure for each (sweeps → appendix).
  - **Selection criterion for "most effective"**: primary kBET; eligibility constraints
    (silhouette, shape, positive population, stability). State this BEFORE the results.
3.5 Implementation and baselines: PyTorch, PyG, hardware, runtime, code availability; how UniFORM,
    ComBat, Z-score, MXnorm and MMD-ResNet were run (settings, versions).
3.6 **Design rationale** (≤ 1.5 p): the 4–5 one-sentence lessons from the OUT list in
    `00_scope_decisions.md`. This is where earlier attempts appear, framed as motivation.

---

## 4 Data and evaluation (8–10 p)
4.1 **PRAD-CyCIF dataset**: 1.76 M cells, 20 markers, 20 patients, 7 batches, 6 cycles.
    Source + what a batch physically is (**VERIFY** in UniFORM supplement). `[wang2025uniform]`
4.2 Preprocessing: log1p; `(sample, scene)` as the tissue unit. Explicitly note that `scene_id` is
    reused across samples.
4.3 **Evaluation axes** (one subsection each: definition, what it detects, what it is blind to)
  - Batch mixing: kBET on 5 clinical groups, UMAP rep. `[buttner2019kbet]`
  - Cluster preservation: per-sample silhouette, 3 cell types (ECAD/CD45/aSMA), and why
    per-sample rather than per-group (batch confound). `[rousseeuw1987silhouette, wang2025uniform]`
  - Marginal shape: peak / variance / IQR ratios vs raw.
  - Positive population: per-sample GMM threshold transferred. **State that it differs from
    UniFORM's definition.**
  - Supporting: per-marker batch adj-R², and why it is blind to local mixing.
4.4 Why multiple axes: a table of which metric detects which failure (blur, compression, shift…).
4.5 **Statistical protocol**: seeds 0/1/2, mean ± SD, paired differences, the pre-set decision
    rules (e.g. silhouette ≥ Stage 1 − 0.01). Mention the w_adv single-seed result that did not
    replicate as the motivating example.

---

## 5 Results (18–22 p): one section per objective
5.1 **O1 — Existing normalizers.** Table: Raw / Z-score / ComBat / MXnorm / UniFORM / Stage 1 ×
    (kBET, silhouette, shape, pos-pop). Histogram figure (Raw | UniFORM | Stage 1 | Stage 2).
5.2 **O2 — Paradigm comparison.** One table, all arms on Stage 1 input, 3 seeds each:
    graph (α=0.6) | no-graph | OT-CFM | DDPM | S1→MMD-ResNet (+ raw MMD-ResNet). Same columns as 5.1.
    Apply the pre-stated criterion → most effective eligible method. Then *why*: graph ablation
    summary (the graph itself adds no kBET; the per-cell residual + safeguards do).
5.3 **O3 — Proposed method vs existing techniques.** Batch metrics first (kBET per group, vs
    UniFORM), then biology axes. Matched-noise control: noise beats UniFORM on kBET, and only
    ≈ +0.018 of the Stage 2 lift is structured. This motivates the multi-axis reading.
5.4 **O4 — Detailed experimental analysis.** Safeguard ladder (masking = biology, Huber = stability,
    NT-Xent = no effect), α trade-off, seed variability, cases where metrics disagree
    (MMD-ResNet best adj-R² but worst biology; CFM good silhouette but reshaped marginals).

Every results table: mean ± SD, n seeds, and Stage 1 as the reference row.

---

## 6 Discussion (6–8 p)
6.1 Answers to O1–O4, one paragraph each.
6.2 Interpretation: per-cell residual (graph) vs distribution transport (flow) vs denoising
    (diffusion): why they distort different aspects of biology; why bimodal masking works.
6.3 Relation to prior work (UniFORM, MMD-ResNet, scIB trade-off). `[luecken2022scib]`
6.4 **Limitations**: one dataset; batch definition; pos-pop definition; 3 seeds; kBET sensitivity
    to perturbation; graph not effective; 10-epoch budget.
6.5 Future work: external dataset, truly spatial correction with (sample, scene) graphs, other
    platforms.

## 7 Conclusion (1–2 p)
No new numbers. Restate the answers to O1–O4 and the contributions.

---

## Appendices
A. Hyperparameters and full loss-weight table
B. Per-marker positive-population table (20 rows, mean Δ ± SD)
C. Per-group kBET tables
D. Graph ablation (old-graph / no-graph / graph-fix / neighbour-mean, 3 seeds)
E. OT-CFM and DDPM hyperparameter sweeps (n_steps; t_infer × cfg_scale)
F. Code and notebook index (which notebook produced which table)

---

## Figures plan (draft)
| # | Figure | Chapter |
|---|---|---|
| 1 | Study-structure hierarchy (sample/slide/scene/batch/cycle/cell) | 2.1 |
| 2 | Raw histograms showing batch effect on 2–3 markers | 2.2 |
| 3 | Pipeline diagram (Stage 1 + Stage 2) | 3.1 |
| 4 | Metric-failure-mode schematic (what each metric sees) | 4.4 |
| 5 | Per-sample histogram grid, Raw / UniFORM / Stage 1 / Stage 2 (unimodal markers) | 5.1 |
| 6 | kBET vs silhouette scatter: benchmarks, all paradigms, controls, α points (the key figure) | 5.2–5.4 |
| 7 | UMAP raw vs normalized, coloured by batch | 5.3 |
| 8 | Ladder bar chart (Δ silhouette, Δ kBET per removed safeguard) | 5.4 |
| 9 | Paradigm comparison histograms: graph vs OT-CFM vs DDPM on 3 unimodal markers | 5.2 |

Figure 6 can carry the whole thesis: one kBET–silhouette plane showing that noise moves right but
down, MMD-ResNet moves right and far down, and Stage 2 moves right while staying near Stage 1.
