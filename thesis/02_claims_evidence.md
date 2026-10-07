# Claims → evidence map

Rule: a number appears in the thesis only if it has a row here. Status:
✅ = seed-repeated / deterministic, usable · ⚠️ = single run or caveat, label it · ❌ = do not use.
Numbers copied from `CLAUDE.md` and project notes (2026-10-07 state). Re-check against notebook
outputs before final.

> Section headers use the earlier RQ numbering. Mapping to thesis-plan objectives:
> RQ1 → O1, RQ2 → O2/O3, RQ3 → O4, RQ4 → O3. The paradigm comparison (O2) is below.

## O2 — paradigm comparison (graph / flow / diffusion)

| Claim | Number | Source | Status |
|---|---|---|---|
| Graph-based Stage 2 | kBET 0.709 ± 0.006, sil 0.360 ± 0.002, 0 distorted | see RQ2 | ✅ |
| Flow (OT-CFM) higher kBET but reshapes marginals | kBET 0.758, sil 0.350, 16/20 shape-distorted | `spancy_shift_cfm_explore.ipynb` | ⚠️ single run → **rerun 3 seeds** |
| Diffusion (DDPM + SDEdit) | kBET 0.735 | `spancy_shift_ddpm_explore.ipynb` / `ddpm_eval` | ⚠️ single run; pos-pop invalid (global threshold); silhouette + shape **pending** |
| Coupling flow (SpaNCy-Flow) failed | no final metrics | `spancy_flow.py` history | ✅ qualitative only (design rationale) |
| Most effective eligible method = graph-based | depends on the criterion in 3.4 | — | ❌ until CFM/DDPM are seed-repeated |
| CFM/DDPM lift vs perturbation size | — | — | not run; optional matched-noise control |

## RQ1 — Stage 1 vs existing normalizers

| Claim | Number | Source | Status |
|---|---|---|---|
| Stage 1 kBET ≈ UniFORM | Stage 1 0.620 vs UniFORM 0.6315 | `spancy_shift_explore.ipynb` §8; `mxnorm_benchmark.ipynb` | ✅ Stage 1 is deterministic. Note it is slightly *below* UniFORM since the neg-peak change |
| Stage 1 is silhouette-neutral | 0.3670 vs raw 0.3669 (19 samples) | shift notebook §8b | ✅ |
| UniFORM silhouette ≈ raw | 0.3636 vs 0.3639 | `mxnorm_benchmark.ipynb` | ⚠️ different sample guard than shift-repo run; compare within file only, or recompute both with the same guard |
| ComBat / Z-score / MXnorm poor kBET | 0.286 / 0.293 / 0.244 | benchmark | ✅ (deterministic methods) |
| MXnorm scrambles clusters | silhouette −0.028 | benchmark | ✅ |
| Shape: UniFORM 0/20 distorted; ComBat 15, MXnorm 15, Z-score 19 | — | benchmark cell 5g | ✅ flag uses var only (MXnorm over-counted) |
| ECAD neg-peak vs sigmoid | var ratio 0.997 vs 0.856 | shift notebook | ✅ design rationale only |

## RQ2 — Stage 2 improves mixing

| Claim | Number | Source | Status |
|---|---|---|---|
| Stage 2 α=0.6 kBET | **0.709 ± 0.006** (n=6) | ablation §9 + `spancy_shift_mmdresnet_s1_wadv03.ipynb` | ✅ replace article's single-run 0.708 |
| Stage 2 α=0.6 silhouette | **0.360 ± 0.002** (n=3) | `..._wadv03.ipynb` | ✅ replace the old 0.365 |
| Stage 2 shape clean | 0 distorted, var ≈ 0.95 | ladder | ✅ |
| α=1 more kBET, less biology | kBET 0.750 (seed 0); sil −0.017 every seed | ladder | ⚠️ kBET single seed |
| Stage 1 enables MMD-ResNet | raw 0.555 ± 0.025 → on S1 0.718 ± 0.024 | `spancy_shift_mmdresnet_s1*.ipynb` | ✅ |
| Our Stage 2 ties S1→MMD-ResNet on kBET, beats on silhouette | sil 0.3595 vs 0.3288, every seed | `..._wadv03.ipynb` | ✅ |
| Stage 2 ≫ UniFORM on kBET | +0.077 | — | ✅ but read with the noise-control rows below |

## RQ2 — how much is learned (noise control)

| Claim | Number | Source | Status |
|---|---|---|---|
| Matched noise alone lifts kBET | S1 + noise 0.692 ± 0.008 | `spancy_shift_noisectrl_seeds.ipynb` | ✅ |
| Stage 2 beats noise every seed | +0.019 kBET, +0.010 silhouette (paired) | same | ✅ |
| ~80% of the +0.09 lift is perturbation size | (0.692−0.620)/(0.710−0.620) | same | ✅ state as an estimate |
| Structured gain concentrated in g5/g2; on g3 the controls score higher | per-group | same | ⚠️ per-group swings ±0.08, appendix only |

## RQ3 — safeguards and architecture choices

| Claim | Number | Source | Status |
|---|---|---|---|
| Removing bimodal masking costs biology | sil −0.063; ECAD moved 0.39 log1p, var 0.84 | `spancy_shift_ladder.ipynb` | ✅ 3 seeds |
| Masking is free on kBET | +0.010 (seed 0) | same | ⚠️ single seed, within noise |
| Without Huber, training diverges | 3/3 seeds | same | ✅ |
| NT-Xent has no measurable effect | sign-inconsistent | same | ✅ (old "CD20 compression" claim ❌) |
| Bimodal markers' Stage 2 output = Stage 1 exactly | bim_delta 0.000 | same | ✅ explain mechanism (zero-init + masked MMD + zero Huber gradient at 0) |
| Graph adds no kBET | no-graph 0.721 ± 0.014 vs 0.706 ± 0.006 | `..._wadv03.ipynb` | ✅ |
| Graph kept for silhouette | no-graph −0.004, every seed | same | ✅ small effect; say so |
| Neighbour-mean input does not help | kBET +0.012 fully explained by larger correction; structured gain −0.003 ± 0.015 | `spancy_shift_nbr_confirm.ipynb` | ✅ one sentence, appendix table |
| Adversarial CE: single-seed gain did not replicate | +0.048 / −0.015 / +0.002 | ablation §9 | ✅ methods example |

## RQ4 — evaluation

| Claim | Number | Source | Status |
|---|---|---|---|
| kBET alone is gameable | noise 0.692 > UniFORM 0.6315 | noise ctrl | ✅ |
| adj-R² blind to local mixing | MMD-ResNet best adj-R² 0.0037, worst kBET/biology | graphfix A/B | ✅ |
| CFM vs Stage 2 distort opposite things | CFM 16/20 shape-distorted, sil 0.350 | CFM notebook | ⚠️ single run |
| Old per-group silhouette was batch-confounded | g2 0.327 → 0.159 | — | ✅ methods justification |

## Do NOT use

| Number / claim | Why |
|---|---|
| "spatial GNN", "spatial context improves mixing" | no-graph ≈ graph; graph mixed patients (`scene_id` reuse); neighbour-mean also rejected |
| "first method to clear the dual target" | target was revised after results |
| 0.717, 0.712, 0.665, 0.6698 kBET for Stage 2 | superseded runs / older Stage 1 |
| 0.743 / "drop CE" | did not replicate |
| CFM/DDPM per-marker violations (CD45 −23%, PD1 −30%, ChromA −34%) | global-threshold artifacts |
| Positive-pop numbers compared with UniFORM paper values | different metric definition (our best match: MAE 16.7 pp) |
| "No prior OT/diffusion work on CyCIF" without qualification | CellOT evaluated multiplexed imaging |
| 50-epoch results as final | rejected; 10 epochs is final |
