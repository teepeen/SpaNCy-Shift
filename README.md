# SpaNCy-Shift

Two-stage batch normalization for multiplexed tissue imaging (CyCIF), implemented in a single module,
`spancy_shift_spatial.py`.

**Stage 1 (analytic).** For every marker, each sample is shifted in log1p space toward a reference
sample, the KL medoid of the per-sample histograms. Unimodal markers are shifted by the difference of
medians; bimodal markers (two or more histogram peaks in at least half of the batches) by the
difference of their leftmost (negative) peaks. Output: `layers['normalized_base']`.

**Stage 2 (graph neural network).** Each cell is connected to its 15 nearest cells within the same
scanned region of the same sample. A two-layer GATv2 encoder and a zero-initialized residual decoder
compute a correction for every cell from its own markers and its two-hop neighborhood. Training
(10 epochs, AdamW) combines a Huber penalty on the correction (weight 0.1), an NT-Xent contrastive loss
with a spatial neighbor as positive (0.5), an adversarial batch classifier behind gradient reversal
(0.3) and a multi-scale RBF MMD between batches (1.0), evaluated on unimodal markers only, so bimodal
markers keep their Stage 1 values. The output is Stage 1 plus `alpha` times the correction
(default `alpha = 0.6`): `layers['normalized']`.

## Installation

```bash
pip install "git+https://github.com/teepeen/SpaNCy-Shift.git"
```

or, from a clone of this repository, `pip install -r requirements.txt` and import
`spancy_shift_spatial.py` directly. `torch-geometric` must match your PyTorch and CUDA version; see the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
A GPU is recommended.

## Input

An AnnData `.h5ad` file with raw (non-log) intensities in `X` and, in `obs`:

| Column | Accepted names |
|---|---|
| batch | `batch_id`, `batch`, `Batch`, `BatchID` |
| sample | `sample_id`, `sample`, `patient_id` |
| scanned region (optional) | `scene_id` — regions are `sample\|scene`; without it, each sample is one region |
| coordinates | `x`/`y`, `X`/`Y` or `centroid_x`/`centroid_y` |

If `var_names` are numeric, marker names are taken from `var['marker_name']` (or `marker`, `gene`,
`protein`).

## Usage

Command line:

```bash
python spancy_shift_spatial.py --input data.h5ad --output data_normalized.h5ad --device cuda
```

Options: `--epochs` (10), `--lr` (1e-3), `--seed` (0), `--n_per_batch` (512), `--k_neighbors` (15),
`--alpha` (0.6), `--w_recon` (0.1), `--w_contrast` (0.5), `--w_adv` (0.3), `--w_mmd` (1.0),
`--mmd_samples` (256), `--grl_max` (1.0), `--bimodal_min_batch_frac` (0.5), `--layer_name` (`normalized`).

Python:

```python
import spancy_shift_spatial as ss

adata = ss.load_adata('data.h5ad')
prep = ss.prepare(adata)                        # Stage 1 + scaling + region graph
model, history = ss.train(prep, device_str='cuda')
delta = ss.correct(prep, model, device_str='cuda')
X_out = ss.apply(prep, delta, alpha=0.6)        # any alpha without re-running the network
# or in one call:
adata_out = ss.normalize_adata(prep, model, alpha=0.6, device_str='cuda')
```

The correction is computed once (`correct`), so different values of `alpha` can be compared with
`apply` alone; `alpha = 0` returns Stage 1.

## Example notebook

`spancy_shift_demo.ipynb` runs the whole pipeline on PRAD-CyCIF; in Google Colab just run all cells
(`spancy_shift_spatial.py` is downloaded from this repository next to the notebook, nothing to
upload): download,
Stage 1, training, inference at `alpha = 0.6`, per-sample histograms of all markers
(Raw | Stage 1 | SpaNCy-Shift) and writing the normalized `.h5ad`
(`X` = raw, `layers['normalized_base']` = Stage 1, `layers['normalized']` = SpaNCy-Shift).

## Data

The PRAD-CyCIF dataset (1.76 M cells, 20 markers, 20 samples, 7 batches) is available on Zenodo:
<https://zenodo.org/records/16383766> (`PRAD_anndata.h5ad`). On a Colab GPU, preparation takes about
one minute and training about 15 minutes.

## Reproducibility

GATv2 message passing on the GPU is not bit-reproducible, so repeated runs with the same seed differ
slightly. Report results as mean ± SD over several training seeds.

## Citation

If you use this code, please cite the accompanying article (reference to be added on publication).
