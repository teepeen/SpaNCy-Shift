#!/usr/bin/env python
"""
SpaNCy-Shift spatial: two-stage batch normalization for multiplexed tissue imaging.

Stage 1 (analytic): per-marker shift of every sample toward a KL-medoid reference sample in log1p
  space (median shift for unimodal markers, negative-peak shift for bimodal markers).
Stage 2 (graph neural network): a 2-layer GATv2 encoder + residual decoder corrects each cell from its
  own markers and its k spatial neighbours (same sample, same scanned region, 2-hop receptive field).
  Losses: MMD between batches (bimodal markers masked), Huber on the correction, NT-Xent (one spatial
  neighbour as positive), adversarial batch classifier with gradient reversal.
  Output = Stage 1 + alpha * correction (alpha = 0.6).

Training draws, per batch and step, random seed cells from the whole batch plus their k neighbours.

Usage:
    adata = load_adata('data.h5ad')
    prep = prepare(adata)                 # Stage 1 + region graph
    model, history = train(prep, device_str='cuda')
    delta = correct(prep, model, device_str='cuda')
    X_out = apply(prep, delta, alpha=0.6)
    # or, in one call: adata_out = normalize_adata(prep, model, device_str='cuda')

CLI:
    python spancy_shift_spatial.py --input data.h5ad --output data_normalized.h5ad --device cuda
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as _sp
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks as _scipy_find_peaks
from scipy.stats import entropy as _kl_entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.nn import GATv2Conv

log = logging.getLogger("spancy_shift_spatial")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)
    log.propagate = False


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def load_adata(path: str) -> ad.AnnData:
    """Load an AnnData file, map the batch column to obs['batch_id'] and, if var_names are numeric, set
    them from a marker-name column in var.

    Args:
        path: path to the .h5ad file.
    """
    log.info("Loading %s", path)
    adata = ad.read_h5ad(path)
    log.info("Loaded %d cells x %d markers", adata.n_obs, adata.n_vars)

    for col in ("batch_id", "batch", "Batch", "BatchID"):
        if col in adata.obs.columns:
            if col != "batch_id":
                adata.obs["batch_id"] = adata.obs[col]
                log.info("Mapped obs['%s'] → obs['batch_id']", col)
            break
    else:
        raise ValueError("No batch column found (tried: batch_id, batch, Batch, BatchID)")

    if all(str(v).isdigit() for v in adata.var_names):
        for col in ("marker_name", "marker", "gene", "protein"):
            if col in adata.var.columns:
                adata.var_names = adata.var[col].values
                log.info("Set var_names from var['%s']", col)
                break

    return adata


def _get_col(adata: ad.AnnData, candidates: List[str], fallback: Optional[str]) -> Optional[str]:
    """Return the first of `candidates` that is an obs column, otherwise `fallback`.

    Args:
        adata: AnnData whose obs columns are searched.
        candidates: column names in order of preference.
        fallback: name returned when no candidate exists.
    """
    for c in candidates:
        if c in adata.obs.columns:
            return c
    return fallback


def log1p_scale(X: np.ndarray) -> Tuple[np.ndarray, RobustScaler]:
    """log1p followed by a per-marker RobustScaler (median/IQR); returns the scaled data and the scaler.

    Args:
        X: (cells, markers) count-scale expression.
    """
    X_log = np.log1p(np.clip(X, 0, None))
    scaler = RobustScaler()
    return scaler.fit_transform(X_log).astype(np.float32), scaler


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: analytic shift normalization
# ──────────────────────────────────────────────────────────────────────────────

def _find_peaks_1d(values: np.ndarray, min_prominence_frac: float) -> np.ndarray:
    """Positions of the peaks of a smoothed 150-bin histogram (the highest bin if no peak is found).

    Args:
        values: 1D values.
        min_prominence_frac: minimum peak prominence as a fraction of the smoothed histogram maximum.
    """
    counts, edges = np.histogram(values, bins=150)
    centers = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
    prom = min_prominence_frac * float(smoothed.max())
    peak_idx, _ = _scipy_find_peaks(smoothed, prominence=prom)
    if len(peak_idx) == 0:
        return np.array([centers[int(np.argmax(smoothed))]])
    return centers[peak_idx]


def detect_bimodal_markers(
    X_log: np.ndarray,
    marker_names: List[str],
    batch_codes: np.ndarray,
    min_prominence_frac: float,
    bimodal_min_batch_frac: float,
) -> np.ndarray:
    """Boolean mask of bimodal markers: a marker is bimodal if at least `bimodal_min_batch_frac` of the
    batches (with >= 100 cells) show two or more histogram peaks.

    Args:
        X_log: (cells, markers) log1p expression.
        marker_names: marker names, for logging.
        batch_codes: integer batch label per cell.
        min_prominence_frac: minimum peak prominence as a fraction of the histogram maximum.
        bimodal_min_batch_frac: fraction of batches that must show two or more peaks.
    """
    n_markers = X_log.shape[1]
    is_bimodal = np.zeros(n_markers, dtype=bool)
    unique_batches = np.unique(batch_codes)

    log.info("Bimodal detection (%d markers, %d batches):", n_markers, len(unique_batches))
    for m, mname in enumerate(marker_names):
        x = X_log[:, m]
        n_valid = n_bimodal = 0
        for b in unique_batches:
            x_b = x[batch_codes == b]
            if len(x_b) < 100:
                continue
            n_valid += 1
            if len(_find_peaks_1d(x_b, min_prominence_frac)) >= 2:
                n_bimodal += 1
        if n_valid > 0 and (n_bimodal / n_valid) >= bimodal_min_batch_frac:
            is_bimodal[m] = True
            log.info("  %-14s  BIMODAL  (%d/%d batches)", mname, n_bimodal, n_valid)
        else:
            log.info("  %-14s  unimodal (%d/%d batches bimodal)", mname, n_bimodal, n_valid)

    return is_bimodal


def find_best_sample_per_marker(adata: ad.AnnData) -> Dict[str, Optional[str]]:
    """Reference sample per marker: the sample with the lowest mean pairwise symmetric KL divergence to
    all other samples (50-bin log1p histograms, samples with >= 100 cells; None if no sample qualifies).

    Args:
        adata: AnnData with raw expression in X.
    """
    X = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X, dtype=np.float32)
    sample_col = _get_col(adata, ["sample_id", "sample", "patient_id"], "batch_id")
    samples = adata.obs[sample_col].values
    unique_samples = np.unique(samples)
    marker_names = list(adata.var_names)
    result: Dict[str, Optional[str]] = {}

    log.info("Reference selection: %d markers × %d samples (KL medoid)",
             len(marker_names), len(unique_samples))

    for i, mname in enumerate(marker_names):
        X_log = np.log1p(X[:, i])
        bins = np.histogram_bin_edges(X_log, bins=50)
        hists: Dict[str, np.ndarray] = {}
        for s in unique_samples:
            v = X_log[samples == s]
            if len(v) < 100:
                continue
            p, _ = np.histogram(v, bins=bins, density=True)
            p = (p + 1e-8) / (p + 1e-8).sum()
            hists[s] = p
        valid = list(hists.keys())
        if not valid:
            result[mname] = None
            continue
        mean_kl = {
            s: sum(_kl_entropy(hists[s], hists[t]) + _kl_entropy(hists[t], hists[s])
                   for t in valid if t != s) / max(len(valid) - 1, 1)
            for s in valid
        }
        best = min(mean_kl, key=mean_kl.get)
        result[mname] = best
        if i % 5 == 0:
            log.info("  %d/%d  %-14s → %s  (mean_kl=%.3f)",
                     i + 1, len(marker_names), mname, best, mean_kl[best])

    return result


def _shifts_unimodal(
    X_log: np.ndarray, sample_labels: np.ndarray, target_sample: str,
) -> Dict[str, float]:
    """Per-sample shift that moves the sample median onto the reference median (0 for < 10 cells).

    Args:
        X_log: log1p values of one marker.
        sample_labels: sample label per cell.
        target_sample: reference sample.
    """
    ref_med = float(np.median(X_log[sample_labels == target_sample]))
    return {
        s: ref_med - float(np.median(X_log[sample_labels == s]))
        if (sample_labels == s).sum() >= 10 else 0.0
        for s in np.unique(sample_labels)
    }


def _shifts_negative_peak(
    X_log: np.ndarray, sample_labels: np.ndarray, target_sample: str, min_prominence_frac: float,
) -> Dict[str, float]:
    """Per-sample shift that moves the sample's leftmost histogram peak onto the reference's leftmost
    peak (0 for < 10 cells).

    Args:
        X_log: log1p values of one marker.
        sample_labels: sample label per cell.
        target_sample: reference sample.
        min_prominence_frac: minimum peak prominence as a fraction of the histogram maximum.
    """
    ref_neg = float(np.min(_find_peaks_1d(X_log[sample_labels == target_sample], min_prominence_frac)))
    shifts = {}
    for s in np.unique(sample_labels):
        v = X_log[sample_labels == s]
        if len(v) < 10:
            shifts[s] = 0.0
            continue
        shifts[s] = ref_neg - float(np.min(_find_peaks_1d(v, min_prominence_frac)))
    return shifts


def _apply_shifts(
    X_log: np.ndarray, sample_labels: np.ndarray, shifts: Dict[str, float],
) -> np.ndarray:
    """Add each sample's shift to its cells (used for unimodal and bimodal markers).

    Args:
        X_log: log1p values of one marker.
        sample_labels: sample label per cell.
        shifts: shift per sample.
    """
    out = X_log.copy()
    for s, shift in shifts.items():
        mask = sample_labels == s
        out[mask] = X_log[mask] + shift
    return out


def shift_normalize_per_marker(
    adata: ad.AnnData,
    marker_to_best_sample: Dict[str, Optional[str]],
    min_prominence_frac: float,
    bimodal_min_batch_frac: float,
) -> Tuple[ad.AnnData, np.ndarray]:
    """Stage 1: shift every sample toward its per-marker reference sample in log1p space (median shift
    for unimodal markers, leftmost-peak shift for bimodal markers). Returns a copy of adata with the
    count-scale result in layers['normalized_base'] and the bimodal-marker mask.

    Args:
        adata: AnnData with raw expression in X.
        marker_to_best_sample: reference sample per marker (markers mapped to None are left unchanged).
        min_prominence_frac: minimum peak prominence as a fraction of the histogram maximum.
        bimodal_min_batch_frac: fraction of batches that must show two or more peaks for a bimodal marker.
    """
    X = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X, dtype=np.float64)
    sample_col = _get_col(adata, ["sample_id", "sample", "patient_id"], "batch_id")
    samples = adata.obs[sample_col].values
    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = pd.Categorical(adata.obs[batch_col].values).codes
    marker_names = list(adata.var_names)

    X_log_all = np.log1p(X)
    is_bimodal = detect_bimodal_markers(
        X_log_all, marker_names, batch_codes,
        min_prominence_frac=min_prominence_frac,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
    )

    normalized = X.copy()
    for k, mname in enumerate(marker_names):
        target = marker_to_best_sample.get(mname)
        if target is None:
            log.info("[%d/%d] %s — skipped (no reference)", k + 1, len(marker_names), mname)
            continue
        X_log = X_log_all[:, k]
        if is_bimodal[k]:
            log.info("[%d/%d] %s  BIMODAL  ref=%s", k + 1, len(marker_names), mname, target)
            shifts = _shifts_negative_peak(X_log, samples, target, min_prominence_frac)
        else:
            log.info("[%d/%d] %s  unimodal  ref=%s", k + 1, len(marker_names), mname, target)
            shifts = _shifts_unimodal(X_log, samples, target)
        X_log_out = _apply_shifts(X_log, samples, shifts)
        normalized[:, k] = np.clip(np.expm1(X_log_out), 0, None)

    adata_out = adata.copy()
    adata_out.layers["normalized_base"] = normalized.astype(np.float32)
    log.info("Stage 1 done. Layer 'normalized_base' written (%d cells × %d markers).",
             adata.n_obs, adata.n_vars)
    return adata_out, is_bimodal


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: model
# ──────────────────────────────────────────────────────────────────────────────

class _GradRevFn(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by -lam in the backward pass."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GradientReversal(nn.Module):
    """Gradient reversal layer; its strength `lam` is set by the training loop."""

    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def forward(self, x):
        return _GradRevFn.apply(x, self.lam)


class SpatialGNNEncoder(nn.Module):
    """Linear projection + GELU, then two GATv2 layers (residual + LayerNorm + GELU after the first,
    LayerNorm after the second). `hidden` must be divisible by `n_heads`.

    Args:
        n_markers: number of input markers.
        hidden: hidden width.
        latent: latent width.
        n_heads: attention heads of the first GATv2 layer.
    """

    def __init__(self, n_markers: int, hidden: int, latent: int, n_heads: int):
        super().__init__()
        self.proj = nn.Linear(n_markers, hidden)
        self.conv1 = GATv2Conv(hidden, hidden // n_heads, heads=n_heads, concat=True,
                               dropout=0.1, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden, latent, heads=1, concat=False,
                               dropout=0.1, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(latent)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode cells into the latent space.

        Args:
            x: (cells, markers) scaled expression.
            edge_index: (2, E) edges neighbour -> cell, local indices.
        """
        h = self.act(self.proj(x))
        h = self.act(self.norm1(self.conv1(h, edge_index) + h))
        return self.norm2(self.conv2(h, edge_index))


class _ResidualDecoder(nn.Module):
    """MLP from the latent space to a per-marker correction; the last layer starts at zero.

    Args:
        latent: latent width.
        hidden: hidden width.
        n_markers: number of output markers.
    """

    def __init__(self, latent: int, hidden: int, n_markers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(),
            nn.Linear(hidden, n_markers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Correction for each cell.

        Args:
            z: (cells, latent) latent codes.
        """
        return self.net(z)


class _ProjectionHead(nn.Module):
    """MLP from the latent space to a 32-d L2-normalized embedding for the contrastive loss.

    Args:
        latent: latent width.
    """

    def __init__(self, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, latent), nn.GELU(),
            nn.Linear(latent, 32),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """L2-normalized embedding for each cell.

        Args:
            z: (cells, latent) latent codes.
        """
        return nn.functional.normalize(self.net(z), dim=1)


class _BatchDiscriminator(nn.Module):
    """Batch classifier on the latent space, behind a gradient reversal layer.

    Args:
        latent: latent width.
        n_batches: number of batches.
    """

    def __init__(self, latent: int, n_batches: int):
        super().__init__()
        self.grl = GradientReversal()
        self.net = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, n_batches),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Batch logits for each cell.

        Args:
            z: (cells, latent) latent codes.
        """
        return self.net(self.grl(z))


class GNNStage2(nn.Module):
    """Stage 2 model: GATv2 encoder -> residual decoder, projection head and batch discriminator.

    Args:
        n_markers: number of markers.
        n_batches: number of batches.
        hidden: hidden width.
        latent: latent width.
        n_heads: attention heads of the first GATv2 layer.
    """

    def __init__(self, n_markers: int, n_batches: int, hidden: int, latent: int, n_heads: int):
        super().__init__()
        self.encoder = SpatialGNNEncoder(n_markers, hidden, latent, n_heads)
        self.decoder = _ResidualDecoder(latent, hidden, n_markers)
        self.proj_head = _ProjectionHead(latent)
        self.discriminator = _BatchDiscriminator(latent, n_batches)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Correction, contrastive embedding and batch logits for each cell.

        Args:
            x: (cells, markers) scaled expression.
            edge_index: (2, E) edges neighbour -> cell, local indices.
        """
        z = self.encoder(x, edge_index)
        delta = self.decoder(z)
        z_proj = self.proj_head(z)
        batch_logits = self.discriminator(z)
        return delta, z_proj, batch_logits


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: spatial graph (one k-NN per tissue region) and sampler
# ──────────────────────────────────────────────────────────────────────────────

def region_labels(adata: ad.AnnData) -> np.ndarray:
    """One label per tissue region: sample|scene_id (scene ids restart for every sample). Without a
    scene column the sample is the region; without either, the batch.

    Args:
        adata: AnnData with sample and scene columns in obs.
    """
    sample_col = _get_col(adata, ["sample_id", "sample", "patient_id"], None)
    scene_col = _get_col(adata, ["scene_id"], None)
    if scene_col is None:
        return adata.obs[sample_col or "batch_id"].astype(str).values
    scenes = adata.obs[scene_col].astype(str).values
    if sample_col is not None:
        scenes = adata.obs[sample_col].astype(str).values + "|" + scenes
    return scenes


def build_region_graph(adata: ad.AnnData, regions: np.ndarray, k: int) -> np.ndarray:
    """(cells, k) int32 matrix of each cell's k nearest cells (x, y) within its own region.

    Args:
        adata: AnnData with x, y coordinates in obs.
        regions: region label per cell.
        k: number of neighbours.
    """
    xy_x = _get_col(adata, ["x", "X", "centroid_x"], None)
    xy_y = _get_col(adata, ["y", "Y", "centroid_y"], None)
    if xy_x is None or xy_y is None:
        raise ValueError("Spatial coordinates (x, y) not found in adata.obs")
    unique_regions = np.unique(regions)
    n_cells = adata.n_obs
    # Rows never filled (regions with < 2 cells) point at the cell itself, never at another region.
    knn = np.repeat(np.arange(n_cells, dtype=np.int32)[:, None], k, axis=1)
    log.info("Building spatial k-NN (k=%d) for %d regions...", k, len(unique_regions))
    for rg in unique_regions:
        mask = regions == rg
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        xy = adata.obs.loc[mask, [xy_x, xy_y]].values.astype(np.float32)
        k_eff = min(k + 1, len(idx))
        _, nbr = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree").fit(xy).kneighbors(xy)
        global_nbrs = idx[nbr[:, 1:k + 1]]              # skip self; pad if fewer than k
        if global_nbrs.shape[1] < k:
            pad = np.tile(global_nbrs[:, :1], (1, k - global_nbrs.shape[1]))
            global_nbrs = np.concatenate([global_nbrs, pad], axis=1)
        knn[idx] = global_nbrs.astype(np.int32)
    return knn


class AdjacencyIndex:
    """k-NN matrix with halo and subgraph extraction; two cell-sized buffers are reused across calls.

    Args:
        knn: (cells, k) neighbour matrix.
    """

    def __init__(self, knn: np.ndarray):
        self.knn = knn
        self.N, self.k = knn.shape
        self._mask = np.zeros(self.N, dtype=bool)
        self._loc = np.full(self.N, -1, dtype=np.int64)

    def expand(self, idx: np.ndarray) -> np.ndarray:
        """idx followed by the new cells of its 2-hop halo (targets keep rows 0..len(idx)-1).

        Args:
            idx: target cells (global indices).
        """
        mask = self._mask
        mask[idx] = True
        parts = [np.asarray(idx)]
        frontier = np.asarray(idx)
        for _ in range(2):
            nbrs = np.unique(self.knn[frontier].ravel())
            new = nbrs[~mask[nbrs]]
            if len(new) == 0:
                break
            mask[new] = True
            parts.append(new)
            frontier = new
        ctx = np.concatenate(parts)
        mask[ctx] = False
        return ctx

    def get_subgraph(self, idx: np.ndarray) -> np.ndarray:
        """(2, E) edge_index among idx in local coordinates, edges neighbour -> cell.

        Args:
            idx: cells of the subgraph (global indices).
        """
        B, k = len(idx), self.k
        flat = self.knn[idx].ravel()
        mask = self._mask
        mask[idx] = True
        keep = mask[flat] & (flat != np.repeat(idx, k))
        mask[idx] = False
        if not keep.any():
            return np.zeros((2, 0), dtype=np.int64)
        cell_local = np.repeat(np.arange(B), k)[keep]
        nbr_local = self.local_positions(idx, flat[keep])
        return np.stack([nbr_local, cell_local])

    def local_positions(self, ctx: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Row of each of `cells` within ctx (-1 if absent).

        Args:
            ctx: cells in local order (global indices).
            cells: cells to locate (global indices).
        """
        loc = self._loc
        loc[ctx] = np.arange(len(ctx))
        out = loc[cells]
        loc[ctx] = -1
        return out


class NeighbourPatchSampler:
    """Per batch and step: n_per_batch // (k+1) seed cells drawn uniformly from all cells of the batch,
    each with its k spatial neighbours.

    Args:
        batch_codes: integer batch label per cell.
        knn: (cells, k) neighbour matrix.
        n_per_batch: cells per batch per step.
    """

    def __init__(self, batch_codes: np.ndarray, knn: np.ndarray, n_per_batch: int):
        self.rng = np.random.RandomState(42)
        self.knn = knn
        self.n_seeds = max(1, n_per_batch // (knn.shape[1] + 1))
        self.unique_batches = np.unique(batch_codes)
        self.batch_indices = {int(b): np.where(batch_codes == b)[0] for b in self.unique_batches}
        self._n_steps = max(1, len(batch_codes) // (n_per_batch * len(self.unique_batches)))

    def sample(self) -> np.ndarray:
        """Cell indices of one training step (duplicates from overlapping patches removed)."""
        parts = []
        for b in self.unique_batches:
            idx = self.batch_indices[int(b)]
            seeds = self.rng.choice(idx, size=min(self.n_seeds, len(idx)), replace=False)
            parts.append(seeds)
            parts.append(self.knn[seeds].ravel())
        cat = np.concatenate(parts)
        _, first = np.unique(cat, return_index=True)
        return cat[np.sort(first)]

    def __len__(self) -> int:
        """Number of steps per epoch."""
        return self._n_steps


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: losses
# ──────────────────────────────────────────────────────────────────────────────

def mmd_rbf_loss(
    X_out: torch.Tensor,
    batch_ids: torch.Tensor,
    unimodal_mask: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Multi-scale RBF MMD² (bandwidths 0.5, 1, 5) averaged over all batch pairs, on unimodal markers.

    Args:
        X_out: (cells, markers) corrected scaled expression.
        batch_ids: batch label per cell.
        unimodal_mask: boolean mask of the markers used.
        n_samples: maximum cells per batch in the estimate.
    """
    bandwidths = (0.5, 1.0, 5.0)
    X = X_out[:, unimodal_mask]
    unique_batches = batch_ids.unique()
    if len(unique_batches) < 2:
        return torch.tensor(0.0, device=X.device)

    def rbf_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # (n, m, d)
        sq = (diff ** 2).sum(-1)                # (n, m)
        return sum(torch.exp(-sq / (2 * bw ** 2)) for bw in bandwidths)

    loss = torch.tensor(0.0, device=X.device)
    n_pairs = 0
    batches = list(unique_batches)
    for i in range(len(batches)):
        xi = X[batch_ids == batches[i]]
        if len(xi) > n_samples:
            xi = xi[torch.randperm(len(xi), device=X.device)[:n_samples]]
        for j in range(i + 1, len(batches)):
            xj = X[batch_ids == batches[j]]
            if len(xj) > n_samples:
                xj = xj[torch.randperm(len(xj), device=X.device)[:n_samples]]
            if len(xi) < 4 or len(xj) < 4:
                continue
            k_xx = rbf_kernel(xi, xi).mean()
            k_yy = rbf_kernel(xj, xj).mean()
            k_xy = rbf_kernel(xi, xj).mean()
            loss = loss + (k_xx + k_yy - 2 * k_xy).clamp(min=0)
            n_pairs += 1

    return loss / max(n_pairs, 1)


def nt_xent_pairs(za: torch.Tensor, zb: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR NT-Xent for positive pairs (za[i], zb[i]); all other rows are negatives.

    Args:
        za: (B, d) L2-normalized embeddings.
        zb: (B, d) L2-normalized embeddings of the positives.
        temperature: softmax temperature.
    """
    B = za.size(0)
    z = torch.cat([za, zb], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return nn.functional.cross_entropy(sim, target)


# ──────────────────────────────────────────────────────────────────────────────
# Preparation: Stage 1 + scaling + graph, computed once
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Prepared:
    """Everything train() and correct() need, computed once by prepare().

    Args:
        adata_base: copy of the input with the Stage 1 output in layers['normalized_base'].
        X_scaled: log1p(Stage 1) after the RobustScaler, float32.
        scaler: fitted RobustScaler.
        is_bimodal: boolean mask of bimodal markers.
        batch_codes: integer batch label per cell.
        regions: region label per cell.
        adj: region k-NN graph.
    """
    adata_base: ad.AnnData
    X_scaled: np.ndarray
    scaler: RobustScaler
    is_bimodal: np.ndarray
    batch_codes: np.ndarray
    regions: np.ndarray
    adj: AdjacencyIndex


def prepare(
    adata: ad.AnnData,
    k_neighbors: int = 15,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
) -> Prepared:
    """Run Stage 1, scale its output and build the region k-NN graph.

    Args:
        adata: AnnData with raw expression in X and sample, scene, batch, x, y columns in obs.
        k_neighbors: spatial neighbours per cell.
        bimodal_prominence: minimum peak prominence (fraction of the histogram maximum) for peak detection.
        bimodal_min_batch_frac: fraction of batches that must show two or more peaks for a bimodal marker.
    """
    t0 = time.time()
    ref_sample_per_marker = find_best_sample_per_marker(adata)
    adata_base, is_bimodal = shift_normalize_per_marker(
        adata, ref_sample_per_marker,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
    )
    X_base = np.asarray(adata_base.layers["normalized_base"], dtype=np.float32)
    X_scaled, scaler = log1p_scale(X_base)
    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = adata.obs[batch_col].astype("category").cat.codes.values.astype(np.int64)
    regions = region_labels(adata)
    adj = AdjacencyIndex(build_region_graph(adata, regions, k=k_neighbors))
    log.info("Prepared: Stage 1, scaling and region graph in %.0f s; bimodal markers excluded from MMD: %s",
             time.time() - t0, [n for n, b in zip(adata.var_names, is_bimodal) if b])
    return Prepared(adata_base, X_scaled, scaler, np.asarray(is_bimodal, dtype=bool), batch_codes,
                    regions, adj)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    prep: Prepared,
    n_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    device_str: str = "cpu",
    n_per_batch: int = 512,
    gnn_hidden: int = 128,
    gnn_latent: int = 64,
    gnn_heads: int = 4,
    temperature: float = 0.07,
    w_recon: float = 0.1,
    w_contrast: float = 0.5,
    w_adv: float = 0.3,
    w_mmd: float = 1.0,
    mmd_samples: int = 256,
    grl_max: float = 1.0,
) -> Tuple[GNNStage2, Dict[str, List[float]]]:
    """Train the Stage 2 model; returns the model and the per-epoch loss history.

    Args:
        prep: output of prepare().
        n_epochs: training epochs.
        lr: peak learning rate (AdamW).
        weight_decay: AdamW weight decay.
        warmup_epochs: epochs of linear learning-rate warm-up before cosine decay.
        device_str: torch device.
        n_per_batch: cells per batch per step.
        gnn_hidden: hidden width.
        gnn_latent: latent width.
        gnn_heads: attention heads of the first GATv2 layer.
        temperature: NT-Xent temperature.
        w_recon: weight of the Huber loss on the correction.
        w_contrast: weight of the NT-Xent loss.
        w_adv: weight of the adversarial batch-classification loss.
        w_mmd: weight of the MMD loss.
        mmd_samples: maximum cells per batch in the MMD estimate.
        grl_max: final gradient-reversal strength (ramped from 0).
    """
    t0 = time.time()
    device = torch.device(device_str)
    adj, knn = prep.adj, prep.adj.knn
    n_batches = int(prep.batch_codes.max()) + 1

    model = GNNStage2(n_markers=prep.X_scaled.shape[1], n_batches=n_batches, hidden=gnn_hidden,
                      latent=gnn_latent, n_heads=gnn_heads).to(device)
    unimodal_mask = torch.tensor(~prep.is_bimodal, dtype=torch.bool, device=device)
    sampler = NeighbourPatchSampler(prep.batch_codes, knn, n_per_batch=n_per_batch)
    steps = len(sampler)
    pos_rng = np.random.default_rng(123)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
                    CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs))],
        milestones=[warmup_epochs])

    X_dev = torch.tensor(prep.X_scaled, dtype=torch.float32, device=device)
    batch_dev = torch.tensor(prep.batch_codes, dtype=torch.long, device=device)
    huber = nn.HuberLoss(delta=1.0)
    ce = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {k: [] for k in
                                       ("loss", "recon", "contrast", "adv", "mmd", "lr", "grl_lambda")}
    log.info("Stage 2 training: %d epochs x %d steps, w_recon=%.2f w_contrast=%.2f w_adv=%.2f w_mmd=%.2f",
             n_epochs, steps, w_recon, w_contrast, w_adv, w_mmd)

    for epoch in range(n_epochs):
        grl_lam = grl_max * min(1.0, 2.0 * epoch / max(n_epochs - 1, 1))
        model.discriminator.grl.lam = grl_lam
        sums = dict(loss=0.0, recon=0.0, contrast=0.0, adv=0.0, mmd=0.0)
        for _ in range(steps):
            idx = sampler.sample()
            B = len(idx)
            pos_global = knn[idx, pos_rng.integers(0, knn.shape[1], size=B)]
            ctx = adj.expand(idx)
            edge_arr = adj.get_subgraph(ctx)
            pos_local = adj.local_positions(ctx, pos_global)

            ctx_t = torch.as_tensor(ctx, dtype=torch.long, device=device)
            edge_index = torch.as_tensor(edge_arr, dtype=torch.long, device=device)
            delta_all, z_proj_all, logits_all = model(X_dev[ctx_t], edge_index)
            delta, z_proj, logits = delta_all[:B], z_proj_all[:B], logits_all[:B]
            X_batch = X_dev[ctx_t[:B]]
            b_ids = batch_dev[ctx_t[:B]]

            X_out = X_batch + delta
            loss_recon = huber(X_out, X_batch)
            pos_t = torch.as_tensor(pos_local, dtype=torch.long, device=device)
            loss_contrast = nt_xent_pairs(z_proj, z_proj_all[pos_t], temperature)
            loss_adv = ce(logits, b_ids)
            loss_mmd = mmd_rbf_loss(X_out, b_ids, unimodal_mask, n_samples=mmd_samples)
            loss = (w_recon * loss_recon + w_contrast * loss_contrast
                    + w_adv * loss_adv + w_mmd * loss_mmd)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k_, v in (("loss", loss), ("recon", loss_recon), ("contrast", loss_contrast),
                          ("adv", loss_adv), ("mmd", loss_mmd)):
                sums[k_] += v.item()

        epoch_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        for k_ in sums:
            history[k_].append(sums[k_] / max(steps, 1))
        history["lr"].append(epoch_lr)
        history["grl_lambda"].append(grl_lam)
        log.info("Epoch %2d/%d  loss=%.4f  recon=%.4f  contrast=%.4f  adv=%.4f  mmd=%.4f  (%.1f min)",
                 epoch + 1, n_epochs, history["loss"][-1], history["recon"][-1],
                 history["contrast"][-1], history["adv"][-1], history["mmd"][-1], (time.time() - t0) / 60)

    log.info("Training complete in %.1f min", (time.time() - t0) / 60)
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def correct(prep: Prepared, model: GNNStage2, device_str: str = "cpu") -> np.ndarray:
    """Stage 2 correction (scaled space, alpha = 1) for every cell. Each region is processed in chunks
    of 2048 cells run together with their 2-hop halo, so every cell sees its full receptive field.

    Args:
        prep: output of prepare().
        model: trained Stage 2 model.
        device_str: torch device.
    """
    t0 = time.time()
    device = torch.device(device_str)
    model = model.to(device).eval()
    X_dev = torch.tensor(prep.X_scaled, dtype=torch.float32, device=device)
    delta_out = np.zeros_like(prep.X_scaled)
    for rg in np.unique(prep.regions):
        reg_idx = np.where(prep.regions == rg)[0]
        for start in range(0, len(reg_idx), 2048):
            chunk = reg_idx[start:start + 2048]
            ctx = prep.adj.expand(chunk)
            edge_index = torch.as_tensor(prep.adj.get_subgraph(ctx), dtype=torch.long, device=device)
            delta, _, _ = model(X_dev[torch.as_tensor(ctx, dtype=torch.long, device=device)], edge_index)
            delta_out[chunk] = delta[:len(chunk)].cpu().numpy()
    log.info("Correction computed in %.1f s", time.time() - t0)
    return delta_out


def apply(prep: Prepared, delta: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Count-scale output: inverse-scale(Stage 1 scaled + alpha * correction) -> expm1 -> clipped at 0.

    Args:
        prep: output of prepare().
        delta: output of correct().
        alpha: strength of the Stage 2 correction (0 = Stage 1 only).
    """
    X_log = prep.scaler.inverse_transform(prep.X_scaled + alpha * delta)
    return np.clip(np.expm1(X_log), 0, None).astype(np.float32)


def normalize_adata(prep: Prepared, model: GNNStage2, alpha: float = 0.6,
                    device_str: str = "cpu", layer_name: str = "normalized") -> ad.AnnData:
    """Copy of the Stage 1 AnnData with the Stage 2 output in layers[layer_name].

    Args:
        prep: output of prepare().
        model: trained Stage 2 model.
        alpha: strength of the Stage 2 correction (0 = Stage 1 only).
        device_str: torch device.
        layer_name: layer that receives the output.
    """
    out = prep.adata_base.copy()
    out.layers[layer_name] = apply(prep, correct(prep, model, device_str=device_str), alpha)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point: load, prepare, train, normalize and write an .h5ad file."""
    parser = argparse.ArgumentParser(
        description="SpaNCy-Shift spatial: Stage 1 analytic shift + Stage 2 region GNN batch correction")
    parser.add_argument("--input", required=True, help="Input .h5ad")
    parser.add_argument("--output", required=True, help="Output .h5ad")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="torch seed for the model initialisation")
    parser.add_argument("--n_per_batch", type=int, default=512)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Stage 2 correction strength: 0 = Stage 1 only, 1 = full correction")
    parser.add_argument("--w_recon", type=float, default=0.1)
    parser.add_argument("--w_contrast", type=float, default=0.5)
    parser.add_argument("--w_adv", type=float, default=0.3)
    parser.add_argument("--w_mmd", type=float, default=1.0)
    parser.add_argument("--mmd_samples", type=int, default=256)
    parser.add_argument("--grl_max", type=float, default=1.0)
    parser.add_argument("--bimodal_min_batch_frac", type=float, default=0.5)
    parser.add_argument("--layer_name", default="normalized")
    args = parser.parse_args()

    adata = load_adata(args.input)
    prep = prepare(adata, k_neighbors=args.k_neighbors, bimodal_min_batch_frac=args.bimodal_min_batch_frac)
    torch.manual_seed(args.seed)
    model, _ = train(
        prep,
        n_epochs=args.epochs,
        lr=args.lr,
        device_str=args.device,
        n_per_batch=args.n_per_batch,
        w_recon=args.w_recon,
        w_contrast=args.w_contrast,
        w_adv=args.w_adv,
        w_mmd=args.w_mmd,
        mmd_samples=args.mmd_samples,
        grl_max=args.grl_max,
    )
    adata_out = normalize_adata(prep, model, alpha=args.alpha, device_str=args.device,
                                layer_name=args.layer_name)
    adata_out.write_h5ad(args.output)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
