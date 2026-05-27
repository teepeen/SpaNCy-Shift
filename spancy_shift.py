#!/usr/bin/env python
"""
SpaNCy-Shift: Two-Stage Batch Normalization for CyCIF Multiplexed Imaging.

Stage 1 (analytic): Per-marker shift correction toward a KL-medoid reference sample.
  - Unimodal markers: single median shift per sample in log1p space.
  - Bimodal markers:  separate neg/pos peak shifts blended by sigmoid.
  - No parameters to learn. Output: X_base with aligned 1D histograms (kBET ≈ 0.631).

Stage 2 (GNN): Spatial GATv2 encoder + residual decoder trained on Stage 1 output.
  - Spatial k-NN graph built from (x, y) per scene.
  - NT-Xent contrastive loss (spatial neighbors as positives) + adversarial GRL.
  - Residual decoder outputs delta; inference: X_out = X_base + alpha * delta.
  - Operates at the cell level → can improve 20D covariance structure for kBET.
  - Output: X_final stored in adata.layers['normalized'].

Usage:
    model, scaler, ref_sample_per_marker, history = train(adata, n_epochs=50, device_str='cuda')
    adata_norm = normalize_adata(adata, model, scaler, ref_sample_per_marker)
"""

import argparse
import logging
import math
import sys
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as _sp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks as _scipy_find_peaks
from scipy.stats import entropy as _kl_entropy
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

log = logging.getLogger("spancy_shift")
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
    """Load AnnData, auto-detect batch column, set var_names from marker_name."""
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


def _get_col(adata: ad.AnnData, candidates: List[str], fallback: str) -> str:
    for c in candidates:
        if c in adata.obs.columns:
            return c
    return fallback


def log1p_scale(X: np.ndarray) -> Tuple[np.ndarray, RobustScaler]:
    """log1p → per-marker RobustScaler (median/IQR). Returns (X_scaled, scaler)."""
    X_log = np.log1p(np.clip(X, 0, None))
    scaler = RobustScaler()
    return scaler.fit_transform(X_log).astype(np.float32), scaler


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Analytic bimodal-aware shift normalization
# ──────────────────────────────────────────────────────────────────────────────

def _find_peaks_1d(
    values: np.ndarray,
    n_bins: int = 150,
    min_prominence_frac: float = 0.05,
    sigma: float = 2.0,
) -> np.ndarray:
    """Find histogram peaks in a 1D distribution. Returns array of peak positions."""
    counts, edges = np.histogram(values, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    prom = min_prominence_frac * float(smoothed.max())
    peak_idx, _ = _scipy_find_peaks(smoothed, prominence=prom)
    if len(peak_idx) == 0:
        return np.array([centers[int(np.argmax(smoothed))]])
    return centers[peak_idx]


def detect_bimodal_markers(
    X_log: np.ndarray,
    marker_names: List[str],
    batch_codes: Optional[np.ndarray] = None,
    n_bins: int = 150,
    min_prominence_frac: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
    sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify markers as bimodal via per-batch histogram peak voting.

    A marker is bimodal if >= bimodal_min_batch_frac of batches independently
    show >= 2 peaks. Thresholds are midpoints between neg/pos peaks, in the
    same space as X_log.

    Returns (is_bimodal [M, bool], thresholds [M, float]).
    """
    n_markers = X_log.shape[1]
    is_bimodal = np.zeros(n_markers, dtype=bool)
    thresholds = np.zeros(n_markers, dtype=float)

    if batch_codes is None:
        batch_codes = np.zeros(len(X_log), dtype=int)
    unique_batches = np.unique(batch_codes)

    log.info("Bimodal detection (%d markers, %d batches):", n_markers, len(unique_batches))
    for m, mname in enumerate(marker_names):
        x = X_log[:, m]
        n_valid = n_bimodal = 0
        midpoints = []
        for b in unique_batches:
            x_b = x[batch_codes == b]
            if len(x_b) < 100:
                continue
            n_valid += 1
            peaks = _find_peaks_1d(x_b, n_bins=n_bins,
                                   min_prominence_frac=min_prominence_frac, sigma=sigma)
            if len(peaks) >= 2:
                n_bimodal += 1
                s = np.sort(peaks)
                midpoints.append((s[0] + s[1]) / 2.0)
        if n_valid > 0 and (n_bimodal / n_valid) >= bimodal_min_batch_frac:
            is_bimodal[m] = True
            thresholds[m] = float(np.median(midpoints)) if midpoints else 0.0
            log.info("  %-14s  BIMODAL  threshold=%.3f  (%d/%d batches)",
                     mname, thresholds[m], n_bimodal, n_valid)
        else:
            log.info("  %-14s  unimodal (%d/%d batches bimodal)",
                     mname, n_bimodal, n_valid if n_valid else 0)

    return is_bimodal, thresholds


def find_best_sample_per_marker(
    adata: ad.AnnData,
    n_bins: int = 50,
    min_cells_per_sample: int = 100,
) -> Dict[str, Optional[str]]:
    """KL medoid reference: select the sample with lowest mean pairwise symmetric KL
    divergence to all others (log1p histogram space), per marker.

    Returns dict {marker_name: sample_id}.
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
        bins = np.histogram_bin_edges(X_log, bins=n_bins)
        hists: Dict[str, np.ndarray] = {}
        for s in unique_samples:
            v = X_log[samples == s]
            if len(v) < min_cells_per_sample:
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
    ref_med = float(np.median(X_log[sample_labels == target_sample]))
    return {
        s: ref_med - float(np.median(X_log[sample_labels == s]))
        if (sample_labels == s).sum() >= 10 else 0.0
        for s in np.unique(sample_labels)
    }


def _shifts_bimodal(
    X_log: np.ndarray, sample_labels: np.ndarray, target_sample: str,
    threshold: float, min_prominence_frac: float = 0.05, sigma: float = 2.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    ref_peaks = np.sort(_find_peaks_1d(X_log[sample_labels == target_sample],
                                       min_prominence_frac=min_prominence_frac, sigma=sigma))
    ref_neg = float(ref_peaks[0])
    ref_pos = float(ref_peaks[-1]) if len(ref_peaks) > 1 else ref_neg
    shifts_neg, shifts_pos = {}, {}
    for s in np.unique(sample_labels):
        v = X_log[sample_labels == s]
        if len(v) < 10:
            shifts_neg[s] = shifts_pos[s] = 0.0
            continue
        s_peaks = np.sort(_find_peaks_1d(v, min_prominence_frac=min_prominence_frac, sigma=sigma))
        if len(s_peaks) >= 2:
            s_neg, s_pos = float(s_peaks[0]), float(s_peaks[-1])
        else:
            s_neg = s_pos = float(s_peaks[0])
        shifts_neg[s] = ref_neg - s_neg
        shifts_pos[s] = ref_pos - s_pos
    return shifts_neg, shifts_pos


def _apply_bimodal(
    X_log: np.ndarray, sample_labels: np.ndarray,
    shifts_neg: Dict, shifts_pos: Dict, threshold: float, sharpness: float = 10.0,
) -> np.ndarray:
    out = X_log.copy()
    for s in np.unique(sample_labels):
        mask = sample_labels == s
        x = X_log[mask]
        w_pos = 1.0 / (1.0 + np.exp(-(x - threshold) * sharpness))
        out[mask] = x + (1.0 - w_pos) * shifts_neg[s] + w_pos * shifts_pos[s]
    return out


def _apply_unimodal(
    X_log: np.ndarray, sample_labels: np.ndarray, shifts: Dict[str, float],
) -> np.ndarray:
    out = X_log.copy()
    for s, shift in shifts.items():
        mask = sample_labels == s
        out[mask] = X_log[mask] + shift
    return out


def shift_normalize_per_marker(
    adata: ad.AnnData,
    marker_to_best_sample: Dict[str, Optional[str]],
    min_prominence_frac: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
    sharpness: float = 10.0,
    sigma: float = 2.0,
    layer_name: str = "normalized_base",
) -> ad.AnnData:
    """Stage 1: Analytically shift each sample toward its per-marker KL-medoid reference.

    Bimodal markers: separate neg/pos peak shifts blended by sigmoid.
    Unimodal markers: single median shift.
    All computation in log1p space; output is expm1-transformed back to count scale.

    Returns a copy of adata with layers[layer_name] = normalized expression.
    """
    X = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X, dtype=np.float64)
    sample_col = _get_col(adata, ["sample_id", "sample", "patient_id"], "batch_id")
    samples = adata.obs[sample_col].values
    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = pd.Categorical(adata.obs[batch_col].values).codes
    marker_names = list(adata.var_names)

    X_log_all = np.log1p(X)
    is_bimodal, thresholds = detect_bimodal_markers(
        X_log_all, marker_names, batch_codes=batch_codes,
        min_prominence_frac=min_prominence_frac,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
        sigma=sigma,
    )

    normalized = X.copy()
    for k, mname in enumerate(marker_names):
        target = marker_to_best_sample.get(mname)
        if target is None:
            log.info("[%d/%d] %s — skipped (no reference)", k + 1, len(marker_names), mname)
            continue
        X_log = X_log_all[:, k]
        if is_bimodal[k]:
            log.info("[%d/%d] %s  BIMODAL  ref=%s  threshold=%.3f",
                     k + 1, len(marker_names), mname, target, thresholds[k])
            sn, sp2 = _shifts_bimodal(X_log, samples, target, thresholds[k],
                                      min_prominence_frac=min_prominence_frac, sigma=sigma)
            X_log_out = _apply_bimodal(X_log, samples, sn, sp2, thresholds[k], sharpness)
        else:
            log.info("[%d/%d] %s  unimodal  ref=%s", k + 1, len(marker_names), mname, target)
            shifts = _shifts_unimodal(X_log, samples, target)
            X_log_out = _apply_unimodal(X_log, samples, shifts)
        normalized[:, k] = np.clip(np.expm1(X_log_out), 0, None)

    adata_out = adata.copy()
    adata_out.layers[layer_name] = normalized.astype(np.float32)
    log.info("Stage 1 done. Layer '%s' written (%d cells × %d markers).",
             layer_name, adata.n_obs, adata.n_vars)
    return adata_out, is_bimodal, thresholds


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def _otsu_threshold(values: np.ndarray, n_bins: int = 200) -> float:
    lo, hi = np.percentile(values, [1, 99])
    if hi - lo < 1e-8:
        return float(np.median(values))
    counts, edges = np.histogram(values, bins=np.linspace(lo, hi, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    if total == 0:
        return float(np.median(values))
    best_thresh, best_var = centers[0], -1.0
    sum_total = (counts * centers).sum()
    sum_bg = weight_bg = 0.0
    for i in range(len(counts)):
        weight_bg += counts[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += counts[i] * centers[i]
        var_b = weight_bg * weight_fg * (sum_bg / weight_bg - (sum_total - sum_bg) / weight_fg) ** 2
        if var_b > best_var:
            best_var = var_b
            best_thresh = centers[i]
    return float(best_thresh)


def _gmm_threshold(values: np.ndarray, max_cells: int = 50_000) -> float:
    """GMM-based pos/neg threshold matching UniFORM paper methodology.

    Fits 2-component GMM (subsampled for speed), hard-assigns all values via predict(),
    returns max(negative class). Falls back to Otsu if degenerate.
    """
    v_fit = values
    if len(v_fit) > max_cells:
        v_fit = np.random.default_rng(0).choice(v_fit, size=max_cells, replace=False)
    try:
        gm = GaussianMixture(n_components=2, random_state=0, max_iter=200).fit(
            v_fit.reshape(-1, 1)
        )
        neg_comp = int(np.argmin(gm.means_.ravel()))
        labels = gm.predict(values.reshape(-1, 1))
        neg_mask = labels == neg_comp
        if neg_mask.sum() > 0:
            thr = float(values[neg_mask].max())
            lo, hi = np.percentile(values, [5, 95])
            if lo < thr < hi:
                return thr
    except Exception:
        pass
    return _otsu_threshold(values)


def positive_population_table(
    adata: ad.AnnData,
    raw_layer: Optional[str] = None,
    norm_layer: str = "normalized",
    sample_col: str = "sample_id",
    marker_names: Optional[List[str]] = None,
    log_transform: bool = True,
) -> pd.DataFrame:
    """Per-marker per-sample positive cell % — per-sample local GMM threshold.

    GMM threshold computed per sample on raw data, applied consistently to both raw and
    normalized for that sample. Delta measures only cells crossing the sample's own
    positive/negative boundary due to normalization — the most direct measure of
    within-sample biology preservation. Robust to inter-sample intensity variation
    (e.g., one outlier sample cannot skew the threshold for all others).
    Target: |delta| < 5% per marker.
    """
    if marker_names is None:
        marker_names = list(adata.var_names)
    X_raw = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X)
    if raw_layer is not None:
        X_raw = np.asarray(adata.layers[raw_layer])
    X_norm = np.asarray(adata.layers[norm_layer])
    if log_transform:
        X_raw = np.log10(np.clip(X_raw, 0, None) + 1.0)
        X_norm = np.log10(np.clip(X_norm, 0, None) + 1.0)
    s_col = _get_col(adata, [sample_col, "sample_id", "sample", "patient_id"], "batch_id")
    sample_ids = adata.obs[s_col].values
    rows = []
    for m, mname in enumerate(marker_names):
        for s in sorted(np.unique(sample_ids).tolist()):
            mask = sample_ids == s
            if mask.sum() < 10:
                continue
            thr_local = _gmm_threshold(X_raw[mask, m])
            pr = 100.0 * (X_raw[mask, m] > thr_local).mean()
            pn = 100.0 * (X_norm[mask, m] > thr_local).mean()
            rows.append({"marker": mname, "sample": s,
                         "pct_pos_raw": round(pr, 2), "pct_pos_norm": round(pn, 2),
                         "delta": round(pn - pr, 2)})
    return pd.DataFrame(rows)


def per_marker_batch_r2(
    X: np.ndarray,
    batch_labels: np.ndarray,
    marker_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Adjusted R² for each marker regressed on batch one-hot. Lower = less batch effect."""
    enc = OneHotEncoder(sparse_output=False, drop="first")
    B = enc.fit_transform(batch_labels.reshape(-1, 1))
    n, k = B.shape
    rows = []
    for j in range(X.shape[1]):
        y = X[:, j]
        y_hat = LinearRegression().fit(B, y).predict(B)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        rows.append({"marker": marker_names[j] if marker_names else f"m{j}",
                     "adj_r2": float(adj_r2)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Spatial GNN residual correction
# ──────────────────────────────────────────────────────────────────────────────

try:
    from torch_geometric.nn import GATv2Conv as _GATv2Conv
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _GATv2Conv = None  # type: ignore[assignment,misc]
    _HAS_TORCH_GEOMETRIC = False


class _GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def forward(self, x):
        return _GradRevFn.apply(x, self.lam)


class SpatialGNNEncoder(nn.Module):
    def __init__(self, n_markers: int, hidden: int = 128, latent: int = 64, n_heads: int = 4):
        super().__init__()
        if not _HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for Stage 2 GNN. "
                "Install: pip install torch-geometric torch-scatter torch-sparse"
            )
        self.proj = nn.Linear(n_markers, hidden)
        self.conv1 = _GATv2Conv(hidden, hidden // n_heads, heads=n_heads, concat=True,
                                dropout=0.1, add_self_loops=True)
        self.conv2 = _GATv2Conv(hidden, latent, heads=1, concat=False,
                                dropout=0.1, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(latent)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(x))
        h = self.act(self.norm1(self.conv1(h, edge_index) + h))
        return self.norm2(self.conv2(h, edge_index))


class _ResidualDecoder(nn.Module):
    def __init__(self, latent: int = 64, hidden: int = 128, n_markers: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(),
            nn.Linear(hidden, n_markers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _ProjectionHead(nn.Module):
    def __init__(self, latent: int = 64, proj_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, latent), nn.GELU(),
            nn.Linear(latent, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(z), dim=1)


class _BatchDiscriminator(nn.Module):
    def __init__(self, latent: int = 64, hidden: int = 32, n_batches: int = 7):
        super().__init__()
        self.grl = GradientReversal()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, n_batches),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(z))


class GNNStage2(nn.Module):
    """Stage 2: Spatial GATv2 encoder → residual decoder + projection head + batch discriminator."""

    def __init__(
        self,
        n_markers: int,
        n_batches: int,
        hidden: int = 128,
        latent: int = 64,
        proj_dim: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()
        self.encoder = SpatialGNNEncoder(n_markers, hidden, latent, n_heads)
        self.decoder = _ResidualDecoder(latent, hidden, n_markers)
        self.proj_head = _ProjectionHead(latent, proj_dim)
        self.discriminator = _BatchDiscriminator(latent, 32, n_batches)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x, edge_index)
        delta = self.decoder(z)
        z_proj = self.proj_head(z)
        batch_logits = self.discriminator(z)
        return z, delta, z_proj, batch_logits


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Spatial k-NN graph
# ──────────────────────────────────────────────────────────────────────────────

def build_knn_graphs(
    adata: ad.AnnData,
    k: int = 15,
) -> np.ndarray:
    """Build per-scene spatial k-NN. Returns (N, k) int32 array of neighbor global indices."""
    scene_col = _get_col(adata, ["scene_id", "sample_id"], "batch_id")
    xy_x = _get_col(adata, ["x", "X", "centroid_x"], None)
    xy_y = _get_col(adata, ["y", "Y", "centroid_y"], None)
    if xy_x is None or xy_y is None or xy_x not in adata.obs or xy_y not in adata.obs:
        raise ValueError("Spatial coordinates (x, y) not found in adata.obs")

    scenes = adata.obs[scene_col].values
    unique_scenes = np.unique(scenes)
    n_cells = adata.n_obs
    knn_matrix = np.zeros((n_cells, k), dtype=np.int32)

    log.info("Building spatial k-NN (k=%d) for %d scenes...", k, len(unique_scenes))
    for sc in unique_scenes:
        mask = scenes == sc
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        xy = adata.obs.loc[mask, [xy_x, xy_y]].values.astype(np.float32)
        k_eff = min(k + 1, len(idx))
        nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree").fit(xy)
        _, indices = nbrs.kneighbors(xy)
        global_nbrs = idx[indices[:, 1:k + 1]]  # skip self; pad if fewer than k
        if global_nbrs.shape[1] < k:
            pad = np.tile(global_nbrs[:, :1], (1, k - global_nbrs.shape[1]))
            global_nbrs = np.concatenate([global_nbrs, pad], axis=1)
        knn_matrix[idx] = global_nbrs.astype(np.int32)

    log.info("k-NN built. Matrix shape: %s", knn_matrix.shape)
    return knn_matrix


class AdjacencyIndex:
    """Precomputed (N, k) k-NN matrix for vectorized subgraph extraction."""

    def __init__(self, knn_matrix: np.ndarray):
        self.knn = knn_matrix  # (N, k) int32
        self.N = knn_matrix.shape[0]

    def get_subgraph(self, idx: np.ndarray) -> np.ndarray:
        """Return edge_index (2, E) in local coordinates for cells idx."""
        B, k = len(idx), self.knn.shape[1]
        adj = self.knn[idx]  # (B, k) global neighbor indices
        in_set = np.zeros(self.N, dtype=bool)
        in_set[idx] = True
        in_batch = in_set[adj.ravel()].reshape(B, k)
        if not in_batch.any():
            return np.zeros((2, 0), dtype=np.int64)
        src_local = np.repeat(np.arange(B), k)[in_batch.ravel()]
        dst_global = adj.ravel()[in_batch.ravel()]
        local_map = np.full(self.N, -1, dtype=np.int64)
        local_map[idx] = np.arange(B)
        dst_local = local_map[dst_global]
        return np.stack([src_local, dst_local])


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Losses and sampler
# ──────────────────────────────────────────────────────────────────────────────

def nt_xent_loss(
    z_proj: torch.Tensor,
    edge_index: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent with spatial neighbors as positive pairs."""
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=z_proj.device)
    src, dst = edge_index[0], edge_index[1]
    sim = torch.mm(z_proj, z_proj.t()) / temperature  # (N, N)
    sim.fill_diagonal_(-1e9)
    log_sm = nn.functional.log_softmax(sim[src], dim=1)
    return -log_sm[torch.arange(len(src), device=z_proj.device), dst].mean()


def mmd_rbf_loss(
    X_out: torch.Tensor,
    batch_ids: torch.Tensor,
    unimodal_mask: Optional[torch.Tensor] = None,
    bandwidths: Tuple[float, ...] = (0.5, 1.0, 5.0),
    n_samples: int = 256,
) -> torch.Tensor:
    """Multi-scale RBF MMD² across all batch pairs in the mini-batch.

    Only unimodal marker dims are used (bimodal markers like ECAD are excluded to prevent
    MMD from treating bimodal biology as a batch effect).
    """
    X = X_out[:, unimodal_mask] if unimodal_mask is not None else X_out
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


class SceneBasedSampler:
    """Each step: one scene per batch, sample n_per_batch cells from it."""

    def __init__(
        self,
        batch_codes: np.ndarray,
        scene_codes: np.ndarray,
        n_per_batch: int = 512,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.n_per_batch = n_per_batch
        unique_batches = np.unique(batch_codes)
        self.unique_batches = unique_batches
        self.batch_scenes: Dict[int, np.ndarray] = {}
        self.scene_indices: Dict[Tuple[int, int], np.ndarray] = {}

        for b in unique_batches:
            b_mask = batch_codes == b
            scenes_in_b = np.unique(scene_codes[b_mask])
            self.batch_scenes[int(b)] = scenes_in_b
            for sc in scenes_in_b:
                self.scene_indices[(int(b), int(sc))] = np.where(
                    b_mask & (scene_codes == sc)
                )[0]

        n_total = len(batch_codes)
        cells_per_step = n_per_batch * len(unique_batches)
        self._n_steps = max(1, n_total // cells_per_step)

    def sample(self) -> np.ndarray:
        parts = []
        for b in self.unique_batches:
            sc = int(self.rng.choice(self.batch_scenes[int(b)]))
            idx = self.scene_indices[(int(b), sc)]
            n = min(self.n_per_batch, len(idx))
            parts.append(self.rng.choice(idx, size=n, replace=False))
        return np.concatenate(parts)

    def __len__(self) -> int:
        return self._n_steps


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    adata: ad.AnnData,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cpu",
    warmup_epochs: int = 5,
    n_per_batch: int = 512,
    k_neighbors: int = 15,
    gnn_hidden: int = 128,
    gnn_latent: int = 64,
    gnn_heads: int = 4,
    hybrid_alpha: float = 0.3,
    temperature: float = 0.07,
    w_recon: float = 0.1,
    w_contrast: float = 0.5,
    w_adv: float = 0.3,
    w_mmd: float = 1.0,
    mmd_samples: int = 256,
    grl_max: float = 1.0,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
    ref_sample_per_marker: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple["GNNStage2", RobustScaler, Dict[str, Optional[str]], Dict[str, List[float]]]:
    """Train SpaNCy-Shift two-stage model.

    Stage 1 (analytic): shift_normalize_per_marker → X_base (kBET ≈ 0.631 baseline).
    Stage 2 (GNN): SpatialGNNEncoder + ResidualDecoder trained on X_base.
      Losses: MMD on decoder output (unimodal dims only, across batch pairs) +
              NT-Xent spatial contrastive (encoder) + adversarial GRL (encoder) +
              Huber recon soft regularizer (keeps delta small).
      Inference: X_out = X_base_scaled + hybrid_alpha * delta → inverse_transform.

    Returns (model, scaler, ref_sample_per_marker, history).
    Pass all four return values to normalize_adata().
    """
    device = torch.device(device_str)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if ref_sample_per_marker is None:
        ref_sample_per_marker = find_best_sample_per_marker(adata)

    log.info("Stage 1: analytic shift normalization...")
    adata_base, is_bimodal, _ = shift_normalize_per_marker(
        adata, ref_sample_per_marker,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
        layer_name="normalized_base",
    )
    log.info("Bimodal markers excluded from MMD: %s",
             [list(adata.var_names)[i] for i, b in enumerate(is_bimodal) if b])

    # ── Stage 2 setup ─────────────────────────────────────────────────────────
    X_base = np.asarray(adata_base.layers["normalized_base"], dtype=np.float32)
    X_base_scaled, scaler = log1p_scale(X_base)

    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = adata.obs[batch_col].astype("category").cat.codes.values.astype(np.int64)
    n_batches = int(batch_codes.max()) + 1

    scene_col = _get_col(adata, ["scene_id", "sample_id"], batch_col)
    scene_codes = adata.obs[scene_col].astype("category").cat.codes.values.astype(np.int64)

    marker_names = list(adata.var_names)
    log.info("Stage 2 setup: %d batches, %d markers, device=%s",
             n_batches, len(marker_names), device_str)

    # Build spatial k-NN
    knn_matrix = build_knn_graphs(adata, k=k_neighbors)
    adj_index = AdjacencyIndex(knn_matrix)

    model = GNNStage2(
        n_markers=adata.n_vars,
        n_batches=n_batches,
        hidden=gnn_hidden,
        latent=gnn_latent,
        n_heads=gnn_heads,
    ).to(device)
    log.info("GNNStage2: %d parameters", sum(p.numel() for p in model.parameters()))

    # Unimodal mask: exclude bimodal markers (e.g. ECAD) from MMD to prevent biology destruction
    unimodal_mask = torch.tensor(~is_bimodal, dtype=torch.bool, device=device)

    sampler = SceneBasedSampler(batch_codes, scene_codes, n_per_batch=n_per_batch)
    steps_per_epoch = len(sampler)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                              milestones=[warmup_epochs])

    X_t = torch.tensor(X_base_scaled, dtype=torch.float32)
    batch_t = torch.tensor(batch_codes, dtype=torch.long)
    huber = nn.HuberLoss(delta=1.0)
    ce = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "loss": [], "recon": [], "contrast": [], "adv": [], "mmd": [], "lr": [], "grl_lambda": [],
    }

    log.info(
        "Stage 2 training: %d epochs × %d steps  "
        "w_recon=%.2f  w_contrast=%.2f  w_adv=%.2f  w_mmd=%.2f  k=%d  n_per_batch=%d  device=%s",
        n_epochs, steps_per_epoch, w_recon, w_contrast, w_adv, w_mmd,
        k_neighbors, n_per_batch, device_str,
    )

    for epoch in range(n_epochs):
        grl_lam = grl_max * min(1.0, 2.0 * epoch / max(n_epochs - 1, 1))
        model.discriminator.grl.lam = grl_lam

        e_loss = e_recon = e_contrast = e_adv = e_mmd = 0.0

        for step in range(steps_per_epoch):
            idx = sampler.sample()
            if len(idx) < 4:
                continue

            edge_arr = adj_index.get_subgraph(idx)
            edge_index = torch.tensor(edge_arr, dtype=torch.long, device=device)

            X_batch = X_t[idx].to(device)
            b_ids = batch_t[idx].to(device)

            _, delta, z_proj, batch_logits = model(X_batch, edge_index)

            X_out = X_batch + delta
            loss_recon = huber(X_out, X_batch)   # soft regularizer: keeps delta small
            loss_contrast = nt_xent_loss(z_proj, edge_index, temperature)
            loss_adv = ce(batch_logits, b_ids)
            loss_mmd = mmd_rbf_loss(X_out, b_ids, unimodal_mask,
                                    n_samples=mmd_samples)
            loss = (w_recon * loss_recon + w_contrast * loss_contrast
                    + w_adv * loss_adv + w_mmd * loss_mmd)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            e_loss += loss.item()
            e_recon += loss_recon.item()
            e_contrast += loss_contrast.item()
            e_adv += loss_adv.item()
            e_mmd += loss_mmd.item()

            if steps_per_epoch > 20 and (step + 1) % max(1, steps_per_epoch // 5) == 0:
                log.info(
                    "  E%3d step %d/%d  loss=%.4f  recon=%.4f  contrast=%.4f  "
                    "adv=%.4f  mmd=%.4f",
                    epoch + 1, step + 1, steps_per_epoch,
                    loss.item(), loss_recon.item(), loss_contrast.item(),
                    loss_adv.item(), loss_mmd.item(),
                )

        scheduler.step()
        s = max(steps_per_epoch, 1)
        history["loss"].append(e_loss / s)
        history["recon"].append(e_recon / s)
        history["contrast"].append(e_contrast / s)
        history["adv"].append(e_adv / s)
        history["mmd"].append(e_mmd / s)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grl_lambda"].append(grl_lam)

        log.info(
            "Epoch %3d/%d  loss=%.4f  recon=%.4f  contrast=%.4f  adv=%.4f  "
            "mmd=%.4f  grl_λ=%.3f  lr=%.2e",
            epoch + 1, n_epochs,
            history["loss"][-1], history["recon"][-1],
            history["contrast"][-1], history["adv"][-1],
            history["mmd"][-1], grl_lam, optimizer.param_groups[0]["lr"],
        )

    log.info("Training complete. Final loss=%.4f", history["loss"][-1])
    return model, scaler, ref_sample_per_marker, history


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def normalize_adata(
    adata: ad.AnnData,
    model: "GNNStage2",
    scaler: RobustScaler,
    ref_sample_per_marker: Dict[str, Optional[str]],
    hybrid_alpha: float = 0.3,
    k_neighbors: int = 15,
    inference_batch_size: int = 2048,
    device_str: str = "cpu",
    layer_name: str = "normalized",
    keep_base_layer: bool = True,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
) -> ad.AnnData:
    """Two-stage inference.

    Stage 1: shift_normalize_per_marker → 'normalized_base' layer.
    Stage 2: GNNStage2 residual delta blended by hybrid_alpha → layer_name layer.
      X_out = X_base_scaled + hybrid_alpha * delta → inverse_transform.
    inference_batch_size: cells per chunk within each scene (reduce if OOM).
    """
    # Stage 1
    log.info("Inference Stage 1: analytic shift normalization...")
    adata_out, _, _ = shift_normalize_per_marker(
        adata, ref_sample_per_marker,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
        layer_name="normalized_base",
    )

    # Stage 2
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    X_base = np.asarray(adata_out.layers["normalized_base"], dtype=np.float32)
    X_scaled = scaler.transform(np.log1p(np.clip(X_base, 0, None))).astype(np.float32)

    log.info("Inference Stage 2: building k-NN for %d cells...", adata.n_obs)
    knn_matrix = build_knn_graphs(adata, k=k_neighbors)
    adj_index = AdjacencyIndex(knn_matrix)

    scene_col = _get_col(adata, ["scene_id", "sample_id"], "batch_id")
    scenes = adata.obs[scene_col].values
    unique_scenes = np.unique(scenes)

    X_norm_scaled = X_scaled.copy()
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    n_chunks_total = sum(
        max(1, math.ceil(np.sum(scenes == sc) / inference_batch_size))
        for sc in unique_scenes
    )
    chunk_done = 0

    for sc in unique_scenes:
        scene_idx = np.where(scenes == sc)[0]
        # Process each scene in fixed-size chunks to avoid OOM on large scenes
        for start in range(0, len(scene_idx), inference_batch_size):
            chunk_idx = scene_idx[start:start + inference_batch_size]
            edge_arr = adj_index.get_subgraph(chunk_idx)
            edge_index = torch.tensor(edge_arr, dtype=torch.long, device=device)
            X_chunk = X_t[chunk_idx].to(device)
            _, delta, _, _ = model(X_chunk, edge_index)
            X_norm_scaled[chunk_idx] = (X_chunk + hybrid_alpha * delta).cpu().numpy()
            chunk_done += 1
            if chunk_done % max(1, n_chunks_total // 10) == 0 or chunk_done == n_chunks_total:
                log.info("  inference %d/%d chunks", chunk_done, n_chunks_total)

    X_norm_log = scaler.inverse_transform(X_norm_scaled)
    X_final = np.clip(np.expm1(X_norm_log), 0, None).astype(np.float32)
    adata_out.layers[layer_name] = X_final

    if not keep_base_layer:
        del adata_out.layers["normalized_base"]

    log.info(
        "Done. Layer '%s' written (hybrid_alpha=%.2f). min=%.4f  max=%.4f  mean=%.4f",
        layer_name, hybrid_alpha, X_final.min(), X_final.max(), X_final.mean(),
    )
    return adata_out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpaNCy-Shift: two-stage CyCIF batch correction (Stage 1 analytic + Stage 2 GNN)"
    )
    parser.add_argument("--input", required=True, help="Input .h5ad")
    parser.add_argument("--output", required=True, help="Output .h5ad")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_per_batch", type=int, default=512,
                        help="Cells per batch per step in scene-based sampler")
    parser.add_argument("--k_neighbors", type=int, default=15,
                        help="Spatial k-NN neighbors")
    parser.add_argument("--hybrid_alpha", type=float, default=0.3,
                        help="GNN delta blend: 0=Stage1 only, 1=full GNN residual")
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
    model, scaler, ref_sample_per_marker, history = train(
        adata,
        n_epochs=args.epochs,
        lr=args.lr,
        device_str=args.device,
        n_per_batch=args.n_per_batch,
        k_neighbors=args.k_neighbors,
        hybrid_alpha=args.hybrid_alpha,
        w_recon=args.w_recon,
        w_contrast=args.w_contrast,
        w_adv=args.w_adv,
        w_mmd=args.w_mmd,
        mmd_samples=args.mmd_samples,
        grl_max=args.grl_max,
        bimodal_min_batch_frac=args.bimodal_min_batch_frac,
    )
    adata_norm = normalize_adata(
        adata, model, scaler, ref_sample_per_marker,
        hybrid_alpha=args.hybrid_alpha,
        k_neighbors=args.k_neighbors,
        device_str=args.device,
        layer_name=args.layer_name,
    )
    adata_norm.write_h5ad(args.output)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()

