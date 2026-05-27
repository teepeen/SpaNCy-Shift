#!/usr/bin/env python
"""
SpaNCy-Shift-DL: Single-Stage Deep Learning Batch Normalization for CyCIF Multiplexed Imaging.

Reference selection (analytic, one-time): KL-medoid reference sample per marker.
Bimodal detection (analytic, one-time): per-batch histogram peak voting.
DL training: ResidualShiftModel trained directly on raw log1p data with three losses:
  - L_ref:   per-sample soft-mean alignment to precomputed reference distribution anchors
  - L_mmd:   20D RBF MMD across batches (bimodal dims masked, ramped mmd_ramp_start→mmd_ramp_end)
  - L_recon: Huber to identity (keeps shifts small, biology-preserving)
Output: X_final stored in adata.layers['normalized'].

Unlike spancy_shift.py (two-stage), this module has no analytic Stage 1. The reference
alignment is learned via L_ref rather than computed analytically. The KL-medoid reference
selection and bimodal detection remain analytic (cheap, one-time data statistics).

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
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

log = logging.getLogger("spancy_shift_dl")
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
# Reference selection & bimodal detection (analytic, one-time)
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


def compute_reference_anchors(
    X_scaled: np.ndarray,
    ref_sample_per_marker: Dict[str, Optional[str]],
    sample_labels: np.ndarray,
    marker_names: List[str],
    is_bimodal: np.ndarray,
    thresholds_scaled: np.ndarray,
    sharpness: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute reference distribution anchors in scaled space (one-time, before training).

    For each marker m, extracts cells belonging to that marker's KL-medoid reference sample
    and computes soft-weighted mean intensities for neg and pos sub-populations.
    These fixed anchors are the targets for L_ref during training.

    Unimodal markers:
        neg_anchor[m] = mean of all reference cells for marker m
        pos_anchor[m] = same (pos anchor unused for unimodal markers)

    Bimodal markers:
        w_pos[i] = sigmoid((X_scaled[ref_cells, m] - thresholds_scaled[m]) * sharpness)
        neg_anchor[m] = sum((1 - w_pos) * x_ref) / (sum(1 - w_pos) + eps)
        pos_anchor[m] = sum(w_pos * x_ref) / (sum(w_pos) + eps)

    Returns (neg_anchors [M], pos_anchors [M]) as float32 numpy arrays.
    """
    M = len(marker_names)
    neg_anchors = np.zeros(M, dtype=np.float64)
    pos_anchors = np.zeros(M, dtype=np.float64)
    eps = 1e-8

    for m, mname in enumerate(marker_names):
        ref_s = ref_sample_per_marker.get(mname)
        if ref_s is None:
            log.warning("compute_reference_anchors: no reference for %s, anchor=0", mname)
            continue
        mask = sample_labels == ref_s
        if mask.sum() < 10:
            log.warning("compute_reference_anchors: ref sample '%s' too small for %s", ref_s, mname)
            continue
        x_ref = X_scaled[mask, m].astype(np.float64)

        if is_bimodal[m]:
            w_pos = 1.0 / (1.0 + np.exp(-(x_ref - float(thresholds_scaled[m])) * sharpness))
            w_neg = 1.0 - w_pos
            neg_anchors[m] = float(np.sum(w_neg * x_ref) / (np.sum(w_neg) + eps))
            pos_anchors[m] = float(np.sum(w_pos * x_ref) / (np.sum(w_pos) + eps))
        else:
            neg_anchors[m] = float(np.mean(x_ref))
            pos_anchors[m] = neg_anchors[m]

    bimodal_count = int(is_bimodal.sum())
    log.info("Reference anchors computed: %d markers (%d bimodal, %d unimodal).",
             M, bimodal_count, M - bimodal_count)
    return neg_anchors.astype(np.float32), pos_anchors.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# DL losses
# ──────────────────────────────────────────────────────────────────────────────

def reference_alignment_loss(
    X_shifted: torch.Tensor,
    X_batch_input: torch.Tensor,
    sample_ids: torch.Tensor,
    neg_anchors_t: torch.Tensor,
    pos_anchors_t: torch.Tensor,
    is_bimodal: torch.Tensor,
    thresholds_scaled: torch.Tensor,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """Per-sample reference alignment loss (L_ref).

    For each unique sample s in the mini-batch:
      Unimodal markers: squared error of per-sample mean vs reference neg_anchor.
      Bimodal markers:  soft sigmoid gate (from X_batch_input) splits cells into
                        neg/pos populations; squared error of soft-weighted means
                        vs the respective reference anchors.

    Gate is computed from X_batch_input (the pre-shift model input), not from X_shifted.
    This prevents the model from gaming the loss by moving cells across the bimodal
    threshold to escape the neg-pop penalty.

    Returns scalar mean loss over (samples × terms).
    """
    device = X_shifted.device
    eps = 1e-8
    total = torch.tensor(0.0, device=device)
    n_terms = 0

    uni_mask = ~is_bimodal
    bim_idx = is_bimodal.nonzero(as_tuple=True)[0]
    thr_bim = thresholds_scaled[bim_idx].unsqueeze(0)  # (1, n_bim)

    for s in torch.unique(sample_ids):
        mask = sample_ids == s
        if mask.sum() < 5:
            continue
        X_s = X_shifted[mask]           # (n_s, M) — has gradients
        X_s_in = X_batch_input[mask]    # (n_s, M) — no gradients (from leaf tensor)

        # Unimodal: align per-sample mean to neg_anchor
        if uni_mask.any():
            means_uni = X_s[:, uni_mask].mean(dim=0)   # (n_uni,)
            anchors_uni = neg_anchors_t[uni_mask]       # (n_uni,)
            total = total + ((means_uni - anchors_uni) ** 2).mean()
            n_terms += 1

        # Bimodal: align neg-pop mean to neg_anchor, pos-pop mean to pos_anchor
        if bim_idx.numel() > 0:
            x_bim = X_s[:, bim_idx]        # (n_s, n_bim) — has gradients
            x_bim_in = X_s_in[:, bim_idx]  # (n_s, n_bim) — no gradients, for gate
            w_pos = torch.sigmoid((x_bim_in - thr_bim) * sharpness)  # (n_s, n_bim)
            w_neg = 1.0 - w_pos

            neg_means = (w_neg * x_bim).sum(dim=0) / (w_neg.sum(dim=0) + eps)  # (n_bim,)
            pos_means = (w_pos * x_bim).sum(dim=0) / (w_pos.sum(dim=0) + eps)  # (n_bim,)

            total = total + ((neg_means - neg_anchors_t[bim_idx]) ** 2).mean()
            total = total + ((pos_means - pos_anchors_t[bim_idx]) ** 2).mean()
            n_terms += 2

    return total / max(n_terms, 1)


def _mask_bimodal_dims(
    X: torch.Tensor,
    thresholds: torch.Tensor,
    marker_is_bimodal: torch.Tensor,
) -> torch.Tensor:
    """Set bimodal marker dimensions to the threshold constant before MMD.

    Rationale: RobustScaler is fit on the full bimodal mixture, so batches with
    different ECAD+/CD45+ fractions (biological variation) land at different scaled
    positions — MMD then treats this as batch effect. Setting bimodal dimensions to
    constant makes them zero-variance in MMD distance computations. Unimodal markers
    drive all batch mixing. Gradient through bimodal dims is zero.
    """
    if not marker_is_bimodal.any():
        return X
    X_masked = X.clone()
    bim_idx = marker_is_bimodal.nonzero(as_tuple=True)[0]
    for m in bim_idx:
        X_masked[:, m] = thresholds[m]
    return X_masked


def mmd_rbf_loss(
    X: torch.Tensor,
    batch_ids: torch.Tensor,
    bandwidths: Tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0),
    n_samples: int = 256,
) -> torch.Tensor:
    """Multi-scale RBF MMD² between all batch pairs. Vectorized over bandwidths."""
    unique_batches = torch.unique(batch_ids)
    if unique_batches.size(0) < 2:
        return torch.tensor(0.0, device=X.device)
    batch_samples = {}
    for b in unique_batches:
        idx = (batch_ids == b).nonzero(as_tuple=True)[0]
        n = min(n_samples, idx.size(0))
        perm = torch.randperm(idx.size(0), device=X.device)[:n]
        batch_samples[b.item()] = X[idx[perm]]
    if len(batch_samples) < 2:
        return torch.tensor(0.0, device=X.device)
    gammas = 0.5 / torch.tensor(bandwidths, device=X.device, dtype=X.dtype).pow(2)
    total, n_pairs = torch.tensor(0.0, device=X.device), 0
    keys = list(batch_samples.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            Xi, Xj = batch_samples[keys[i]], batch_samples[keys[j]]
            g = gammas.view(-1, 1, 1)
            k_ii = torch.exp(-g * torch.cdist(Xi, Xi).pow(2).unsqueeze(0)).mean(dim=(-2, -1)).sum()
            k_jj = torch.exp(-g * torch.cdist(Xj, Xj).pow(2).unsqueeze(0)).mean(dim=(-2, -1)).sum()
            k_ij = torch.exp(-g * torch.cdist(Xi, Xj).pow(2).unsqueeze(0)).mean(dim=(-2, -1)).sum()
            total = total + (k_ii + k_jj - 2 * k_ij)
            n_pairs += 1
    return total / max(n_pairs, 1)


# ──────────────────────────────────────────────────────────────────────────────
# ResidualShiftModel
# ──────────────────────────────────────────────────────────────────────────────

class ResidualShiftModel(nn.Module):
    """Per-sample per-marker residual shifts applied to raw log1p scaled data.

    Bimodal markers get separate neg/pos shifts blended by sigmoid.
    Unimodal markers get a single shift.
    Zero-initialized → identity at training start.
    Thresholds are in scaled space (RobustScaler fit on log1p(X_raw)).
    """

    def __init__(
        self,
        n_samples: int,
        n_markers: int,
        marker_is_bimodal: np.ndarray,
        thresholds: np.ndarray,
        sharpness: float = 10.0,
        max_shift: float = 0.5,
    ):
        super().__init__()
        self.sharpness = sharpness
        self.max_shift = max_shift
        self.register_buffer("marker_is_bimodal",
                             torch.from_numpy(marker_is_bimodal.astype(bool)))
        self.register_buffer("thresholds",
                             torch.tensor(thresholds, dtype=torch.float32))
        self.shift_neg = nn.Embedding(n_samples, n_markers)
        self.shift_pos = nn.Embedding(n_samples, n_markers)
        nn.init.zeros_(self.shift_neg.weight)
        nn.init.zeros_(self.shift_pos.weight)

    def forward(self, X: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        """X: (N, M) scaled raw log1p input. Returns X + learned residual."""
        s_neg = self.shift_neg(sample_ids).clamp(-self.max_shift, self.max_shift)  # (N, M)
        if not self.marker_is_bimodal.any():
            return X + s_neg
        s_pos = self.shift_pos(sample_ids).clamp(-self.max_shift, self.max_shift)  # (N, M)
        w_pos = torch.sigmoid((X - self.thresholds.unsqueeze(0)) * self.sharpness)
        delta = torch.where(
            self.marker_is_bimodal.unsqueeze(0).expand_as(X),
            (1.0 - w_pos) * s_neg + w_pos * s_pos,
            s_neg,
        )
        return X + delta


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
    """Per-marker per-sample positive cell % — UniFORM GMM methodology.

    Raw:  per-sample LOCAL threshold (GMM fitted on each sample's raw cells).
    Norm: single GLOBAL threshold (GMM fitted on ALL normalized cells combined).
    Delta = pct_pos_norm(global thr) − pct_pos_raw(local thr).
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
        thr_global = _gmm_threshold(X_norm[:, m])
        for s in sorted(np.unique(sample_ids).tolist()):
            mask = sample_ids == s
            if mask.sum() < 10:
                continue
            thr_local = _gmm_threshold(X_raw[mask, m])
            pr = 100.0 * (X_raw[mask, m] > thr_local).mean()
            pn = 100.0 * (X_norm[mask, m] > thr_global).mean()
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
# Batch-balanced sampler
# ──────────────────────────────────────────────────────────────────────────────

class BatchBalancedSampler:
    """Equal-sized random draws from each batch per step."""

    def __init__(self, batch_codes: np.ndarray, cells_per_step: int = 16000, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.unique_batches = np.unique(batch_codes)
        self.batch_indices = {b: np.where(batch_codes == b)[0] for b in self.unique_batches}
        self.n_per_batch = max(1, cells_per_step // len(self.unique_batches))
        self._total = len(batch_codes)
        self._cells_per_step = cells_per_step

    def sample(self) -> np.ndarray:
        parts = []
        for b in self.unique_batches:
            idx = self.batch_indices[b]
            n = min(self.n_per_batch, len(idx))
            parts.append(self.rng.choice(idx, size=n, replace=False))
        return np.concatenate(parts)

    def __len__(self) -> int:
        return max(1, self._total // self._cells_per_step)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    adata: ad.AnnData,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    cells_per_step: int = 16000,
    device_str: str = "cpu",
    warmup_epochs: int = 5,
    mmd_ramp_start: int = 5,
    mmd_ramp_end: int = 20,
    w_ref: float = 1.0,
    w_mmd: float = 0.5,
    w_recon: float = 0.1,
    mmd_bandwidths: Tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0),
    mmd_samples: int = 256,
    mmd_mask_pospop: bool = True,
    sharpness: float = 10.0,
    max_shift: float = 0.5,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
    ref_sample_per_marker: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[ResidualShiftModel, RobustScaler, Dict[str, Optional[str]], Dict[str, List[float]]]:
    """Train SpaNCy-Shift-DL single-stage model.

    Trains ResidualShiftModel directly on raw log1p data with:
      L_ref  (w_ref=1.0)   — per-sample alignment to reference distribution anchors
      L_mmd  (w_mmd=0.5)   — 20D RBF MMD for multivariate batch mixing (ramped epoch 5→20)
      L_recon (w_recon=0.1) — Huber to identity (keeps shifts small)

    Returns (model, scaler, ref_sample_per_marker, history).
    Pass all four return values to normalize_adata().
    """
    device = torch.device(device_str)

    # ── Reference selection ────────────────────────────────────────────────────
    if ref_sample_per_marker is None:
        ref_sample_per_marker = find_best_sample_per_marker(adata)

    # ── Raw data ───────────────────────────────────────────────────────────────
    X_raw = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X,
                       dtype=np.float32)

    # ── Scale on raw log1p (not Stage 1 output) ───────────────────────────────
    X_raw_scaled, scaler = log1p_scale(X_raw)

    marker_names = list(adata.var_names)
    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = adata.obs[batch_col].astype("category").cat.codes.values.astype(np.int64)
    n_batches = int(batch_codes.max()) + 1

    sample_col = _get_col(adata, ["sample_id", "sample", "patient_id"], batch_col)
    sample_cats = adata.obs[sample_col].astype("category")
    sample_codes = sample_cats.cat.codes.values.astype(np.int64)
    sample_labels = adata.obs[sample_col].values  # string labels for anchor lookup
    n_samples = int(sample_codes.max()) + 1

    log.info("Setup: %d batches, %d samples, %d markers, %d cells",
             n_batches, n_samples, len(marker_names), adata.n_obs)

    # ── Bimodal detection on log1p(X_raw) — thresholds in log1p space ─────────
    X_log_raw = np.log1p(np.clip(X_raw, 0, None)).astype(np.float32)
    is_bimodal, thresholds_log1p = detect_bimodal_markers(
        X_log_raw, marker_names, batch_codes=batch_codes,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
    )

    # ── Convert thresholds to scaled space ────────────────────────────────────
    # scaler.center_ = per-marker median, scaler.scale_ = per-marker IQR (float64)
    thresholds_scaled = ((thresholds_log1p.astype(np.float64) - scaler.center_)
                         / scaler.scale_).astype(np.float32)

    # ── Precompute reference anchors (one-time numpy) ─────────────────────────
    neg_anchors, pos_anchors = compute_reference_anchors(
        X_raw_scaled, ref_sample_per_marker, sample_labels,
        marker_names, is_bimodal, thresholds_scaled, sharpness,
    )
    neg_anchors_t = torch.tensor(neg_anchors, dtype=torch.float32, device=device)
    pos_anchors_t = torch.tensor(pos_anchors, dtype=torch.float32, device=device)
    thresholds_t = torch.tensor(thresholds_scaled, dtype=torch.float32, device=device)
    is_bimodal_t = torch.tensor(is_bimodal, dtype=torch.bool, device=device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ResidualShiftModel(
        n_samples, len(marker_names), is_bimodal, thresholds_scaled, sharpness, max_shift
    ).to(device)
    log.info("ResidualShiftModel: %d parameters",
             sum(p.numel() for p in model.parameters()))

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    sampler = BatchBalancedSampler(batch_codes, cells_per_step=cells_per_step)
    steps_per_epoch = len(sampler)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                              milestones=[warmup_epochs])

    # ── Data tensors (raw scaled) ──────────────────────────────────────────────
    X_t = torch.tensor(X_raw_scaled, dtype=torch.float32)
    batch_t = torch.tensor(batch_codes, dtype=torch.long)
    sample_t = torch.tensor(sample_codes, dtype=torch.long)
    huber = nn.HuberLoss(delta=1.0)

    history: Dict[str, List[float]] = {
        "loss": [], "ref": [], "mmd": [], "recon": [], "lr": [], "mmd_weight": [],
    }

    log.info(
        "Training: %d epochs × %d steps  "
        "w_ref=%.2f  w_mmd=%.2f  w_recon=%.2f  "
        "mmd_ramp=%d→%d  sharpness=%.1f  max_shift=%.2f  "
        "mmd_bimodal_excl=%s  device=%s",
        n_epochs, steps_per_epoch,
        w_ref, w_mmd, w_recon, mmd_ramp_start, mmd_ramp_end,
        sharpness, max_shift, mmd_mask_pospop, device_str,
    )

    for epoch in range(n_epochs):
        e_loss = e_ref = e_mmd = e_recon = 0.0

        # MMD weight ramp
        if epoch < mmd_ramp_start:
            mmd_w = 0.0
        elif epoch < mmd_ramp_end:
            mmd_w = w_mmd * (epoch - mmd_ramp_start) / max(mmd_ramp_end - mmd_ramp_start, 1)
        else:
            mmd_w = w_mmd

        for step in range(steps_per_epoch):
            idx = sampler.sample()
            if len(idx) < 10:
                continue
            X_batch = X_t[idx].to(device)   # raw scaled — no grad (leaf tensor from numpy)
            b_ids = batch_t[idx].to(device)
            s_ids = sample_t[idx].to(device)

            X_shifted = model(X_batch, s_ids)

            # L_ref: pull each sample's distribution toward reference anchors
            loss_ref = reference_alignment_loss(
                X_shifted,
                X_batch,              # pre-shift input — used for bimodal gating only
                s_ids,
                neg_anchors_t, pos_anchors_t,
                is_bimodal_t, thresholds_t,
                sharpness,
            )

            # L_recon: Huber to identity (keep shifts small)
            loss_recon = huber(X_shifted, X_batch)

            # L_mmd: 20D MMD (bimodal dims masked to prevent biology → batch confusion)
            if mmd_w > 0:
                X_mmd = (
                    _mask_bimodal_dims(X_shifted, model.thresholds, model.marker_is_bimodal)
                    if mmd_mask_pospop else X_shifted
                )
                loss_mmd = mmd_rbf_loss(X_mmd, b_ids,
                                        bandwidths=mmd_bandwidths, n_samples=mmd_samples)
            else:
                loss_mmd = torch.tensor(0.0, device=device)

            loss = w_ref * loss_ref + mmd_w * loss_mmd + w_recon * loss_recon

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            e_loss += loss.item()
            e_ref += loss_ref.item()
            e_mmd += loss_mmd.item()
            e_recon += loss_recon.item()

            if steps_per_epoch > 20 and (step + 1) % max(1, steps_per_epoch // 5) == 0:
                log.info("  E%3d step %d/%d  loss=%.4f  ref=%.4f  mmd=%.4f  recon=%.4f",
                         epoch + 1, step + 1, steps_per_epoch,
                         loss.item(), loss_ref.item(), loss_mmd.item(), loss_recon.item())

        scheduler.step()
        s = max(steps_per_epoch, 1)
        history["loss"].append(e_loss / s)
        history["ref"].append(e_ref / s)
        history["mmd"].append(e_mmd / s)
        history["recon"].append(e_recon / s)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["mmd_weight"].append(mmd_w)

        log.info(
            "Epoch %3d/%d  loss=%.4f  ref=%.4f  mmd=%.4f  recon=%.4f  mmd_w=%.2f  lr=%.2e",
            epoch + 1, n_epochs,
            history["loss"][-1], history["ref"][-1], history["mmd"][-1], history["recon"][-1],
            mmd_w, optimizer.param_groups[0]["lr"],
        )

    log.info("Training complete. Final loss=%.4f", history["loss"][-1])
    return model, scaler, ref_sample_per_marker, history


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def normalize_adata(
    adata: ad.AnnData,
    model: ResidualShiftModel,
    scaler: RobustScaler,
    ref_sample_per_marker: Dict[str, Optional[str]],
    device_str: str = "cpu",
    inference_batch_size: int = 50000,
    layer_name: str = "normalized",
) -> ad.AnnData:
    """Single-stage inference: raw log1p → scale → ResidualShiftModel → inverse scale → expm1.

    ref_sample_per_marker is kept in the signature for API compatibility with train() return
    values but is not used during inference (alignment is encoded in trained model weights).
    """
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    X_raw = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X,
                       dtype=np.float32)
    X_scaled = scaler.transform(np.log1p(np.clip(X_raw, 0, None))).astype(np.float32)

    sample_col = _get_col(adata, ["sample_id", "sample", "patient_id"], "batch_id")
    sample_codes = adata.obs[sample_col].astype("category").cat.codes.values.astype(np.int64)

    n_cells = adata.n_obs
    X_norm_scaled = np.zeros((n_cells, adata.n_vars), dtype=np.float32)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    s_t = torch.tensor(sample_codes, dtype=torch.long)

    log.info("Inference: ResidualShiftModel on %d cells "
             "(raw log1p → scale → model → inverse scale → expm1)...", n_cells)
    n_chunks = max(1, math.ceil(n_cells / inference_batch_size))
    for ci in range(n_chunks):
        s_, e_ = ci * inference_batch_size, min((ci + 1) * inference_batch_size, n_cells)
        X_out = model(X_t[s_:e_].to(device), s_t[s_:e_].to(device))
        X_norm_scaled[s_:e_] = X_out.cpu().numpy()
        if (ci + 1) % 10 == 0 or ci == 0:
            log.info("  chunk %d/%d", ci + 1, n_chunks)

    X_norm_log = scaler.inverse_transform(X_norm_scaled)
    X_final = np.clip(np.expm1(X_norm_log), 0, None).astype(np.float32)

    adata_out = adata.copy()
    adata_out.layers[layer_name] = X_final

    log.info("Done. Layer '%s' written. min=%.4f  max=%.4f  mean=%.4f",
             layer_name, X_final.min(), X_final.max(), X_final.mean())
    return adata_out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpaNCy-Shift-DL: single-stage DL CyCIF batch correction"
    )
    parser.add_argument("--input", required=True, help="Input .h5ad")
    parser.add_argument("--output", required=True, help="Output .h5ad")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cells_per_step", type=int, default=16000)
    parser.add_argument("--w_ref", type=float, default=1.0)
    parser.add_argument("--w_mmd", type=float, default=0.5)
    parser.add_argument("--w_recon", type=float, default=0.1)
    parser.add_argument("--mmd_ramp_start", type=int, default=5)
    parser.add_argument("--mmd_ramp_end", type=int, default=20)
    parser.add_argument("--mmd_samples", type=int, default=256)
    parser.add_argument("--no_mmd_mask_pospop", action="store_true",
                        help="Disable bimodal dim masking in MMD (not recommended)")
    parser.add_argument("--sharpness", type=float, default=10.0,
                        help="Sigmoid sharpness for bimodal neg/pos blending")
    parser.add_argument("--max_shift", type=float, default=0.5,
                        help="Max shift magnitude per sample per marker (RobustScaler units)")
    parser.add_argument("--bimodal_min_batch_frac", type=float, default=0.5)
    parser.add_argument("--layer_name", default="normalized")
    args = parser.parse_args()

    adata = load_adata(args.input)
    model, scaler, ref_sample_per_marker, history = train(
        adata,
        n_epochs=args.epochs,
        lr=args.lr,
        cells_per_step=args.cells_per_step,
        device_str=args.device,
        w_ref=args.w_ref,
        w_mmd=args.w_mmd,
        w_recon=args.w_recon,
        mmd_ramp_start=args.mmd_ramp_start,
        mmd_ramp_end=args.mmd_ramp_end,
        mmd_samples=args.mmd_samples,
        mmd_mask_pospop=not args.no_mmd_mask_pospop,
        sharpness=args.sharpness,
        max_shift=args.max_shift,
        bimodal_min_batch_frac=args.bimodal_min_batch_frac,
    )
    adata_norm = normalize_adata(
        adata, model, scaler, ref_sample_per_marker,
        device_str=args.device, layer_name=args.layer_name,
    )
    adata_norm.write_h5ad(args.output)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
