#!/usr/bin/env python
"""
SpaNCy-Shift: Adaptive Per-Sample Shift Correction for CyCIF Batch Normalization.

Combines CycleDegradationModel (batch/sample/cycle-aware affine correction) with
AdaptiveShiftModel (learned per-sample bimodal/unimodal shifts) and 20D MMD loss
for kBET improvement.

Architecture:
    X_raw → log1p → RobustScaler → X_scaled
      → CycleDegradationModel(batch, sample, cycles) → X_affine
      → AdaptiveShiftModel(sample, marker_bimodality) → X_shifted
      → inverse_scale → expm1 → X_normalized

No normalizing flow. Shifts preserve per-marker variance by construction.
Bimodal detection is fully adaptive via _find_peaks() — no hard-coded thresholds.
"""

import argparse
import json
import logging
import math
import sys
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

log = logging.getLogger("spancy_shift")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_handler)
    log.propagate = False

# ──────────────────────────────────────────────────────────────────────────────
# Default cycle assignment for PRAD-CyCIF 20-marker panel
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CYCLE_CONFIG: Dict[int, List[str]] = {
    0: ["DAPI", "DAPI_R1"],
    1: ["EPCAM", "CD56", "CD45"],
    2: ["aSMA", "ChromA", "CK14", "Ki67"],
    3: ["GZMB", "ECAD", "PD1", "CD31"],
    4: ["CD45RA", "HLADRB1", "CD3", "p53"],
    5: ["FOXA1", "CDX2", "CD20", "NOTCH1"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────


def load_adata(path: str) -> ad.AnnData:
    """Load AnnData and validate expected fields."""
    log.info("Loading %s", path)
    adata = ad.read_h5ad(path)
    log.info("Loaded %d cells x %d markers", adata.n_obs, adata.n_vars)

    batch_col = None
    for col in ("batch", "batch_id", "Batch", "BatchID", "batch_ID"):
        if col in adata.obs.columns:
            batch_col = col
            break
    if batch_col is None:
        raise ValueError(
            "adata.obs must contain a batch column "
            "(looked for: batch, batch_id, Batch, BatchID)"
        )
    if batch_col != "batch":
        log.info("Using '%s' as batch column", batch_col)
        adata.obs["batch"] = adata.obs[batch_col]

    if all(v.isdigit() for v in adata.var_names.astype(str)):
        for col in ("marker_name", "marker", "Marker", "gene", "Gene", "protein"):
            if col in adata.var.columns:
                log.info("Setting var_names from var['%s']", col)
                adata.var_names = adata.var[col].values
                break
        else:
            log.warning(
                "var_names are numeric and no marker_name column found in .var; "
                "cycle assignment may not work correctly"
            )

    has_spatial = "spatial" in adata.obsm
    has_xy = {"x", "y"}.issubset(adata.obs.columns)
    if not has_spatial and not has_xy:
        raise ValueError(
            "Need spatial coordinates: adata.obsm['spatial'] or adata.obs[['x','y']]"
        )
    return adata


def get_spatial_coords(adata: ad.AnnData) -> np.ndarray:
    """Extract (x, y) coordinates as (N, 2) float64 array."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])[:, :2]
    else:
        coords = adata.obs[["x", "y"]].values
    return coords.astype(np.float64)


def get_scene_ids(adata: ad.AnnData) -> np.ndarray:
    """Per-cell scene IDs for spatial graph construction."""
    for col in ("scene_id", "scene", "Scene", "SCENE", "sample_id", "sample", "image_id"):
        if col in adata.obs.columns:
            log.info("Using '%s' as scene column for spatial graph", col)
            return adata.obs[col].astype("category").cat.codes.values
    log.warning("No scene column found; using 'batch' for spatial graph")
    return adata.obs["batch"].astype("category").cat.codes.values


def assign_marker_cycles(
    marker_names: List[str],
    cycle_config: Dict[int, List[str]],
) -> np.ndarray:
    """Return (M,) array mapping each marker index to its imaging cycle."""
    marker_to_cycle = {}
    for cyc, markers in cycle_config.items():
        for m in markers:
            marker_to_cycle[m] = int(cyc)

    cycles = np.zeros(len(marker_names), dtype=np.int64)
    for i, name in enumerate(marker_names):
        if name in marker_to_cycle:
            cycles[i] = marker_to_cycle[name]
        else:
            log.warning("Marker '%s' not in cycle config; defaulting to cycle 0", name)
    return cycles


def log1p_scale(X: np.ndarray) -> Tuple[np.ndarray, RobustScaler]:
    """Apply log1p then per-marker RobustScaler (median/IQR)."""
    X_log = np.log1p(np.clip(X, 0, None))
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_log)
    return X_scaled.astype(np.float32), scaler


# ──────────────────────────────────────────────────────────────────────────────
# Peak detection (shared with post-hoc alignment)
# ──────────────────────────────────────────────────────────────────────────────


def _find_peaks(
    counts: np.ndarray,
    bins: np.ndarray,
    min_prominence_frac: float = 0.02,
) -> List[float]:
    """Find prominent peaks in a histogram, ordered left to right."""
    counts_smooth = gaussian_filter1d(counts.astype(np.float64), sigma=2)
    max_height = counts_smooth.max()
    if max_height < 1:
        return [(bins[0] + bins[-1]) / 2]

    peaks, _ = find_peaks(counts_smooth, prominence=max_height * min_prominence_frac)
    if len(peaks) == 0:
        idx = int(counts_smooth.argmax())
        return [(bins[idx] + bins[idx + 1]) / 2]

    return [(bins[p] + bins[p + 1]) / 2 for p in peaks]


def _safe_piecewise_transform(
    vals: np.ndarray,
    src_peaks: List[float],
    dst_peaks: List[float],
) -> np.ndarray:
    """Piecewise linear transform with clamped slope=1 extrapolation."""
    s0, s1 = src_peaks[0], src_peaks[1]
    d0, d1 = dst_peaks[0], dst_peaks[1]

    out = vals.copy()
    out[vals <= s0] = vals[vals <= s0] + (d0 - s0)

    src_span = s1 - s0
    dst_span = d1 - d0
    mask_mid = (vals > s0) & (vals <= s1)
    if src_span > 1e-8:
        out[mask_mid] = d0 + (vals[mask_mid] - s0) * (dst_span / src_span)
    else:
        out[mask_mid] = vals[mask_mid] + (d0 - s0)

    out[vals > s1] = vals[vals > s1] + (d1 - s1)
    return out


def sample_mode_align(
    X: np.ndarray,
    sample_ids: np.ndarray,
    n_bins: int = 200,
    marker_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Post-hoc per-marker per-sample peak alignment (optional)."""
    unique_samples = np.unique(sample_ids)
    X_aligned = X.copy()

    for m in range(X.shape[1]):
        col = X[:, m]
        lo, hi = np.percentile(col, [1, 99])
        if hi - lo < 1e-6:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        counts_global, _ = np.histogram(col, bins=bins)
        global_peaks = _find_peaks(counts_global, bins)

        n_piecewise = n_shift = 0
        for s in unique_samples:
            mask = sample_ids == s
            vals = col[mask]
            if len(vals) < 50:
                continue
            counts_s, _ = np.histogram(vals, bins=bins)
            sample_peaks = _find_peaks(counts_s, bins)
            if len(global_peaks) >= 2 and len(sample_peaks) >= 2:
                X_aligned[mask, m] = _safe_piecewise_transform(
                    vals, sample_peaks[:2], global_peaks[:2]
                )
                n_piecewise += 1
            else:
                X_aligned[mask, m] = vals + (global_peaks[0] - sample_peaks[0])
                n_shift += 1

        mname = marker_names[m] if marker_names and m < len(marker_names) else str(m)
        n_total = n_piecewise + n_shift
        log.info(
            "  %-10s: global_peaks=%d, piecewise=%d/%d, shift=%d/%d",
            mname, len(global_peaks), n_piecewise, n_total, n_shift, n_total,
        )

    return X_aligned


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────


def _otsu_threshold(values: np.ndarray, n_bins: int = 200) -> float:
    """Otsu's threshold for a 1D array."""
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
    sum_bg, weight_bg = 0.0, 0

    for i in range(len(counts)):
        weight_bg += counts[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += counts[i] * centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = centers[i]

    return float(best_thresh)


def positive_population_table(
    adata: ad.AnnData,
    raw_layer: Optional[str] = None,
    norm_layer: str = "normalized",
    sample_col: str = "sample_id",
    marker_names: Optional[List[str]] = None,
    log_transform: bool = True,
) -> pd.DataFrame:
    """Per-marker per-sample positive cell % comparing raw vs normalized.

    Uses Otsu's method per marker. Delta = pct_pos_norm - pct_pos_raw.
    Target: abs(delta) < 5% per marker.
    """
    if marker_names is None:
        marker_names = list(adata.var_names)

    X_raw = np.asarray(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    if raw_layer is not None:
        X_raw = np.asarray(adata.layers[raw_layer])
    X_norm = np.asarray(adata.layers[norm_layer])

    if log_transform:
        X_raw_t = np.log1p(np.clip(X_raw, 0, None))
        X_norm_t = np.log1p(np.clip(X_norm, 0, None))
    else:
        X_raw_t, X_norm_t = X_raw, X_norm

    s_col = None
    for col in (sample_col, "sample_id", "sample", "Sample", "patient_id"):
        if col in adata.obs.columns:
            s_col = col
            break
    if s_col is None:
        s_col = "batch"
    sample_ids = adata.obs[s_col].values
    unique_samples = sorted(np.unique(sample_ids).tolist())

    rows = []
    for m, mname in enumerate(marker_names):
        threshold = _otsu_threshold(X_raw_t[:, m])
        for s in unique_samples:
            mask = sample_ids == s
            n_cells = mask.sum()
            if n_cells < 10:
                continue
            rows.append({
                "marker": mname,
                "sample": s,
                "pct_pos_raw": round(100.0 * (X_raw_t[mask, m] > threshold).sum() / n_cells, 2),
                "pct_pos_norm": round(100.0 * (X_norm_t[mask, m] > threshold).sum() / n_cells, 2),
                "delta": round(100.0 * ((X_norm_t[mask, m] > threshold).sum() - (X_raw_t[mask, m] > threshold).sum()) / n_cells, 2),
            })

    return pd.DataFrame(rows)


def per_marker_batch_r2(
    X: np.ndarray,
    batch_labels: np.ndarray,
    marker_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Adjusted R² for each marker regressed on batch (one-hot encoding).

    Higher adj-R² = more batch effect remaining. Target after correction: < 0.05.
    """
    enc = OneHotEncoder(sparse_output=False, drop="first")
    B = enc.fit_transform(batch_labels.reshape(-1, 1))
    n, p = X.shape
    k = B.shape[1]

    rows = []
    for j in range(p):
        y = X[:, j]
        reg = LinearRegression().fit(B, y)
        ss_res = np.sum((y - reg.predict(B)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            adj_r2 = 0.0
        else:
            r2 = 1 - ss_res / ss_tot
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        name = marker_names[j] if marker_names is not None else f"Marker_{j}"
        rows.append({"marker": name, "adj_r2": float(adj_r2)})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive bimodal detection
# ──────────────────────────────────────────────────────────────────────────────


def detect_bimodal_markers(
    X_scaled: np.ndarray,
    marker_names: List[str],
    batch_codes: Optional[np.ndarray] = None,
    n_bins: int = 150,
    min_prominence_frac: float = 0.02,
    bimodal_min_batch_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify markers as bimodal or unimodal using per-batch voting.

    For each marker, runs _find_peaks() separately on each batch's histogram.
    A marker is bimodal only if >= bimodal_min_batch_frac of batches independently
    show >= 2 peaks. This prevents batch-separated unimodal distributions (e.g.
    ChromA where different batches sit at different positions) from being falsely
    classified as bimodal from the pooled global histogram.

    If batch_codes is None, falls back to single global histogram.

    Returns:
        marker_is_bimodal: (M,) bool
        thresholds: (M,) float — median midpoint across bimodal batches, 0.0 for unimodal
    """
    n_markers = X_scaled.shape[1]
    marker_is_bimodal = np.zeros(n_markers, dtype=bool)
    thresholds = np.zeros(n_markers, dtype=np.float32)

    if batch_codes is not None:
        unique_batches = np.unique(batch_codes)
        log.info(
            "Detecting bimodal markers via per-batch voting "
            "(n_batches=%d, n_bins=%d, min_prominence=%.3f, min_batch_frac=%.2f)...",
            len(unique_batches), n_bins, min_prominence_frac, bimodal_min_batch_frac,
        )
    else:
        unique_batches = None
        log.info("Detecting bimodal markers (n_bins=%d, min_prominence=%.3f)...", n_bins, min_prominence_frac)

    for m in range(n_markers):
        col = X_scaled[:, m]
        mname = marker_names[m] if m < len(marker_names) else str(m)
        lo, hi = np.percentile(col, [1, 99])
        if hi - lo < 1e-6:
            log.info("  %-10s: unimodal (flat distribution)", mname)
            continue

        if unique_batches is not None:
            bimodal_votes = 0
            midpoints = []
            n_valid = 0
            for b in unique_batches:
                mask = batch_codes == b
                if mask.sum() < 30:
                    continue
                n_valid += 1
                col_b = col[mask]
                lo_b, hi_b = np.percentile(col_b, [1, 99])
                if hi_b - lo_b < 1e-6:
                    continue
                bins_b = np.linspace(lo_b, hi_b, n_bins + 1)
                counts_b, _ = np.histogram(col_b, bins=bins_b)
                peaks_b = _find_peaks(counts_b, bins_b, min_prominence_frac=min_prominence_frac)
                if len(peaks_b) >= 2:
                    bimodal_votes += 1
                    midpoints.append((peaks_b[0] + peaks_b[1]) / 2.0)

            frac = bimodal_votes / max(n_valid, 1)
            if frac >= bimodal_min_batch_frac and len(midpoints) > 0:
                marker_is_bimodal[m] = True
                thresholds[m] = float(np.median(midpoints))
                log.info("  %-10s: BIMODAL  (%d/%d batches bimodal, thresh=%.3f)",
                         mname, bimodal_votes, n_valid, thresholds[m])
            else:
                log.info("  %-10s: unimodal (%d/%d batches bimodal)", mname, bimodal_votes, n_valid)
        else:
            bins = np.linspace(lo, hi, n_bins + 1)
            counts, _ = np.histogram(col, bins=bins)
            peaks = _find_peaks(counts, bins, min_prominence_frac=min_prominence_frac)
            if len(peaks) >= 2:
                marker_is_bimodal[m] = True
                thresholds[m] = float((peaks[0] + peaks[1]) / 2.0)
                log.info("  %-10s: BIMODAL  (neg=%.3f, pos=%.3f, thresh=%.3f)",
                         mname, peaks[0], peaks[1], thresholds[m])
            else:
                log.info("  %-10s: unimodal (peak=%.3f)", mname, peaks[0] if len(peaks) else lo)

    n_bim = int(marker_is_bimodal.sum())
    log.info("Result: %d bimodal, %d unimodal markers", n_bim, n_markers - n_bim)
    return marker_is_bimodal, thresholds


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────


def mmd_rbf_loss(
    X: torch.Tensor,
    batch_ids: torch.Tensor,
    bandwidths: Tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0),
    n_samples: int = 256,
) -> torch.Tensor:
    """Multi-scale RBF kernel MMD² between all batch pairs.

    Vectorized over bandwidths — no Python loop per bandwidth.
    """
    unique_batches = torch.unique(batch_ids)
    if unique_batches.size(0) < 2:
        return torch.tensor(0.0, device=X.device)

    batch_samples = {}
    for b in unique_batches:
        b_mask = (batch_ids == b).nonzero(as_tuple=True)[0]
        n_avail = b_mask.size(0)
        if n_avail == 0:
            continue
        n_take = min(n_samples, n_avail)
        perm = torch.randperm(n_avail, device=X.device)[:n_take]
        batch_samples[b.item()] = X[b_mask[perm]]

    if len(batch_samples) < 2:
        return torch.tensor(0.0, device=X.device)

    gammas = 0.5 / torch.tensor(bandwidths, device=X.device, dtype=X.dtype).pow(2)  # (n_bw,)

    total_mmd = torch.tensor(0.0, device=X.device)
    n_pairs = 0

    keys = list(batch_samples.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            X_i = batch_samples[keys[i]]
            X_j = batch_samples[keys[j]]

            d_ii = torch.cdist(X_i, X_i).pow(2)
            d_jj = torch.cdist(X_j, X_j).pow(2)
            d_ij = torch.cdist(X_i, X_j).pow(2)

            # (n_bw, ni, ni/nj) → mean over cells → sum over bandwidths
            g = gammas.view(-1, 1, 1)
            k_ii = torch.exp(-g * d_ii.unsqueeze(0)).mean(dim=(-2, -1)).sum()
            k_jj = torch.exp(-g * d_jj.unsqueeze(0)).mean(dim=(-2, -1)).sum()
            k_ij = torch.exp(-g * d_ij.unsqueeze(0)).mean(dim=(-2, -1)).sum()

            total_mmd = total_mmd + (k_ii + k_jj - 2 * k_ij)
            n_pairs += 1

    return total_mmd / max(n_pairs, 1)


def quantile_alignment_loss(
    X: torch.Tensor,
    sample_ids: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.1, 0.25),
) -> torch.Tensor:
    """Align lower quantiles (10th/25th) across samples.

    These quantiles fall safely within the negative population for all CyCIF
    markers. Penalizes per-sample deviation from the global quantile.
    """
    unique_samples = torch.unique(sample_ids)
    if unique_samples.size(0) < 2:
        return torch.tensor(0.0, device=X.device)

    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=X.device)
    global_q = torch.quantile(X, q_tensor, dim=0)
    q_spread = (global_q[-1] - global_q[0]).clamp(min=1e-6)

    loss = torch.tensor(0.0, device=X.device)
    n_counted = 0

    for s in unique_samples:
        mask = sample_ids == s
        if mask.sum() < 50:
            continue
        sample_q = torch.quantile(X[mask], q_tensor, dim=0)
        loss = loss + (((sample_q - global_q) / q_spread) ** 2).mean()
        n_counted += 1

    return loss / max(n_counted, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────


class CycleDegradationModel(nn.Module):
    """Per-batch, per-sample, per-cycle affine correction (gamma/beta).

    Each marker's correction is conditioned on batch (32d), sample (16d),
    and imaging cycle (16d) embeddings fed through a shared MLP.
    """

    def __init__(self, n_batches: int, n_samples: int, n_cycles: int, n_markers: int):
        super().__init__()
        self.batch_embed = nn.Embedding(n_batches, 32)
        self.sample_embed = nn.Embedding(n_samples, 16)
        self.cycle_embed = nn.Embedding(n_cycles, 16)
        self.n_markers = n_markers

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_emb = self.batch_embed(batch_ids)    # (N, 32)
        s_emb = self.sample_embed(sample_ids)  # (N, 16)
        c_emb = self.cycle_embed(marker_cycles)  # (M, 16)

        N = b_emb.size(0)
        M = self.n_markers
        bs = torch.cat([b_emb, s_emb], dim=-1).unsqueeze(1).expand(N, M, 48)
        c = c_emb.unsqueeze(0).expand(N, M, 16)
        h = torch.cat([bs, c], dim=-1)  # (N, M, 64)

        out = self.mlp(h)
        gamma = F.softplus(out[..., 0]) + 0.1  # (N, M)
        beta = out[..., 1]                       # (N, M)
        return gamma, beta

    def correct(
        self,
        X: torch.Tensor,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self(batch_ids, sample_ids, marker_cycles)
        return (X - beta) / gamma


class AdaptiveShiftModel(nn.Module):
    """Per-sample per-marker shifts with adaptive bimodal/unimodal handling.

    Bimodal markers (detected via _find_peaks) get separate neg/pos peak shifts
    blended smoothly via sigmoid. Unimodal markers get a single shift.
    All shifts initialized to zero → identity at training start.
    No hard-coded thresholds — bimodality determined by peak prominence.
    """

    def __init__(
        self,
        n_samples: int,
        n_markers: int,
        marker_is_bimodal: np.ndarray,
        thresholds: np.ndarray,
        sharpness: float = 5.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_markers = n_markers
        self.sharpness = sharpness

        self.register_buffer(
            "marker_is_bimodal", torch.from_numpy(marker_is_bimodal.astype(bool))
        )
        self.register_buffer(
            "thresholds", torch.tensor(thresholds, dtype=torch.float32)
        )

        # shift_neg: unimodal shift (or neg-peak shift for bimodal)
        # shift_pos: pos-peak shift for bimodal markers (ignored for unimodal)
        self.shift_neg = nn.Embedding(n_samples, n_markers)
        self.shift_pos = nn.Embedding(n_samples, n_markers)
        nn.init.zeros_(self.shift_neg.weight)
        nn.init.zeros_(self.shift_pos.weight)

    def forward(self, X_affine: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_affine: (N, M) affine-corrected expression in scaled space
            sample_ids: (N,) integer sample indices
        Returns:
            X_shifted: (N, M) shift-corrected expression
        """
        s_neg = self.shift_neg(sample_ids)  # (N, M)

        bimodal = self.marker_is_bimodal  # (M,) bool
        if not bimodal.any():
            return X_affine + s_neg

        s_pos = self.shift_pos(sample_ids)  # (N, M)
        thresh = self.thresholds.unsqueeze(0)  # (1, M)
        w_pos = torch.sigmoid((X_affine - thresh) * self.sharpness)  # (N, M)

        delta_bimodal = (1.0 - w_pos) * s_neg + w_pos * s_pos
        # Bimodal markers use blended delta; unimodal use shift_neg
        delta = torch.where(
            bimodal.unsqueeze(0).expand_as(X_affine),
            delta_bimodal,
            s_neg,
        )
        return X_affine + delta


class SpaNCyShift(nn.Module):
    """SpaNCy-Shift: CycleDegradation + AdaptiveShift.

    The CycleDegradationModel applies batch/sample/cycle-aware affine correction.
    AdaptiveShiftModel then applies per-sample bimodal/unimodal shifts to align
    remaining distributional differences while preserving variance by construction.
    20D MMD loss during training drives kBET improvement.
    """

    def __init__(
        self,
        n_markers: int,
        n_batches: int,
        n_samples: int,
        n_cycles: int,
        marker_is_bimodal: np.ndarray,
        thresholds: np.ndarray,
        sharpness: float = 5.0,
    ):
        super().__init__()
        self.cycle_model = CycleDegradationModel(n_batches, n_samples, n_cycles, n_markers)
        self.shift_model = AdaptiveShiftModel(
            n_samples, n_markers, marker_is_bimodal, thresholds, sharpness
        )

    def forward(
        self,
        X: torch.Tensor,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gamma, beta = self.cycle_model(batch_ids, sample_ids, marker_cycles)
        X_affine = (X - beta) / gamma
        X_shifted = self.shift_model(X_affine, sample_ids)
        return {
            "X_affine": X_affine,
            "X_shifted": X_shifted,
            "gamma": gamma,
            "beta": beta,
        }

    @torch.no_grad()
    def normalize(
        self,
        X: torch.Tensor,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Inference: affine-only (alpha=0) or affine + shift blend (alpha>0)."""
        self.eval()
        gamma, beta = self.cycle_model(batch_ids, sample_ids, marker_cycles)
        X_affine = (X - beta) / gamma
        if alpha == 0.0:
            return X_affine
        X_shifted = self.shift_model(X_affine, sample_ids)
        return X_affine + alpha * (X_shifted - X_affine)


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────


class SpatialClusterSampler:
    """Batch-balanced mini-batch sampler using spatial k-means clusters."""

    def __init__(
        self,
        coords: np.ndarray,
        batch_ids: np.ndarray,
        cluster_size: int = 500,
        cells_per_step: int = 6000,
        seed: int = 42,
    ):
        self.batch_ids = batch_ids
        self.cluster_size = cluster_size
        self.cells_per_step = cells_per_step
        self.rng = np.random.RandomState(seed)

        n_clusters = max(1, len(coords) // cluster_size)
        log.info("Fitting %d spatial clusters...", n_clusters)
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=4096)
        self.cluster_labels = km.fit_predict(coords)
        self.unique_batches = np.unique(batch_ids)

    def sample(self) -> np.ndarray:
        indices = []
        target_per_batch = self.cells_per_step // max(len(self.unique_batches), 1)
        for b in self.unique_batches:
            b_mask = self.batch_ids == b
            b_indices = np.where(b_mask)[0]
            b_clusters = self.cluster_labels[b_mask]
            unique_cl = np.unique(b_clusters)
            n_cl = max(1, target_per_batch // self.cluster_size)
            chosen = self.rng.choice(unique_cl, size=min(n_cl, len(unique_cl)), replace=False)
            for cl in chosen:
                cl_idx = b_indices[b_clusters == cl]
                if len(cl_idx) > self.cluster_size:
                    cl_idx = self.rng.choice(cl_idx, size=self.cluster_size, replace=False)
                indices.append(cl_idx)
        return np.concatenate(indices) if indices else np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return max(1, len(self.batch_ids) // self.cells_per_step)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────


def train(
    adata: ad.AnnData,
    cycle_config: Dict[int, List[str]] = None,
    n_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    cluster_size: int = 500,
    cells_per_step: int = 16000,
    device_str: str = "cpu",
    warmup_epochs: int = 3,
    mmd_ramp_start: int = 3,
    mmd_ramp_end: int = 7,
    w_mmd: float = 0.5,
    w_recon: float = 1.0,
    w_identity: float = 0.5,
    w_align: float = 0.5,
    mmd_bandwidths: Tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0),
    mmd_samples: int = 256,
    sharpness: float = 5.0,
    bimodal_prominence: float = 0.02,
    bimodal_min_batch_frac: float = 0.5,
) -> Tuple["SpaNCyShift", RobustScaler, np.ndarray, Dict[str, List[float]]]:
    """Train SpaNCy-Shift model.

    Returns (model, scaler, marker_cycles, history).

    Losses:
      L_recon:    Huber(X_affine, X_scaled) — anchor CycleDegradationModel to input (prevents gamma/beta collapse)
      L_identity: Huber(X_shifted, X_affine) — keep shifts small relative to affine baseline
      L_align:    Lower-quantile alignment across samples (10th/25th percentile)
      L_mmd:      20D MMD across batches for kBET (warmed up epoch mmd_ramp_start→end)
    """
    if cycle_config is None:
        cycle_config = DEFAULT_CYCLE_CONFIG

    device = torch.device(device_str)

    # Preprocessing
    X_raw = np.asarray(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    X_scaled, scaler = log1p_scale(X_raw)

    marker_names = list(adata.var_names)
    marker_cycles = assign_marker_cycles(marker_names, cycle_config)
    marker_cycles_t = torch.tensor(marker_cycles, dtype=torch.long, device=device)
    n_markers = len(marker_names)
    n_cycles = int(marker_cycles.max()) + 1

    # Batch encoding
    batch_cats = adata.obs["batch"].astype("category")
    batch_codes = batch_cats.cat.codes.values.astype(np.int64)
    n_batches = int(batch_codes.max()) + 1

    # Sample encoding
    sample_col = None
    for col in ("sample_id", "sample", "Sample", "patient_id", "patient"):
        if col in adata.obs.columns:
            sample_col = col
            break
    if sample_col is None:
        sample_col = "batch"
        log.warning("No sample column found; using 'batch' as sample proxy")
    sample_cats = adata.obs[sample_col].astype("category")
    sample_codes = sample_cats.cat.codes.values.astype(np.int64)
    n_samples = int(sample_codes.max()) + 1

    log.info(
        "Detected %d batches, %d samples (col='%s'), %d cycles, %d markers",
        n_batches, n_samples, sample_col, n_cycles, n_markers,
    )

    # Adaptive bimodal detection on scaled data (per-batch voting)
    marker_is_bimodal, thresholds = detect_bimodal_markers(
        X_scaled, marker_names,
        batch_codes=batch_codes,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
    )

    # Spatial sampler
    coords = get_spatial_coords(adata)
    sampler = SpatialClusterSampler(
        coords, batch_codes, cluster_size=cluster_size, cells_per_step=cells_per_step
    )
    steps_per_epoch = len(sampler)

    # Model
    model = SpaNCyShift(
        n_markers=n_markers,
        n_batches=n_batches,
        n_samples=n_samples,
        n_cycles=n_cycles,
        marker_is_bimodal=marker_is_bimodal,
        thresholds=thresholds,
        sharpness=sharpness,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    # Tensors
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    batch_all = torch.tensor(batch_codes, dtype=torch.long)
    sample_all = torch.tensor(sample_codes, dtype=torch.long)

    huber = nn.HuberLoss(delta=1.0)

    history: Dict[str, List[float]] = {
        "loss": [], "recon": [], "identity": [], "align": [], "mmd": [], "lr": [], "mmd_weight": [],
    }

    log.info(
        "Starting training: %d epochs x %d steps, device=%s, "
        "w_recon=%.1f, w_identity=%.1f, w_align=%.1f, w_mmd=%.1f (ramp %d→%d)",
        n_epochs, steps_per_epoch, device_str,
        w_recon, w_identity, w_align, w_mmd, mmd_ramp_start, mmd_ramp_end,
    )

    for epoch in range(n_epochs):
        epoch_loss = epoch_identity = epoch_align = epoch_mmd = 0.0

        # MMD weight warmup
        if epoch < mmd_ramp_start:
            mmd_w = 0.0
        elif epoch < mmd_ramp_end:
            mmd_w = w_mmd * (epoch - mmd_ramp_start) / max(mmd_ramp_end - mmd_ramp_start, 1)
        else:
            mmd_w = w_mmd

        epoch_recon = 0.0
        for step in range(steps_per_epoch):
            idx = sampler.sample()
            if len(idx) < 10:
                continue

            X_batch = X_all[idx].to(device)
            batch_ids = batch_all[idx].to(device)
            sample_ids = sample_all[idx].to(device)

            out = model(X_batch, batch_ids, sample_ids, marker_cycles_t)

            # Anchor CycleDegradationModel: X_affine must stay close to scaled input
            loss_recon = huber(out["X_affine"], X_batch)
            # Keep shifts small relative to affine baseline
            loss_identity = huber(out["X_shifted"], out["X_affine"])
            loss_align = quantile_alignment_loss(out["X_shifted"], sample_ids)
            loss_mmd = mmd_rbf_loss(
                out["X_shifted"], batch_ids,
                bandwidths=mmd_bandwidths, n_samples=mmd_samples,
            ) if mmd_w > 0 else torch.tensor(0.0, device=device)

            loss = (w_recon * loss_recon
                    + w_identity * loss_identity
                    + w_align * loss_align
                    + mmd_w * loss_mmd)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_identity += loss_identity.item()
            epoch_align += loss_align.item()
            epoch_mmd += loss_mmd.item()

            if steps_per_epoch > 20 and (step + 1) % max(1, steps_per_epoch // 5) == 0:
                log.info(
                    "  Epoch %3d  step %d/%d  loss=%.4f  recon=%.4f  id=%.4f  align=%.4f  mmd=%.4f",
                    epoch + 1, step + 1, steps_per_epoch,
                    loss.item(), loss_recon.item(), loss_identity.item(), loss_align.item(), loss_mmd.item(),
                )

        scheduler.step()
        n_steps = max(steps_per_epoch, 1)

        history["loss"].append(epoch_loss / n_steps)
        history["recon"].append(epoch_recon / n_steps)
        history["identity"].append(epoch_identity / n_steps)
        history["align"].append(epoch_align / n_steps)
        history["mmd"].append(epoch_mmd / n_steps)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["mmd_weight"].append(mmd_w)

        log.info(
            "Epoch %3d/%d  loss=%.4f  recon=%.4f  id=%.4f  align=%.4f  mmd=%.4f  "
            "mmd_w=%.2f  lr=%.2e",
            epoch + 1, n_epochs,
            history["loss"][-1], history["recon"][-1], history["identity"][-1],
            history["align"][-1], history["mmd"][-1],
            mmd_w, optimizer.param_groups[0]["lr"],
        )

    log.info("Training complete. Final loss=%.4f", history["loss"][-1])
    return model, scaler, marker_cycles, history


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def normalize_adata(
    adata: ad.AnnData,
    model: SpaNCyShift,
    scaler: RobustScaler,
    marker_cycles: np.ndarray,
    device_str: str = "cpu",
    inference_batch_size: int = 50000,
    mode: str = "shift",
    align_samples: bool = False,
    sample_col: str = "sample_id",
    shift_alpha: float = 1.0,
    layer_name: str = "normalized",
) -> ad.AnnData:
    """Run inference and store results in adata.layers[layer_name].

    Args:
        mode: "affine" — CycleDegradationModel only (shape-preserving per-marker).
              "shift"  — (default) Affine + adaptive per-sample shifts.
        shift_alpha: Blending weight 0=pure affine, 1=full shift. Default 1.0.
        align_samples: Apply post-hoc per-sample peak alignment (optional extra step).
        layer_name: Layer name to write (default 'normalized').
    """
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    X_raw = np.asarray(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    X_scaled = scaler.transform(np.log1p(np.clip(X_raw, 0, None))).astype(np.float32)

    batch_codes = adata.obs["batch"].astype("category").cat.codes.values.astype(np.int64)

    s_col = None
    for col in (sample_col, "sample_id", "sample", "Sample", "patient_id", "patient"):
        if col in adata.obs.columns:
            s_col = col
            break
    if s_col is None:
        s_col = "batch"
    sample_codes = adata.obs[s_col].astype("category").cat.codes.values.astype(np.int64)

    marker_cycles_t = torch.tensor(marker_cycles, dtype=torch.long, device=device)
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    batch_all = torch.tensor(batch_codes, dtype=torch.long)
    sample_all = torch.tensor(sample_codes, dtype=torch.long)

    n_cells = adata.n_obs
    n_markers = adata.n_vars
    X_norm_scaled = np.zeros((n_cells, n_markers), dtype=np.float32)
    n_chunks = max(1, math.ceil(n_cells / inference_batch_size))

    alpha = 0.0 if mode == "affine" else shift_alpha
    log.info("Normalizing %d cells, mode=%s, alpha=%.2f...", n_cells, mode, alpha)

    for ci in range(n_chunks):
        start = ci * inference_batch_size
        end = min(start + inference_batch_size, n_cells)

        X_chunk = X_all[start:end].to(device)
        batch_chunk = batch_all[start:end].to(device)
        sample_chunk = sample_all[start:end].to(device)

        X_out = model.normalize(X_chunk, batch_chunk, sample_chunk, marker_cycles_t, alpha=alpha)
        X_norm_scaled[start:end] = X_out.cpu().numpy()

        if (ci + 1) % 10 == 0 or ci == 0:
            log.info("  Inference chunk %d/%d", ci + 1, n_chunks)

    X_norm_log = scaler.inverse_transform(X_norm_scaled)

    if align_samples:
        s_col_align = None
        for col in (sample_col, "sample", "Sample", "patient_id"):
            if col in adata.obs.columns:
                s_col_align = col
                break
        if s_col_align is not None:
            sample_ids = adata.obs[s_col_align].values
            if len(np.unique(sample_ids)) > 1:
                log.info("Applying post-hoc sample mode alignment...")
                X_norm_log = sample_mode_align(
                    X_norm_log, sample_ids, marker_names=list(adata.var_names)
                )

    # Log-space cap at 1.2x raw maximum (prevents gamma/beta overcorrection)
    raw_log_max = np.log1p(np.clip(X_raw, 0, None)).max()
    log_cap = max(raw_log_max * 1.2, 15.0)
    n_capped = int((X_norm_log > log_cap).sum())
    if n_capped > 0:
        log.info("Capping %d values (%.4f%%) exceeding log-space cap %.2f",
                 n_capped, 100.0 * n_capped / X_norm_log.size, log_cap)
    X_norm_log = np.clip(X_norm_log, None, log_cap)

    X_norm = np.clip(np.expm1(X_norm_log), 0, None)
    adata.layers[layer_name] = X_norm

    log.info(
        "Done. mode=%s alpha=%.2f — stored in adata.layers['%s'] "
        "(min=%.4f, max=%.4f, mean=%.4f)",
        mode, alpha, layer_name, X_norm.min(), X_norm.max(), X_norm.mean(),
    )
    return adata


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SpaNCy-Shift: adaptive shift-based CyCIF batch correction"
    )
    parser.add_argument("--input", required=True, help="Input .h5ad path")
    parser.add_argument("--output", required=True, help="Output .h5ad path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cells_per_step", type=int, default=16000)
    parser.add_argument("--w_mmd", type=float, default=0.5)
    parser.add_argument("--w_recon", type=float, default=1.0)
    parser.add_argument("--w_identity", type=float, default=0.5)
    parser.add_argument("--w_align", type=float, default=0.5)
    parser.add_argument("--mmd_ramp_start", type=int, default=3)
    parser.add_argument("--mmd_ramp_end", type=int, default=7)
    parser.add_argument("--mmd_samples", type=int, default=256)
    parser.add_argument("--bimodal_min_batch_frac", type=float, default=0.5,
                        help="Fraction of batches that must show bimodal histogram for a marker to be classified bimodal (default: 0.5)")
    parser.add_argument("--shift_alpha", type=float, default=1.0,
                        help="Blending weight for shift correction (0=affine, 1=full shift)")
    parser.add_argument("--mode", choices=["affine", "shift"], default="shift")
    parser.add_argument("--align_samples", action="store_true",
                        help="Apply post-hoc per-sample peak alignment")
    parser.add_argument("--cycle_config", type=str, default=None,
                        help="Path to JSON file with cycle config (default: PRAD panel)")
    parser.add_argument("--layer_name", default="normalized",
                        help="Output layer name in adata.layers")
    args = parser.parse_args()

    cycle_config = DEFAULT_CYCLE_CONFIG
    if args.cycle_config:
        with open(args.cycle_config) as f:
            raw = json.load(f)
            cycle_config = {int(k): v for k, v in raw.items()}

    adata = load_adata(args.input)

    model, scaler, marker_cycles, history = train(
        adata,
        cycle_config=cycle_config,
        n_epochs=args.epochs,
        lr=args.lr,
        cells_per_step=args.cells_per_step,
        device_str=args.device,
        w_mmd=args.w_mmd,
        w_recon=args.w_recon,
        w_identity=args.w_identity,
        w_align=args.w_align,
        mmd_ramp_start=args.mmd_ramp_start,
        mmd_ramp_end=args.mmd_ramp_end,
        mmd_samples=args.mmd_samples,
        bimodal_min_batch_frac=args.bimodal_min_batch_frac,
    )

    normalize_adata(
        adata, model, scaler, marker_cycles,
        device_str=args.device,
        mode=args.mode,
        shift_alpha=args.shift_alpha,
        align_samples=args.align_samples,
        layer_name=args.layer_name,
    )

    adata.write_h5ad(args.output)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
