#!/usr/bin/env python
"""
UniFORM-style positive population change (Wang et al., Cell Reports Methods 2025).

Definition (STAR Methods, "positive population change analysis"): "For each marker, a local
positive population threshold is calculated for each raw sample, while a global threshold is
determined for the combined normalized samples. Then we evaluated the proportions of cells above
the local threshold and global threshold for each sample and the mean and standard deviation of
changes in positive population proportions can be determined."

GMM threshold: copied from UniFORM's own code (github.com/kunlunW/UniFORM,
UniFORM/normalization.py, plot_gmm): natural log with log(0) -> 0; 2-component GaussianMixture
(init_params='random_from_data', max_iter=1000, random_state=10, covariance_type='full') fitted on
all cells; the negative class is the component whose largest member is smaller; threshold = the
largest value assigned to the negative class (log space).

NOT published by UniFORM, so these are our reading (flagged in the output):
  - change = percentage POINTS (pct_norm - pct_raw); the paper says "changes in positive
    population proportions" and plots the absolute mean change with its SD
  - mean and SD are taken over samples, per marker
UniFORM's preprocessing.py has a second variant that drops intensities < 1 before the log;
`drop_below_one=True` reproduces that one instead.

Differs from spancy_shift.positive_population_table(), which applies each sample's RAW threshold
to that sample's normalized values (log10, max-of-negative-class GMM, subsampled fit).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def _log_values(x: np.ndarray, drop_below_one: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if drop_below_one:
        x = x[x >= 1.0]
    x = np.clip(x, 0, None)
    with np.errstate(divide="ignore"):
        lx = np.log(x)
    lx[np.isneginf(lx)] = 0.0
    return lx


def uniform_gmm_threshold(
    values: np.ndarray,
    drop_below_one: bool = False,
    max_cells: Optional[int] = None,
    seed: int = 0,
) -> float:
    """UniFORM GMM threshold for one 1D intensity array. Returns the threshold in ORIGINAL
    intensity units (exp of the log-space cut), like UniFORM's plot_gmm.

    max_cells: optional subsample for the FIT only (None = all cells, as in UniFORM); the
    threshold is still the max of the negative class over all cells.
    """
    lx = _log_values(values, drop_below_one)
    if lx.size < 2 or np.ptp(lx) == 0:
        return float(np.exp(lx.max())) if lx.size else np.nan
    X = lx.reshape(-1, 1)
    X_fit = X
    if max_cells is not None and len(X) > max_cells:
        X_fit = X[np.random.default_rng(seed).choice(len(X), max_cells, replace=False)]
    gmm = GaussianMixture(n_components=2, init_params="random_from_data", max_iter=1000,
                          random_state=10, covariance_type="full")
    gmm.fit(X_fit)
    preds = gmm.predict(X)
    class0 = lx[preds == 0]
    class1 = lx[preds == 1]
    if class0.size == 0:
        class0 = np.array([0.0])
    if class1.size == 0:
        class1 = np.array([0.0])
    neg = class1 if class0.max() > class1.max() else class0
    return float(np.exp(neg.max()))


def local_thresholds(
    X_raw: np.ndarray,
    sample_ids: np.ndarray,
    marker_names: List[str],
    min_cells: int = 10,
    **kw,
) -> Dict[Tuple[str, str], float]:
    """Per (marker, sample) threshold on RAW data. Compute once and reuse for every method."""
    out = {}
    for m, name in enumerate(marker_names):
        for s in sorted(np.unique(sample_ids).tolist()):
            idx = sample_ids == s
            if idx.sum() >= min_cells:
                out[(name, s)] = uniform_gmm_threshold(X_raw[idx, m], **kw)
    return out


def global_thresholds(X_norm: np.ndarray, marker_names: List[str], **kw) -> Dict[str, float]:
    """One threshold per marker on ALL normalized cells pooled."""
    return {name: uniform_gmm_threshold(X_norm[:, m], **kw) for m, name in enumerate(marker_names)}


def positive_population_uniform(
    X_raw: np.ndarray,
    X_norm: np.ndarray,
    sample_ids: np.ndarray,
    marker_names: List[str],
    local_thr: Optional[Dict[Tuple[str, str], float]] = None,
    **kw,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per (marker, sample): % above the local raw threshold in RAW vs % above the global
    threshold in NORMALIZED. Returns (table, global thresholds)."""
    if local_thr is None:
        local_thr = local_thresholds(X_raw, sample_ids, marker_names, **kw)
    g_thr = global_thresholds(X_norm, marker_names, **kw)
    rows = []
    for m, name in enumerate(marker_names):
        for s in sorted(np.unique(sample_ids).tolist()):
            if (name, s) not in local_thr:
                continue
            idx = sample_ids == s
            pr = 100.0 * float((X_raw[idx, m] > local_thr[(name, s)]).mean())
            pn = 100.0 * float((X_norm[idx, m] > g_thr[name]).mean())
            rows.append(dict(marker=name, sample=s, thr_local=local_thr[(name, s)],
                             thr_global=g_thr[name], pct_raw=pr, pct_norm=pn, delta_pp=pn - pr))
    return pd.DataFrame(rows), g_thr


def summarize_uniform(table: pd.DataFrame, tol_pp: float = 5.0) -> pd.DataFrame:
    """Per marker: mean and SD over samples of the change (percentage points), |mean|, pass flag.

    Also mean_abs_pp = mean over samples of |change|: the paper's table has only positive values
    for every method, which suggests this aggregation; mxnorm_benchmark_pospop_uniform.ipynb tests
    which one reproduces the paper.
    """
    g = table.groupby("marker", sort=False)["delta_pp"]
    s = pd.DataFrame({"mean_delta_pp": g.mean(), "sd_delta_pp": g.std(ddof=1),
                      "mean_abs_delta_pp": g.apply(lambda d: d.abs().mean())})
    s["abs_mean_pp"] = s.mean_delta_pp.abs()
    s["pass"] = s.abs_mean_pp < tol_pp
    return s
