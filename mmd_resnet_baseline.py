#!/usr/bin/env python
"""
MMD-ResNet baseline (Shaham et al., Bioinformatics 2017, "Removal of batch effects using
distribution-matching residual networks"), ported from the authors' Keras code
(github.com/ushaham/BatchEffectRemoval, src/train_MMD_ResNet.py + CostFunctions.py).

Faithful to the original:
  - log transform, then StandardScaler fitted on the SOURCE batch, applied to both
  - 3 residual blocks: BN → ReLU → Dense(25) → BN → ReLU → Dense(d), identity skip
  - Dense weights ~ N(0, 1e-4²), L2 penalty 1e-2 on Dense kernels
  - biased MMD², Gaussian kernels at bandwidths [med/2, med, 2·med], med = median
    k-NN distance within the target batch
  - RMSprop lr 1e-3, ×0.1 every 150 epochs, 500 epochs, batch 1000,
    10% validation split, early stopping on validation MMD with patience 50
  - applied to RAW data (no Stage 1) and to ALL markers (no bimodal masking)

Adaptations (the original maps ONE source batch onto ONE target batch):
  - multi-batch: one network per non-reference batch, each mapping that batch onto a single
    reference batch. The reference batch is the one containing the most per-marker KL-medoid
    reference samples (same rule as the CFM/DDPM Stage 2 references).
  - log1p instead of CyTOF's arcsinh-style transform (same log1p space as SpaNCy-Shift).
  - training uses at most max_train_cells cells per batch (per epoch the original iterates
    over the whole source; PRAD batches have ~250k cells). The trained network is applied to
    every cell of its batch.
  - median k-NN distance estimated on a subsample (n_med_cells) with k = n_neighbors.

Usage:
    nets, info = train_mmd_resnet(adata, ref_sample_per_marker, device_str='cuda')
    X_norm = apply_mmd_resnet(adata, nets, info)          # (N, M) count scale
"""

import logging
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as _sp
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("mmd_resnet")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)
    log.propagate = False


class _ResBlock(nn.Module):
    def __init__(self, d: int, hidden: int = 25):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d)
        self.fc1 = nn.Linear(d, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, d)
        for fc in (self.fc1, self.fc2):
            nn.init.normal_(fc.weight, std=1e-4)
            nn.init.zeros_(fc.bias)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.fc1(self.act(self.bn1(x)))
        h = self.fc2(self.act(self.bn2(h)))
        return x + h


class MMDResNet(nn.Module):
    def __init__(self, d: int, hidden: int = 25, n_blocks: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*[_ResBlock(d, hidden) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(x)

    def l2_penalty(self) -> torch.Tensor:
        return sum((m.weight ** 2).sum() for m in self.modules() if isinstance(m, nn.Linear))


def _median_knn_distance(X: np.ndarray, n_neighbors: int, n_cells: int, rng) -> float:
    if len(X) > n_cells:
        X = X[rng.choice(len(X), n_cells, replace=False)]
    k = min(n_neighbors + 1, len(X))
    dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return float(np.median(dist[:, 1:]))


def _mmd_biased(x: torch.Tensor, y: torch.Tensor, scales) -> torch.Tensor:
    def k(a, b):
        d2 = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d2 / (2.0 * s ** 2)) for s in scales)
    return k(x, x).mean() - 2.0 * k(x, y).mean() + k(y, y).mean()


def _find_ref_batch(adata, ref_sample_per_marker: Dict[str, Optional[str]],
                    batch_col: str, sample_col: str) -> str:
    votes: Dict[str, int] = {}
    obs = adata.obs
    for s in ref_sample_per_marker.values():
        if s is None:
            continue
        m = obs[sample_col] == s
        if m.any():
            b = str(obs.loc[m, batch_col].iloc[0])
            votes[b] = votes.get(b, 0) + 1
    ref = max(votes, key=votes.get)
    log.info("Reference batch: %s  (votes: %s)", ref,
             dict(sorted(votes.items(), key=lambda kv: -kv[1])))
    return ref


def train_mmd_resnet(
    adata,
    ref_sample_per_marker: Dict[str, Optional[str]],
    batch_col: str = "batch_id",
    sample_col: str = "sample_id",
    device_str: str = "cpu",
    n_epochs: int = 500,
    batch_size: int = 1000,
    lr: float = 1e-3,
    lr_drop_every: int = 150,
    l2: float = 1e-2,
    val_frac: float = 0.1,
    patience: int = 50,
    max_train_cells: int = 50_000,
    n_neighbors: int = 25,
    n_med_cells: int = 5_000,
    seed: int = 0,
) -> Tuple[Dict[str, Tuple[MMDResNet, StandardScaler]], dict]:
    """Train one MMD-ResNet per non-reference batch (source) onto the reference batch."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str)

    X = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X, dtype=np.float32)
    X_log = np.log1p(np.clip(X, 0, None))
    batches = adata.obs[batch_col].astype(str).values
    ref = _find_ref_batch(adata, ref_sample_per_marker, batch_col, sample_col)

    tgt_all = X_log[batches == ref]
    nets: Dict[str, Tuple[MMDResNet, StandardScaler]] = {}
    history: Dict[str, dict] = {}

    for b in sorted(set(batches) - {ref}):
        src_all = X_log[batches == b]
        src = src_all[rng.choice(len(src_all), min(max_train_cells, len(src_all)), replace=False)]
        tgt = tgt_all[rng.choice(len(tgt_all), min(max_train_cells, len(tgt_all)), replace=False)]

        scaler = StandardScaler().fit(src)                 # fitted on SOURCE (as original)
        src_s = scaler.transform(src).astype(np.float32)
        tgt_s = scaler.transform(tgt).astype(np.float32)

        med = _median_knn_distance(tgt_s, n_neighbors, n_med_cells, rng)
        scales = [med / 2.0, med, med * 2.0]

        n_val_s, n_val_t = int(len(src_s) * val_frac), int(len(tgt_s) * val_frac)
        ps, pt = rng.permutation(len(src_s)), rng.permutation(len(tgt_s))
        src_tr, src_va = src_s[ps[n_val_s:]], src_s[ps[:n_val_s]]
        tgt_tr, tgt_va = tgt_s[pt[n_val_t:]], tgt_s[pt[:n_val_t]]
        tgt_tr_t = torch.tensor(tgt_tr, device=device)
        src_va_t = torch.tensor(src_va[:batch_size], device=device)
        tgt_va_t = torch.tensor(tgt_va[:batch_size], device=device)

        net = MMDResNet(src_s.shape[1]).to(device)
        opt = torch.optim.RMSprop(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_drop_every, gamma=0.1)

        best, wait, h = float("inf"), 0, {"train": [], "val": []}
        log.info("MMD-ResNet %s → %s: %d src / %d tgt train cells, med kNN dist %.3f",
                 b, ref, len(src_tr), len(tgt_tr), med)
        for epoch in range(n_epochs):
            net.train()
            order = rng.permutation(len(src_tr))
            e_loss, n_steps = 0.0, 0
            for s in range(0, len(order), batch_size):
                xb = torch.tensor(src_tr[order[s:s + batch_size]], device=device)
                if len(xb) < 2:
                    continue
                yb = tgt_tr_t[torch.randint(len(tgt_tr_t), (len(xb),), device=device)]
                loss = _mmd_biased(net(xb), yb, scales) + l2 * net.l2_penalty()
                opt.zero_grad()
                loss.backward()
                opt.step()
                e_loss += loss.item()
                n_steps += 1
            sched.step()

            net.eval()
            with torch.no_grad():
                val = _mmd_biased(net(src_va_t), tgt_va_t, scales).item()
            h["train"].append(e_loss / max(n_steps, 1))
            h["val"].append(val)
            if val < best - 1e-7:
                best, wait = val, 0
            else:
                wait += 1
                if wait >= patience:
                    log.info("  early stop at epoch %d (best val MMD %.5f)", epoch + 1, best)
                    break
            if (epoch + 1) % 50 == 0:
                log.info("  epoch %d  train %.5f  val MMD %.5f", epoch + 1, h["train"][-1], val)

        net.eval()
        nets[b] = (net, scaler)
        history[b] = h

    info = {"ref_batch": ref, "batch_col": batch_col, "history": history}
    return nets, info


@torch.no_grad()
def apply_mmd_resnet(adata, nets, info, device_str: str = "cpu",
                     chunk: int = 100_000) -> np.ndarray:
    """Map every non-reference batch through its network; reference batch unchanged.

    Returns the normalized matrix on count scale (expm1 of log1p space, clipped at 0).
    """
    device = torch.device(device_str)
    X = np.asarray(adata.X.toarray() if _sp.issparse(adata.X) else adata.X, dtype=np.float32)
    X_log = np.log1p(np.clip(X, 0, None))
    out = X_log.copy()
    batches = adata.obs[info["batch_col"]].astype(str).values
    for b, (net, scaler) in nets.items():
        idx = np.where(batches == b)[0]
        net = net.to(device).eval()
        for s in range(0, len(idx), chunk):
            ii = idx[s:s + chunk]
            xs = torch.tensor(scaler.transform(X_log[ii]).astype(np.float32), device=device)
            out[ii] = scaler.inverse_transform(net(xs).cpu().numpy())
    return np.clip(np.expm1(out), 0, None).astype(np.float32)
