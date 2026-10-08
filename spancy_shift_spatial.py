#!/usr/bin/env python
"""
SpaNCy-Shift spatial — the region GNN (primary method, chosen 2026-10-08).

Stage 1 (analytic, unchanged): per-marker shift of every sample toward a KL-medoid reference sample
  (median shift for unimodal markers, negative-peak shift for bimodal ones) — imported from
  spancy_shift.py.
Stage 2 (graph neural network): a 2-layer GATv2 encoder + residual decoder corrects each cell from its
  own markers and its 15 spatial neighbours (same sample, same scanned region, 2-hop receptive field).
  Losses: MMD between batches (bimodal markers masked), Huber on the correction, NT-Xent (one spatial
  neighbour as positive), adversarial batch classifier with gradient reversal.
  Output = Stage 1 + alpha * correction (alpha = 0.6).

Training draws, per batch and step, 32 random seed cells from the whole batch plus their 15
neighbours (NeighbourPatchSampler): representative batches for the MMD, small 2-hop halos for speed.

Same model, losses, sampler and random-number streams as spancy_shift_region.train(sampler='patches',
graph_mode='fixed') + normalize_adata(graph_mode='fixed'). Engineering changes only:
  - prepare(): Stage 1, scaling and the region graph are computed once and reused;
  - cell data and batch labels live on the GPU during training;
  - neighbourhood lookups reuse two buffers instead of allocating cell-sized arrays every step;
  - correct() computes the correction once; apply() turns it into any alpha by a multiplication.

Usage:
    prep = prepare(adata)                 # Stage 1 + region graph
    model, history = train(prep, device_str='cuda')
    delta = correct(prep, model, device_str='cuda')
    adata.layers['normalized'] = apply(prep, delta, alpha=0.6)
    # or, in one call: adata_out = normalize_adata(adata, prep, model, device_str='cuda')
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from spancy_shift import (  # noqa: F401  (re-exported for notebooks)
    GNNStage2,
    _get_col,
    detect_bimodal_markers,
    find_best_sample_per_marker,
    load_adata,
    log,
    log1p_scale,
    mmd_rbf_loss,
    positive_population_table,
    shift_normalize_per_marker,
)


# ──────────────────────────────────────────────────────────────────────────────
# Spatial graph (one k-NN per tissue region)
# ──────────────────────────────────────────────────────────────────────────────

def region_labels(adata: ad.AnnData) -> np.ndarray:
    """One label per tissue region: sample_id|scene_id (scene_id alone restarts for every sample)."""
    scene_col = _get_col(adata, ["scene_id", "sample_id"], "batch_id")
    scenes = adata.obs[scene_col].astype(str).values
    if "sample_id" in adata.obs and scene_col != "sample_id":
        scenes = adata.obs["sample_id"].astype(str).values + "|" + scenes
    return scenes


def build_region_graph(adata: ad.AnnData, k: int = 15) -> np.ndarray:
    """(N, k) int32 matrix of each cell's k nearest cells in its own region (x, y)."""
    xy_x = _get_col(adata, ["x", "X", "centroid_x"], None)
    xy_y = _get_col(adata, ["y", "Y", "centroid_y"], None)
    if xy_x is None or xy_y is None or xy_x not in adata.obs or xy_y not in adata.obs:
        raise ValueError("Spatial coordinates (x, y) not found in adata.obs")
    regions = region_labels(adata)
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
    """k-NN matrix with subgraph / halo extraction. Two cell-sized buffers are reused across calls."""

    def __init__(self, knn: np.ndarray):
        self.knn = knn
        self.N, self.k = knn.shape
        self._mask = np.zeros(self.N, dtype=bool)
        self._loc = np.full(self.N, -1, dtype=np.int64)

    def expand(self, idx: np.ndarray, hops: int = 2) -> np.ndarray:
        """idx followed by the new cells of its `hops`-hop halo (targets keep rows 0..len(idx)-1)."""
        mask = self._mask
        mask[idx] = True
        parts = [np.asarray(idx)]
        frontier = np.asarray(idx)
        for _ in range(hops):
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
        """edge_index (2, E) among idx, local coordinates, edges neighbour -> cell (PyG flow)."""
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
        """Row of each of `cells` within ctx (-1 if absent)."""
        loc = self._loc
        loc[ctx] = np.arange(len(ctx))
        out = loc[cells]
        loc[ctx] = -1
        return out

    @staticmethod
    def target_in_degree(n_targets: int, edge_index: np.ndarray) -> float:
        if edge_index.shape[1] == 0:
            return 0.0
        return float(np.bincount(edge_index[1], minlength=n_targets)[:n_targets].mean())


class NeighbourPatchSampler:
    """Per batch and step: n_per_batch // (k+1) seed cells drawn uniformly from ALL cells of the
    batch, each with its k spatial neighbours. Representative batches for the MMD, small halos."""

    def __init__(self, batch_codes: np.ndarray, knn: np.ndarray, n_per_batch: int = 512, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.knn = knn
        self.n_seeds = max(1, n_per_batch // (knn.shape[1] + 1))
        self.unique_batches = np.unique(batch_codes)
        self.batch_indices = {int(b): np.where(batch_codes == b)[0] for b in self.unique_batches}
        self._n_steps = max(1, len(batch_codes) // (n_per_batch * len(self.unique_batches)))

    def sample(self) -> np.ndarray:
        parts = []
        for b in self.unique_batches:
            idx = self.batch_indices[int(b)]
            seeds = self.rng.choice(idx, size=min(self.n_seeds, len(idx)), replace=False)
            parts.append(seeds)
            parts.append(self.knn[seeds].ravel())
        cat = np.concatenate(parts)
        _, first = np.unique(cat, return_index=True)      # patches can overlap; keep first
        return cat[np.sort(first)]

    def __len__(self) -> int:
        return self._n_steps


def nt_xent_pairs(za: torch.Tensor, zb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """SimCLR NT-Xent for positive pairs (za[i], zb[i]); all other rows are negatives."""
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
    adata_base: ad.AnnData            # Stage 1 output in layers['normalized_base']
    X_scaled: np.ndarray              # log1p(Stage 1) -> RobustScaler, float32
    scaler: RobustScaler
    is_bimodal: np.ndarray
    batch_codes: np.ndarray
    regions: np.ndarray
    adj: AdjacencyIndex
    ref_sample_per_marker: Dict[str, Optional[str]]
    seconds: float


def prepare(
    adata: ad.AnnData,
    ref_sample_per_marker: Optional[Dict[str, Optional[str]]] = None,
    k_neighbors: int = 15,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
) -> Prepared:
    t0 = time.time()
    if ref_sample_per_marker is None:
        ref_sample_per_marker = find_best_sample_per_marker(adata)
    adata_base, is_bimodal, _ = shift_normalize_per_marker(
        adata, ref_sample_per_marker,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
        layer_name="normalized_base",
    )
    X_base = np.asarray(adata_base.layers["normalized_base"], dtype=np.float32)
    X_scaled, scaler = log1p_scale(X_base)
    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = adata.obs[batch_col].astype("category").cat.codes.values.astype(np.int64)
    adj = AdjacencyIndex(build_region_graph(adata, k=k_neighbors))
    log.info("Prepared: Stage 1, scaling and region graph in %.0f s; bimodal markers excluded from MMD: %s",
             time.time() - t0, [n for n, b in zip(adata.var_names, is_bimodal) if b])
    return Prepared(adata_base, X_scaled, scaler, np.asarray(is_bimodal, dtype=bool), batch_codes,
                    region_labels(adata), adj, ref_sample_per_marker, time.time() - t0)


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
    max_steps_per_epoch: Optional[int] = None,
) -> Tuple[GNNStage2, Dict[str, List[float]]]:
    """Train the Stage 2 region GNN. Returns (model, history); history['train_min'] = wall time."""
    t0 = time.time()
    device = torch.device(device_str)
    adj, knn = prep.adj, prep.adj.knn
    n_batches = int(prep.batch_codes.max()) + 1

    model = GNNStage2(n_markers=prep.X_scaled.shape[1], n_batches=n_batches, hidden=gnn_hidden,
                      latent=gnn_latent, n_heads=gnn_heads).to(device)
    unimodal_mask = torch.tensor(~prep.is_bimodal, dtype=torch.bool, device=device)
    sampler = NeighbourPatchSampler(prep.batch_codes, knn, n_per_batch=n_per_batch)
    steps = len(sampler) if max_steps_per_epoch is None else min(len(sampler), max_steps_per_epoch)
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
    logged = False

    for epoch in range(n_epochs):
        grl_lam = grl_max * min(1.0, 2.0 * epoch / max(n_epochs - 1, 1))
        model.discriminator.grl.lam = grl_lam
        sums = dict(loss=0.0, recon=0.0, contrast=0.0, adv=0.0, mmd=0.0)
        for _ in range(steps):
            idx = sampler.sample()
            B = len(idx)
            pos_global = knn[idx, pos_rng.integers(0, knn.shape[1], size=B)]
            ctx = adj.expand(idx, hops=2)
            edge_arr = adj.get_subgraph(ctx)
            pos_local = adj.local_positions(ctx, pos_global)
            if (pos_local < 0).any():
                raise RuntimeError("neighbour positive missing from the 2-hop halo")
            if not logged:
                log.info("Graph check (training): %d targets, %d context cells, mean target in-degree %.2f",
                         B, len(ctx), adj.target_in_degree(B, edge_arr))
                logged = True

            ctx_t = torch.as_tensor(ctx, dtype=torch.long, device=device)
            edge_index = torch.as_tensor(edge_arr, dtype=torch.long, device=device)
            _, delta_all, z_proj_all, logits_all = model(X_dev[ctx_t], edge_index)
            delta, z_proj, logits = delta_all[:B], z_proj_all[:B], logits_all[:B]
            X_batch = X_dev[ctx_t[:B]]
            b_ids = batch_dev[ctx_t[:B]]

            X_out = X_batch + delta
            loss_recon = huber(X_out, X_batch)
            pos_t = torch.as_tensor(pos_local, dtype=torch.long, device=device)
            loss_contrast = nt_xent_pairs(z_proj, z_proj_all[pos_t], temperature)
            loss_adv = ce(logits, b_ids) if w_adv != 0 else torch.zeros((), device=device)
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

        scheduler.step()
        for k_ in sums:
            history[k_].append(sums[k_] / max(steps, 1))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grl_lambda"].append(grl_lam)
        log.info("Epoch %2d/%d  loss=%.4f  recon=%.4f  contrast=%.4f  adv=%.4f  mmd=%.4f  (%.1f min)",
                 epoch + 1, n_epochs, history["loss"][-1], history["recon"][-1],
                 history["contrast"][-1], history["adv"][-1], history["mmd"][-1], (time.time() - t0) / 60)

    history["train_min"] = [(time.time() - t0) / 60]
    log.info("Training complete in %.1f min", history["train_min"][0])
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def correct(prep: Prepared, model: GNNStage2, device_str: str = "cpu", chunk_size: int = 2048) -> np.ndarray:
    """Stage 2 correction (scaled space, alpha = 1) for every cell. Each region is processed in
    chunks run together with their 2-hop halo, so every cell sees its full receptive field."""
    t0 = time.time()
    device = torch.device(device_str)
    model = model.to(device).eval()
    X_dev = torch.tensor(prep.X_scaled, dtype=torch.float32, device=device)
    delta_out = np.zeros_like(prep.X_scaled)
    for rg in np.unique(prep.regions):
        reg_idx = np.where(prep.regions == rg)[0]
        for start in range(0, len(reg_idx), chunk_size):
            chunk = reg_idx[start:start + chunk_size]
            ctx = prep.adj.expand(chunk, hops=2)
            edge_index = torch.as_tensor(prep.adj.get_subgraph(ctx), dtype=torch.long, device=device)
            _, delta, _, _ = model(X_dev[torch.as_tensor(ctx, dtype=torch.long, device=device)], edge_index)
            delta_out[chunk] = delta[:len(chunk)].cpu().numpy()
    log.info("Correction computed in %.1f s", time.time() - t0)
    return delta_out


def apply(prep: Prepared, delta: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Count-scale output: inverse-scale(Stage 1 scaled + alpha * correction) -> expm1 -> >= 0."""
    X_log = prep.scaler.inverse_transform(prep.X_scaled + alpha * delta)
    return np.clip(np.expm1(X_log), 0, None).astype(np.float32)


def normalize_adata(adata: ad.AnnData, prep: Prepared, model: GNNStage2, alpha: float = 0.6,
                    device_str: str = "cpu", layer_name: str = "normalized") -> ad.AnnData:
    """Copy of the Stage 1 AnnData with the Stage 2 output in layers[layer_name]."""
    out = prep.adata_base.copy()
    out.layers[layer_name] = apply(prep, correct(prep, model, device_str=device_str), alpha)
    return out
