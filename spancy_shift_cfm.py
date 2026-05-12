#!/usr/bin/env python
"""
SpaNCy-Shift CFM: OT-Conditional Flow Matching for CyCIF batch normalization.

Stage 1 (analytic, identical to spancy_shift.py):
    shift_normalize_per_marker() → X_base  [kBET ≈ 0.631]

Stage 2 (OT-CFM):
    FlowMLP conditioned on source batch + time t ∈ [0,1] learns a velocity field
    transporting source-batch cells toward the reference-batch distribution.

    Training:
        - Mini-batch OT coupling (Hungarian algorithm on L2 cost matrix)
          pairs source cells with nearest reference cells.
        - Loss: MSE between predicted velocity and straight-line velocity (x_1 - x_0).
        - Conditioning on SOURCE batch → model learns per-batch corrections.
        - No spatial graph, no torch-geometric, no adversarial training.

    Inference:
        - Euler ODE integration from t=0 to t=1 (n_steps steps).
        - n_steps controls correction strength: 10 → conservative, 50 → full transport.
        - Reference-batch cells should receive near-zero corrections (already at target).

OT-CFM preserves biology via minimum-transport coupling: cells move the least
distance necessary to match the reference distribution. No explicit bimodal masking
needed — the velocity field learns the full joint distribution including bimodal structure.

Usage:
    model, scaler, ref, history = train_cfm(adata, n_epochs=50, device_str='cuda')
    adata_norm = normalize_adata_cfm(adata, model, scaler, ref, n_steps=20)
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Shared helpers from spancy_shift.py (Stage 1 + diagnostics)
from spancy_shift import (
    load_adata,
    log1p_scale,
    _get_col,
    find_best_sample_per_marker,
    shift_normalize_per_marker,
    detect_bimodal_markers,
    per_marker_batch_r2,
    positive_population_table,
)

log = logging.getLogger("spancy_cfm")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)
    log.propagate = False


# ──────────────────────────────────────────────────────────────────────────────
# Mini-batch OT coupling
# ──────────────────────────────────────────────────────────────────────────────

def ot_couple(
    x_src: np.ndarray,          # (N_src, M)
    x_ref: np.ndarray,          # (N_ref, M)
    n_samples: int = 256,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mini-batch optimal-transport coupling via Hungarian algorithm on L2 cost.

    Subsamples both sides to n_samples for computational tractability.
    Returns (x_src_paired, x_ref_paired, src_local_indices) — each length n_samples.
    The OT coupling minimises total squared displacement, so phenotype-similar
    cells get paired → biology-preserving transport.
    """
    rng = rng or np.random.RandomState()
    n = min(len(x_src), len(x_ref), n_samples)

    src_idx = rng.choice(len(x_src), n, replace=False)
    ref_idx = rng.choice(len(x_ref), n, replace=(len(x_ref) < n))
    xs = x_src[src_idx].astype(np.float32)
    xr = x_ref[ref_idx].astype(np.float32)

    C = cdist(xs, xr, metric="sqeuclidean")     # (n, n) cost matrix
    row_ind, col_ind = linear_sum_assignment(C)  # optimal assignment

    return xs[row_ind], xr[col_ind], src_idx[row_ind]


# ──────────────────────────────────────────────────────────────────────────────
# FlowMLP: velocity field network
# ──────────────────────────────────────────────────────────────────────────────

class _CondResBlock(nn.Module):
    """Residual block with AdaLN conditioning from batch + time embeddings."""

    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, 2 * hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + gamma) + beta
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h


class FlowMLP(nn.Module):
    """Velocity field for OT-CFM Stage 2.

    Inputs:
        x_t      (B, n_markers) — interpolated expression in RobustScaler-log1p space
        t        (B,) float  ∈ [0, 1]
        batch_ids (B,) int   — source batch codes

    Output:
        velocity (B, n_markers) — direction to integrate toward reference distribution
    """

    def __init__(
        self,
        n_markers: int = 20,
        n_batches: int = 7,
        hidden: int = 512,
        n_layers: int = 6,
        batch_emb_dim: int = 32,
        t_emb_dim: int = 64,
    ):
        super().__init__()
        cond_dim = batch_emb_dim + t_emb_dim
        self.batch_embed = nn.Embedding(n_batches, batch_emb_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_emb_dim), nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.input_proj = nn.Linear(n_markers, hidden)
        self.blocks = nn.ModuleList(
            [_CondResBlock(hidden, cond_dim) for _ in range(n_layers)]
        )
        self.output = nn.Linear(hidden, n_markers)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        b_emb = self.batch_embed(batch_ids)
        t_emb = self.t_embed(t.float().unsqueeze(-1))
        cond = torch.cat([b_emb, t_emb], dim=-1)
        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.output(h)


# ──────────────────────────────────────────────────────────────────────────────
# Batch-balanced sampler
# ──────────────────────────────────────────────────────────────────────────────

class BatchBalancedSampler:
    """Equal-sized random draws from each batch per step."""

    def __init__(
        self,
        batch_codes: np.ndarray,
        n_per_batch: int = 256,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.n_per_batch = n_per_batch
        self.unique_batches = np.unique(batch_codes)
        self.batch_idx: Dict[int, np.ndarray] = {
            int(b): np.where(batch_codes == b)[0] for b in self.unique_batches
        }
        cells_per_step = n_per_batch * len(self.unique_batches)
        self._n_steps = max(1, len(batch_codes) // cells_per_step)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (cell_indices, batch_codes) both (B,) numpy arrays."""
        idxs, bs = [], []
        for b in self.unique_batches:
            pool = self.batch_idx[int(b)]
            n = min(self.n_per_batch, len(pool))
            chosen = self.rng.choice(pool, size=n, replace=False)
            idxs.append(chosen)
            bs.append(np.full(n, b, dtype=np.int64))
        return np.concatenate(idxs), np.concatenate(bs)

    def __len__(self) -> int:
        return self._n_steps


# ──────────────────────────────────────────────────────────────────────────────
# Reference batch detection
# ──────────────────────────────────────────────────────────────────────────────

def _find_ref_batch(
    adata: ad.AnnData,
    ref_sample_per_marker: Dict[str, Optional[str]],
    batch_col: str,
) -> Tuple[int, str]:
    """Majority-vote: which batch contains the most marker-wise reference samples."""
    sample_col = _get_col(adata, ["sample_id", "sample"], batch_col)
    votes: Dict[str, int] = {}
    for ref_sample in ref_sample_per_marker.values():
        if ref_sample is None:
            continue
        mask = adata.obs[sample_col] == ref_sample if sample_col in adata.obs.columns \
            else adata.obs[batch_col] == ref_sample
        if mask.any():
            b = str(adata.obs.loc[mask, batch_col].iloc[0])
            votes[b] = votes.get(b, 0) + 1
    if not votes:
        # Fallback: use first batch
        first_batch = str(adata.obs[batch_col].astype("category").cat.categories[0])
        votes = {first_batch: 1}
    ref_name = max(votes, key=votes.get)
    categories = list(adata.obs[batch_col].astype("category").cat.categories)
    ref_code = categories.index(ref_name)
    log.info(
        "Reference batch: %s (code=%d). Votes: %s",
        ref_name, ref_code,
        dict(sorted(votes.items(), key=lambda kv: -kv[1])),
    )
    return ref_code, ref_name


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_cfm(
    adata: ad.AnnData,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cpu",
    warmup_epochs: int = 5,
    n_per_batch: int = 256,
    ot_samples: int = 256,
    sigma_min: float = 0.01,
    hidden: int = 512,
    n_layers: int = 6,
    batch_emb_dim: int = 32,
    t_emb_dim: int = 64,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
    ref_sample_per_marker: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[FlowMLP, RobustScaler, Dict[str, Optional[str]], Dict[str, List[float]]]:
    """Train OT-CFM Stage 2 model.

    Stage 1 (analytic): shift_normalize_per_marker → X_base (kBET ≈ 0.631).
    Stage 2 (OT-CFM): FlowMLP trained with MSE velocity loss + mini-batch OT coupling.
      Each step: sample source cells from all batches, sample reference cells,
      run OT coupling (256×256 Hungarian), interpolate at random t, predict velocity.

    Returns (model, scaler, ref_sample_per_marker, history).
    Pass all four to normalize_adata_cfm().
    """
    device = torch.device(device_str)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if ref_sample_per_marker is None:
        ref_sample_per_marker = find_best_sample_per_marker(adata)

    log.info("Stage 1: analytic shift normalization...")
    adata_base, _, _ = shift_normalize_per_marker(
        adata, ref_sample_per_marker,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
        layer_name="normalized_base",
    )

    # ── Stage 2 setup ─────────────────────────────────────────────────────────
    X_base = np.asarray(adata_base.layers["normalized_base"], dtype=np.float32)
    X_scaled, scaler = log1p_scale(X_base)

    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = adata.obs[batch_col].astype("category").cat.codes.values.astype(np.int64)
    n_batches = int(batch_codes.max()) + 1

    ref_code, ref_name = _find_ref_batch(adata, ref_sample_per_marker, batch_col)
    X_ref = X_scaled[batch_codes == ref_code]
    log.info(
        "Reference cells: %d / %d total (%.1f%%)",
        len(X_ref), len(X_scaled), 100 * len(X_ref) / len(X_scaled),
    )

    model = FlowMLP(
        n_markers=adata.n_vars,
        n_batches=n_batches,
        hidden=hidden,
        n_layers=n_layers,
        batch_emb_dim=batch_emb_dim,
        t_emb_dim=t_emb_dim,
    ).to(device)
    # Store reference info for inference
    model._ref_batch_code = ref_code
    model._ref_batch_name = ref_name
    log.info("FlowMLP: %d parameters", sum(p.numel() for p in model.parameters()))

    sampler = BatchBalancedSampler(batch_codes, n_per_batch=n_per_batch)
    steps_per_epoch = len(sampler)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                              milestones=[warmup_epochs])

    mse = nn.MSELoss()
    history: Dict[str, List[float]] = {"loss": [], "lr": []}
    rng = np.random.RandomState(42)

    log.info(
        "OT-CFM training: %d epochs × %d steps  "
        "n_per_batch=%d  ot_samples=%d  sigma_min=%.4f  hidden=%d  n_layers=%d  device=%s",
        n_epochs, steps_per_epoch, n_per_batch, ot_samples, sigma_min,
        hidden, n_layers, device_str,
    )

    for epoch in range(n_epochs):
        e_loss = 0.0
        model.train()

        for step in range(steps_per_epoch):
            src_idx, b_src = sampler.sample()   # (B,), (B,)
            x_src = X_scaled[src_idx]            # (B, M) source cells

            # OT coupling: pair source cells with reference cells
            x_0, x_1, row_keep = ot_couple(x_src, X_ref, n_samples=ot_samples, rng=rng)
            b_0 = b_src[row_keep].astype(np.int64)
            n_p = len(x_0)

            if n_p < 4:
                continue

            # Sample interpolation time t ~ U[0, 1]
            t_np = rng.uniform(0.0, 1.0, size=n_p).astype(np.float32)
            tc = t_np[:, None]

            # Straight-line interpolation + small Gaussian regularisation
            x_t_np = (1.0 - tc) * x_0 + tc * x_1
            x_t_np += (sigma_min * rng.randn(*x_t_np.shape)).astype(np.float32)

            # Target velocity: direction of the straight-line path
            v_np = x_1 - x_0  # (n_p, M)

            x_t = torch.tensor(x_t_np, device=device)
            t_t = torch.tensor(t_np, device=device)
            v_t = torch.tensor(v_np, device=device)
            b_t = torch.tensor(b_0, dtype=torch.long, device=device)

            v_pred = model(x_t, t_t, b_t)
            loss = mse(v_pred, v_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            e_loss += loss.item()

            if steps_per_epoch > 20 and (step + 1) % max(1, steps_per_epoch // 5) == 0:
                log.info(
                    "  E%3d step %d/%d  loss=%.6f",
                    epoch + 1, step + 1, steps_per_epoch, loss.item(),
                )

        scheduler.step()
        s = max(steps_per_epoch, 1)
        history["loss"].append(e_loss / s)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        log.info(
            "Epoch %3d/%d  loss=%.6f  lr=%.2e",
            epoch + 1, n_epochs, history["loss"][-1], history["lr"][-1],
        )

    log.info("Training complete. Final loss=%.6f", history["loss"][-1])
    return model, scaler, ref_sample_per_marker, history


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def normalize_adata_cfm(
    adata: ad.AnnData,
    model: FlowMLP,
    scaler: RobustScaler,
    ref_sample_per_marker: Dict[str, Optional[str]],
    n_steps: int = 20,
    inference_batch_size: int = 4096,
    device_str: str = "cpu",
    layer_name: str = "normalized",
    keep_base_layer: bool = True,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
) -> ad.AnnData:
    """Two-stage inference: Stage 1 (analytic) + Stage 2 (OT-CFM ODE integration).

    Stage 1: shift_normalize_per_marker → 'normalized_base' layer.
    Stage 2: Euler integration of FlowMLP from t=0 → t=1 conditioned on source batch.
      n_steps controls correction strength: fewer steps = more conservative.

    Reference-batch cells receive near-zero correction (already at target).
    """
    log.info("Inference Stage 1: analytic shift normalization...")
    adata_out, _, _ = shift_normalize_per_marker(
        adata, ref_sample_per_marker,
        min_prominence_frac=bimodal_prominence,
        bimodal_min_batch_frac=bimodal_min_batch_frac,
        layer_name="normalized_base",
    )

    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    X_base = np.asarray(adata_out.layers["normalized_base"], dtype=np.float32)
    X_scaled = scaler.transform(np.log1p(np.clip(X_base, 0, None))).astype(np.float32)

    batch_col = _get_col(adata, ["batch_id", "batch"], "batch_id")
    batch_codes = adata.obs[batch_col].astype("category").cat.codes.values.astype(np.int64)

    N = len(X_scaled)
    X_norm_scaled = np.empty_like(X_scaled)
    dt = 1.0 / max(n_steps, 1)
    n_chunks = max(1, (N + inference_batch_size - 1) // inference_batch_size)

    log.info(
        "Inference Stage 2: Euler ODE (%d steps, %d chunks, batch_size=%d)...",
        n_steps, n_chunks, inference_batch_size,
    )

    for ci, start in enumerate(range(0, N, inference_batch_size)):
        end = min(start + inference_batch_size, N)
        x = torch.tensor(X_scaled[start:end], device=device)
        b = torch.tensor(batch_codes[start:end], dtype=torch.long, device=device)

        # Euler ODE: t = 0 → 1
        for step in range(n_steps):
            t_val = step * dt
            t_t = torch.full((len(x),), t_val, device=device, dtype=torch.float32)
            v = model(x, t_t, b)
            x = x + dt * v

        X_norm_scaled[start:end] = x.cpu().numpy()

        if (ci + 1) % max(1, n_chunks // 10) == 0 or ci + 1 == n_chunks:
            log.info("  inference %d/%d chunks", ci + 1, n_chunks)

    X_norm_log = scaler.inverse_transform(X_norm_scaled)
    X_final = np.clip(np.expm1(X_norm_log), 0, None).astype(np.float32)
    adata_out.layers[layer_name] = X_final

    if not keep_base_layer:
        del adata_out.layers["normalized_base"]

    log.info(
        "Done. Layer '%s' written (n_steps=%d). min=%.4f  max=%.4f  mean=%.4f",
        layer_name, n_steps, X_final.min(), X_final.max(), X_final.mean(),
    )
    return adata_out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpaNCy-Shift CFM: OT-Conditional Flow Matching Stage 2"
    )
    parser.add_argument("--input", required=True, help="Input .h5ad")
    parser.add_argument("--output", required=True, help="Output .h5ad")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_per_batch", type=int, default=256,
                        help="Source cells per batch per training step")
    parser.add_argument("--ot_samples", type=int, default=256,
                        help="Cells per side for OT coupling (256×256 cost matrix)")
    parser.add_argument("--sigma_min", type=float, default=0.01,
                        help="Gaussian noise on interpolated path (path regularisation)")
    parser.add_argument("--n_steps", type=int, default=20,
                        help="Euler ODE steps at inference (more=stronger correction)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--bimodal_min_batch_frac", type=float, default=0.5)
    parser.add_argument("--layer_name", default="normalized")
    args = parser.parse_args()

    adata = load_adata(args.input)
    model, scaler, ref, history = train_cfm(
        adata,
        n_epochs=args.epochs,
        lr=args.lr,
        device_str=args.device,
        n_per_batch=args.n_per_batch,
        ot_samples=args.ot_samples,
        sigma_min=args.sigma_min,
        hidden=args.hidden,
        n_layers=args.n_layers,
        bimodal_min_batch_frac=args.bimodal_min_batch_frac,
    )
    adata_norm = normalize_adata_cfm(
        adata, model, scaler, ref,
        n_steps=args.n_steps,
        device_str=args.device,
        layer_name=args.layer_name,
    )
    adata_norm.write_h5ad(args.output)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
