#!/usr/bin/env python
"""
SpaNCy-Shift DDPM: Denoising Diffusion Probabilistic Model for CyCIF batch normalization.

Stage 1 (analytic, identical to spancy_shift.py):
    shift_normalize_per_marker() → X_base  [kBET ≈ 0.631]

Stage 2 (DDPM + SDEdit):
    DenoisingMLP conditioned on batch learns p(x | batch) — the per-batch distribution.

    Training:
        Standard DDPM noise prediction (Ho et al. 2020).
        Classifier-free guidance: 10% of steps drop the batch label → null token.
        Loss: MSE between predicted and actual added noise.
        No bimodal masking — the score function learns the full joint distribution
        including bimodal geometry (ECAD, etc.) naturally.

    Inference (SDEdit, Song et al. 2021):
        1. Start from X_base (Stage 1 output, already aligned in 1D).
        2. Add partial noise up to timestep t_infer.
        3. Run DDPM reverse loop from t_infer → 0, conditioned on reference batch
           via classifier-free guidance: eps = eps_uncond + s*(eps_ref - eps_uncond).
        4. Output corrects 20D covariance structure while preserving large-scale signal.

    t_infer controls correction strength (T=200 schedule):
        - t_infer=10:  very conservative, ~5% noise added. Near-Stage-1 output.
        - t_infer=30:  moderate. ~20% noise. Recommended starting point.
        - t_infer=80:  aggressive. ~50% noise. Risk of bimodal distortion.

    cfg_scale controls guidance strength toward reference batch:
        - 1.0: no extra guidance (pure conditional sampling)
        - 1.5: moderate (recommended)
        - 3.0: strong push; may over-correct

Usage:
    model, scheduler, scaler, ref, history = train_ddpm(adata, n_epochs=50, device_str='cuda')
    adata_norm = normalize_adata_ddpm(adata, model, scheduler, scaler, ref,
                                       t_infer=30, cfg_scale=1.5)
"""

import argparse
import logging
import math
import sys
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
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

log = logging.getLogger("spancy_ddpm")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)
    log.propagate = False


# ──────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ──────────────────────────────────────────────────────────────────────────────

class DDPMScheduler:
    """Linear beta schedule with precomputed forward-process statistics.

    T=200 steps is sufficient for 20D tabular data (images typically use T=1000).
    beta_start=1e-4, beta_end=0.02 → alpha_bar_T ≈ 0.12 (nearly full noise at T).
    """

    def __init__(
        self,
        T: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.T = T
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        alphas_bar_prev = np.concatenate([[1.0], alphas_bar[:-1]])

        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        posterior_var[0] = betas[0]  # avoid division by near-zero at t=1

        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alphas_bar = torch.tensor(alphas_bar, dtype=torch.float32)
        self.sqrt_alphas_bar = torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32)
        self.sqrt_one_minus_alphas_bar = torch.tensor(np.sqrt(1.0 - alphas_bar), dtype=torch.float32)
        self.sqrt_recip_alphas = torch.tensor(np.sqrt(1.0 / alphas), dtype=torch.float32)
        self.posterior_var = torch.tensor(posterior_var, dtype=torch.float32)
        self.posterior_log_var = torch.tensor(np.log(np.maximum(posterior_var, 1e-20)), dtype=torch.float32)

    def to(self, device: torch.device) -> "DDPMScheduler":
        for attr in ("betas", "alphas", "alphas_bar", "sqrt_alphas_bar",
                     "sqrt_one_minus_alphas_bar", "sqrt_recip_alphas",
                     "posterior_var", "posterior_log_var"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward process: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*eps."""
        if eps is None:
            eps = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alphas_bar[t - 1].unsqueeze(-1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t - 1].unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_1mab * eps, eps

    def p_step(
        self,
        x_t: torch.Tensor,
        t_int: int,
        eps_pred: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """Single DDPM reverse step: x_{t-1} from x_t and predicted eps.

        x_{t-1} = (1/sqrt(α_t)) * (x_t - (1-α_t)/sqrt(1-ᾱ_t) * eps_pred)
                  + sqrt(β̃_t) * z
        """
        alpha_t = self.alphas[t_int - 1]
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t_int - 1]
        recip_sqrt_alpha = self.sqrt_recip_alphas[t_int - 1]
        coeff = (1.0 - alpha_t) / sqrt_1mab

        mean = recip_sqrt_alpha * (x_t - coeff * eps_pred)

        if add_noise and t_int > 1:
            z = torch.randn_like(x_t)
            std = (0.5 * self.posterior_log_var[t_int - 1]).exp()
            return mean + std * z
        return mean


# ──────────────────────────────────────────────────────────────────────────────
# DenoisingMLP architecture
# ──────────────────────────────────────────────────────────────────────────────

class _SinusoidalEmbed(nn.Module):
    """Sinusoidal time-step embedding (fixed, no learnable params)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class _AdaLNResBlock(nn.Module):
    """Residual block with Adaptive Layer Norm conditioned on time+batch embedding."""

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


class DenoisingMLP(nn.Module):
    """Score network for DDPM.

    Predicts the noise eps added to x_0 at timestep t, conditioned on batch.
    Classifier-free guidance: batch_id == n_batches → null token (unconditional).

    Inputs:
        x_noisy  (B, n_markers) — noisy expression in RobustScaler-log1p space
        t        (B,) int  ∈ {1, ..., T}
        batch_ids (B,) int  ∈ {0, ..., n_batches} (n_batches = null token)

    Output:
        eps_pred (B, n_markers) — predicted noise
    """

    def __init__(
        self,
        n_markers: int = 20,
        n_batches: int = 7,
        hidden: int = 512,
        n_layers: int = 6,
        time_dim: int = 256,
        batch_emb_dim: int = 32,
    ):
        super().__init__()
        cond_dim = time_dim + batch_emb_dim
        # n_batches+1 embeddings: 0..n_batches-1 are real batches, n_batches is null token
        self.batch_embed = nn.Embedding(n_batches + 1, batch_emb_dim)
        self.time_embed = nn.Sequential(
            _SinusoidalEmbed(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Linear(n_markers, hidden)
        self.blocks = nn.ModuleList(
            [_AdaLNResBlock(hidden, cond_dim) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, n_markers)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)                      # (B, time_dim)
        b_emb = self.batch_embed(batch_ids)              # (B, batch_emb_dim)
        cond = torch.cat([t_emb, b_emb], dim=-1)         # (B, cond_dim)
        h = self.input_proj(x_noisy)
        for block in self.blocks:
            h = block(h, cond)
        return self.output(self.out_norm(h))


# ──────────────────────────────────────────────────────────────────────────────
# Batch-balanced sampler (same as CFM version)
# ──────────────────────────────────────────────────────────────────────────────

class BatchBalancedSampler:
    """Equal-sized random draws from each batch per step."""

    def __init__(
        self,
        batch_codes: np.ndarray,
        n_per_batch: int = 512,
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
# Reference batch detection (reused logic from CFM)
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
        first = str(adata.obs[batch_col].astype("category").cat.categories[0])
        votes = {first: 1}
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

def train_ddpm(
    adata: ad.AnnData,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cpu",
    warmup_epochs: int = 5,
    n_per_batch: int = 512,
    T: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    cfg_dropout: float = 0.1,
    hidden: int = 512,
    n_layers: int = 6,
    time_dim: int = 256,
    batch_emb_dim: int = 32,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
    ref_sample_per_marker: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[DenoisingMLP, DDPMScheduler, RobustScaler, Dict[str, Optional[str]], Dict[str, List[float]]]:
    """Train DDPM Stage 2 model.

    Stage 1 (analytic): shift_normalize_per_marker → X_base (kBET ≈ 0.631).
    Stage 2 (DDPM): DenoisingMLP trained with standard MSE noise-prediction loss.
      Classifier-free guidance: cfg_dropout fraction of steps use null batch token.

    Returns (model, scheduler, scaler, ref_sample_per_marker, history).
    Pass all five to normalize_adata_ddpm().
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

    scheduler = DDPMScheduler(T=T, beta_start=beta_start, beta_end=beta_end)
    scheduler.to(device)

    model = DenoisingMLP(
        n_markers=adata.n_vars,
        n_batches=n_batches,
        hidden=hidden,
        n_layers=n_layers,
        time_dim=time_dim,
        batch_emb_dim=batch_emb_dim,
    ).to(device)
    model._ref_batch_code = ref_code
    model._ref_batch_name = ref_name
    model._n_batches = n_batches
    log.info("DenoisingMLP: %d parameters", sum(p.numel() for p in model.parameters()))

    sampler = BatchBalancedSampler(batch_codes, n_per_batch=n_per_batch)
    steps_per_epoch = len(sampler)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs))
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_epochs])

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    batch_t = torch.tensor(batch_codes, dtype=torch.long)

    history: Dict[str, List[float]] = {"loss": [], "lr": []}
    rng_np = np.random.RandomState(0)

    log.info(
        "DDPM training: %d epochs × %d steps  "
        "T=%d  cfg_dropout=%.2f  hidden=%d  n_layers=%d  device=%s",
        n_epochs, steps_per_epoch, T, cfg_dropout, hidden, n_layers, device_str,
    )

    for epoch in range(n_epochs):
        e_loss = 0.0
        model.train()

        for step in range(steps_per_epoch):
            src_idx, b_src = sampler.sample()
            x_0 = X_t[src_idx].to(device)             # (B, M) clean
            b_ids = torch.tensor(b_src, dtype=torch.long, device=device)

            # Random timesteps t ~ Uniform(1, T)
            t = torch.randint(1, T + 1, (len(x_0),), device=device)

            # Forward process: add noise
            x_noisy, eps = scheduler.q_sample(x_0, t)

            # Classifier-free guidance: replace batch label with null token for cfg_dropout fraction
            cfg_mask = torch.rand(len(b_ids), device=device) < cfg_dropout
            b_ids_dropped = b_ids.clone()
            b_ids_dropped[cfg_mask] = n_batches  # null token index

            eps_pred = model(x_noisy, t, b_ids_dropped)
            loss = F.mse_loss(eps_pred, eps)

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

        lr_scheduler.step()
        s = max(steps_per_epoch, 1)
        history["loss"].append(e_loss / s)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        log.info(
            "Epoch %3d/%d  loss=%.6f  lr=%.2e",
            epoch + 1, n_epochs, history["loss"][-1], history["lr"][-1],
        )

    log.info("Training complete. Final loss=%.6f", history["loss"][-1])
    return model, scheduler, scaler, ref_sample_per_marker, history


# ──────────────────────────────────────────────────────────────────────────────
# Inference (SDEdit)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def normalize_adata_ddpm(
    adata: ad.AnnData,
    model: DenoisingMLP,
    scheduler: DDPMScheduler,
    scaler: RobustScaler,
    ref_sample_per_marker: Dict[str, Optional[str]],
    t_infer: int = 30,
    cfg_scale: float = 1.5,
    inference_batch_size: int = 4096,
    device_str: str = "cpu",
    layer_name: str = "normalized",
    keep_base_layer: bool = True,
    bimodal_prominence: float = 0.05,
    bimodal_min_batch_frac: float = 0.5,
) -> ad.AnnData:
    """SDEdit inference: Stage 1 + DDPM reverse diffusion toward reference batch.

    Stage 1: shift_normalize_per_marker → 'normalized_base'.
    Stage 2: Add noise to x_0 at t_infer, reverse-diffuse conditioned on ref batch.
      Classifier-free guidance: eps = eps_uncond + cfg_scale*(eps_ref - eps_uncond).

    t_infer: partial noise timestep (1..T). Lower = more conservative, higher = stronger.
    cfg_scale: guidance strength. 1.0 = no guidance, 1.5 = recommended, 3.0 = aggressive.
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
    scheduler.to(device)

    T = scheduler.T
    if t_infer < 1 or t_infer > T:
        raise ValueError(f"t_infer must be in [1, {T}], got {t_infer}")

    n_batches = model._n_batches
    ref_code = model._ref_batch_code

    X_base = np.asarray(adata_out.layers["normalized_base"], dtype=np.float32)
    X_scaled = scaler.transform(np.log1p(np.clip(X_base, 0, None))).astype(np.float32)

    N = len(X_scaled)
    X_norm_scaled = np.empty_like(X_scaled)
    n_chunks = max(1, (N + inference_batch_size - 1) // inference_batch_size)

    # Precompute forward-process stats for t_infer (on device)
    sqrt_ab_tinfer = scheduler.sqrt_alphas_bar[t_infer - 1]
    sqrt_1mab_tinfer = scheduler.sqrt_one_minus_alphas_bar[t_infer - 1]

    log.info(
        "Inference Stage 2: SDEdit t_infer=%d/%d  cfg_scale=%.1f  "
        "ref_batch=%s  (%d chunks)...",
        t_infer, T, cfg_scale, model._ref_batch_name, n_chunks,
    )
    log.info(
        "  Noise fraction at t_infer=%d: signal=%.3f  noise=%.3f",
        t_infer, sqrt_ab_tinfer.item(), sqrt_1mab_tinfer.item(),
    )

    for ci, start in enumerate(range(0, N, inference_batch_size)):
        end = min(start + inference_batch_size, N)
        x_0 = torch.tensor(X_scaled[start:end], device=device)  # (B, M)
        B = len(x_0)

        # SDEdit step 1: add partial noise to t_infer
        eps_init = torch.randn_like(x_0)
        x_t = sqrt_ab_tinfer * x_0 + sqrt_1mab_tinfer * eps_init

        # SDEdit step 2: reverse DDPM from t_infer → 1
        ref_ids = torch.full((B,), ref_code, dtype=torch.long, device=device)
        null_ids = torch.full((B,), n_batches, dtype=torch.long, device=device)

        for t_step in range(t_infer, 0, -1):
            t_tensor = torch.full((B,), t_step, dtype=torch.long, device=device)

            if cfg_scale == 1.0:
                # Pure conditional — no guidance overhead
                eps_pred = model(x_t, t_tensor, ref_ids)
            else:
                # Classifier-free guidance: two forward passes
                eps_cond   = model(x_t, t_tensor, ref_ids)
                eps_uncond = model(x_t, t_tensor, null_ids)
                eps_pred   = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            x_t = scheduler.p_step(x_t, t_step, eps_pred, add_noise=(t_step > 1))

        X_norm_scaled[start:end] = x_t.cpu().numpy()

        if (ci + 1) % max(1, n_chunks // 10) == 0 or ci + 1 == n_chunks:
            log.info("  inference %d/%d chunks", ci + 1, n_chunks)

    X_norm_log = scaler.inverse_transform(X_norm_scaled)
    X_final = np.clip(np.expm1(X_norm_log), 0, None).astype(np.float32)
    adata_out.layers[layer_name] = X_final

    if not keep_base_layer:
        del adata_out.layers["normalized_base"]

    log.info(
        "Done. Layer '%s' written (t_infer=%d, cfg_scale=%.1f). "
        "min=%.4f  max=%.4f  mean=%.4f",
        layer_name, t_infer, cfg_scale,
        X_final.min(), X_final.max(), X_final.mean(),
    )
    return adata_out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpaNCy-Shift DDPM: Denoising Diffusion Stage 2 (SDEdit inference)"
    )
    parser.add_argument("--input", required=True, help="Input .h5ad")
    parser.add_argument("--output", required=True, help="Output .h5ad")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_per_batch", type=int, default=512,
                        help="Cells per batch per training step")
    parser.add_argument("--T", type=int, default=200,
                        help="Total diffusion timesteps")
    parser.add_argument("--cfg_dropout", type=float, default=0.1,
                        help="Fraction of training steps using null batch token (CFG training)")
    parser.add_argument("--t_infer", type=int, default=30,
                        help="SDEdit noise level at inference (1..T). Lower=conservative.")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="CFG guidance strength at inference (1.0=no guidance)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--bimodal_min_batch_frac", type=float, default=0.5)
    parser.add_argument("--layer_name", default="normalized")
    args = parser.parse_args()

    adata = load_adata(args.input)
    model, scheduler, scaler, ref, history = train_ddpm(
        adata,
        n_epochs=args.epochs,
        lr=args.lr,
        device_str=args.device,
        n_per_batch=args.n_per_batch,
        T=args.T,
        cfg_dropout=args.cfg_dropout,
        hidden=args.hidden,
        n_layers=args.n_layers,
        bimodal_min_batch_frac=args.bimodal_min_batch_frac,
    )
    adata_norm = normalize_adata_ddpm(
        adata, model, scheduler, scaler, ref,
        t_infer=args.t_infer,
        cfg_scale=args.cfg_scale,
        device_str=args.device,
        layer_name=args.layer_name,
    )
    adata_norm.write_h5ad(args.output)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
