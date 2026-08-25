# soil_dl_models_spatial.py
# Spatial field prediction for DoD project
# Revised: Soil_Temp_L1 only, grad features,
# EntropyTracker, full spatial field (B,N,T)

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

class EntropyTracker:
    def __init__(self, n_bins=50):
        self.n_bins  = n_bins
        self.history = []

    def compute(self, predictions, epoch):
        preds = predictions.flatten()
        preds = preds[~np.isnan(preds)]
        if len(preds) < 10:
            return np.nan
        counts, _ = np.histogram(
            preds, bins=self.n_bins)
        probs  = counts / (counts.sum() + 1e-10)
        probs  = probs[probs > 0]
        H      = float(
            -np.sum(probs * np.log(probs + 1e-10)))
        H_max  = np.log(self.n_bins)
        H_norm = H / (H_max + 1e-10)
        self.history.append({
            'epoch'    : epoch,
            'entropy'  : round(H,      4),
            'H_norm'   : round(H_norm, 4),
            'pred_std' : round(float(np.std(preds)), 4),
            'pred_mean': round(float(np.mean(preds)),4),
        })
        return H_norm

    def is_seasonal_fitting(self, window=5,
                             threshold=0.01):
        if len(self.history) < window + 1:
            return False
        recent = [h['H_norm']
                  for h in self.history[-window:]]
        return (max(recent) - min(recent)) < threshold

    def summary(self):
        if not self.history:
            return {}
        H_vals = [h['H_norm']
                  for h in self.history]
        return {
            'initial_H' : H_vals[0],
            'final_H'   : H_vals[-1],
            'max_H'     : max(H_vals),
            'delta_H'   : H_vals[-1] - H_vals[0],
            'plateaued' : self.is_seasonal_fitting(),
            'diagnosis' : (
                'SEASONAL_FITTING'
                if self.is_seasonal_fitting()
                else 'LEARNING_DYNAMICS'),
        }

def build_spatial_graph(df, k_neighbors=4):
    nodes = (df[['smap_node_x', 'smap_node_y']]
             .drop_duplicates()
             .sort_values(['smap_node_x',
                           'smap_node_y'])
             .reset_index(drop=True))
    node_coords = nodes.values.astype(np.float32)
    N           = len(nodes)
    node_ids    = [tuple(r) for r in node_coords]
    node_to_idx = {nid: i
                   for i, nid in
                   enumerate(node_ids)}
    tree = cKDTree(node_coords)
    dists, indices = tree.query(
        node_coords,
        k=min(k_neighbors + 1, N))
    sigma = np.median(dists[:, 1:]) + 1e-8
    A     = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j_pos in range(1, dists.shape[1]):
            j       = indices[i, j_pos]
            d       = dists[i, j_pos]
            w       = float(np.exp(-d / sigma))
            A[i, j] = w
            A[j, i] = w
    A      = A + np.eye(N)
    D_inv  = np.diag(
        1.0 / (A.sum(1) ** 0.5))
    A_norm = D_inv @ A @ D_inv
    print(f'  Graph: {N} nodes | '
          f'k={k_neighbors} | '
          f'sigma={sigma:.3f}')
    return (node_coords,
            torch.tensor(A_norm),
            node_ids,
            node_to_idx)

class SpatialGCN(nn.Module):
    def __init__(self, d_model,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model,
                      bias=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A_norm):
        for layer, norm in zip(
                self.layers, self.norms):
            msg = torch.einsum(
                'nm,bmd->bnd',
                A_norm.to(H.device),
                self.drop(layer(H)))
            H   = norm(torch.nn.functional.gelu(msg) + H)
        return H

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16,
                 d_conv=4, expand=2,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2,
            bias=False)
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size = d_conv,
    print('  Graph: ' + str(N) + ' nodes | k=' + str(k_neighbors) + ' | sigma=' + str(round(sigma, 3)))
            groups      = self.d_inner,
            bias        = True)
        self.act     = nn.SiLU()
        self.x_proj  = nn.Linear(
            self.d_inner,
            d_state * 2 + self.d_inner,
            bias=False)
        self.dt_proj = nn.Linear(
            self.d_inner, self.d_inner,
            bias=True)
        A = torch.arange(
            1, d_state + 1,
            dtype=torch.float32
        ).unsqueeze(0).repeat(
            self.d_inner, 1)
        self.A_log   = nn.Parameter(
            torch.log(A))
        self.D       = nn.Parameter(
            torch.ones(self.d_inner))
        self.out_proj= nn.Linear(
            self.d_inner, d_model,
            bias=False)
        self.drop    = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def ssm_scan(self, x):
        B, L, D  = x.shape
        S        = self.d_state
        x_dbl    = self.x_proj(x)
        delta, B_p, C = x_dbl.split(
            [D, S, S], dim=-1)
        delta = F.softplus(
            self.dt_proj(delta))
        A     = -torch.exp(
            self.A_log.float())
        dA    = torch.exp(
            torch.einsum(
                'bld,ds->blds', delta, A))
        dB    = torch.einsum(
            'bld,bls->blds', delta, B_p)
        h     = torch.zeros(
            B, D, S,
            device=x.device,
            dtype=x.dtype)
        ys = []
        for i in range(L):
            h  = (dA[:, i] * h
                  + dB[:, i]
                  * x[:, i, :, None])
            y  = torch.einsum(
                'bds,bs->bd',
                h, C[:, i, :])
            ys.append(y)
        return torch.stack(ys, dim=1) * self.D

    def forward(self, x):
        residual = x
        xz       = self.in_proj(x)
        x_, z    = xz.chunk(2, dim=-1)
        x_       = x_.transpose(1, 2)
        x_       = self.conv1d(x_)[
            ..., :x.shape[1]]
        x_       = x_.transpose(1, 2)
        x_       = self.act(x_)
        y        = self.ssm_scan(x_)
        y        = y * self.act(z)
        y        = self.out_proj(
            self.drop(y))
        return self.norm(residual + y)

class SpatialBiGRU(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim=128,
                 n_layers=2, n_heads=4,
                 n_nodes=256, gnn_layers=2,
                 n_targets=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(
            input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = (
                dropout if n_layers > 1
                else 0.0))
        self.attn  = nn.MultiheadAttention(
            hidden_dim * 2, n_heads,
            dropout=dropout,
            batch_first=True)
        self.norm1 = nn.LayerNorm(
            hidden_dim * 2)
        self.norm2 = nn.LayerNorm(
            hidden_dim * 2)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim * 2,
                      hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4,
                      hidden_dim * 2),
        )
        self.gcn = SpatialGCN(
            hidden_dim * 2,
            gnn_layers, dropout)
        self.node_emb = nn.Embedding(
            n_nodes, hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4,
                      hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2,
                      n_targets),
        )

    def forward(self, x, A_norm):
        B, N, L, F = x.shape
        x_flat = x.reshape(B * N, L, F)
        x_proj = self.input_proj(x_flat)
        h, _   = self.gru(x_proj)
        a, _   = self.attn(h, h, h)
        h      = self.norm1(h + a)
        h      = self.norm2(
            h + self.ffn(h))
        h_last = h[:, -1, :].reshape(
            B, N, -1)
        node_idx = torch.arange(
            N, device=x.device).clamp(
            max=self.node_emb
            .num_embeddings - 1)
        h_last   = (
            h_last
            + self.node_emb(node_idx)
            .unsqueeze(0))
        h_spatial = self.gcn(
            h_last, A_norm)
        h_cat = torch.cat(
            [h_last, h_spatial], dim=-1)
        return self.head(h_cat)

class SpatialMamba(nn.Module):
    def __init__(self, input_dim,
                 d_model=128, n_layers=4,
                 d_state=16, d_conv=4,
                 n_nodes=256, gnn_layers=2,
                 n_targets=1, dropout=0.1):
        super().__init__()
        self.embed  = nn.Linear(
            input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state,
                       d_conv,
                       dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.gcn      = SpatialGCN(
            d_model, gnn_layers, dropout)
        self.node_emb = nn.Embedding(
            n_nodes, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, x, A_norm):
        B, N, L, F = x.shape
        x_flat = x.reshape(B * N, L, F)
        h      = self.embed(x_flat)
        for blk in self.blocks:
            h = blk(h)
        h_last = self.norm(
            h[:, -1, :]).reshape(B, N, -1)
        node_idx = torch.arange(
            N, device=x.device).clamp(
            max=self.node_emb
            .num_embeddings - 1)
        h_last   = (
            h_last
            + self.node_emb(node_idx)
            .unsqueeze(0))
        h_spatial = self.gcn(
            h_last, A_norm)
        h_cat = torch.cat(
            [h_last, h_spatial], dim=-1)
        return self.head(h_cat)

class S4Layer(nn.Module):
    def __init__(self, d_model, d_state=64,
                 dropout=0.1,
                 bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidir   = bidirectional

        def hippo_legs(N):
            A = torch.zeros(N, N)
            for n in range(N):
                for m in range(n):
                    A[n, m] = (
                        -(2*n+1)**0.5
                        * (2*m+1)**0.5)
                A[n, n] = -(n + 1)
            return A

        A = hippo_legs(d_state)
        self.A = nn.Parameter(
            A, requires_grad=False)
        self.B = nn.Parameter(
            torch.randn(d_state, 1) * 0.01)
        self.C = nn.Parameter(
            torch.randn(d_model, d_state))
        self.D = nn.Parameter(
            torch.ones(d_model))
        self.log_dt = nn.Parameter(
            torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(
            d_model, d_model)
        dirs      = 2 if bidirectional else 1
        self.mix  = nn.Linear(
            d_model * dirs, d_model)

    def _scan(self, u):
        B_seq, L, d = u.shape
        dA  = torch.matrix_exp(self.A)
        dB  = self.B.squeeze(-1)
        h   = torch.zeros(
            B_seq, d, self.d_state,
            device=u.device)
        ys  = []
        for t in range(L):
            ut = u[:, t, :]
            h  = (h @ dA.T
                  + ut.unsqueeze(-1) * dB)
            y  = (h * self.C
                  .unsqueeze(0)).sum(-1)
            ys.append(y + self.D * ut)
        return torch.stack(ys, dim=1)

    def forward(self, x):
        if self.bidir:
            yf = self._scan(x)
            yr = self._scan(
                x.flip(1)).flip(1)
            y  = self.mix(torch.cat(
                [yf, yr], dim=-1))
        else:
            y = self._scan(x)
        return self.norm(
            x + self.drop(self.out(y)))


class SpatialS4(nn.Module):
    def __init__(self, input_dim,
                 d_model=128, n_layers=4,
                 d_state=64, n_nodes=256,
                 gnn_layers=2,
                 n_targets=1,
                 dropout=0.1):
        super().__init__()
        self.embed  = nn.Linear(
            input_dim, d_model)
        self.layers = nn.ModuleList([
            S4Layer(d_model, d_state,
                    dropout=dropout,
                    bidirectional=True)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.gcn      = SpatialGCN(
            d_model, gnn_layers, dropout)
        self.node_emb = nn.Embedding(
            n_nodes, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, x, A_norm):
        B, N, L, F = x.shape
        x_flat = x.reshape(B * N, L, F)
        h      = self.embed(x_flat)
        for layer in self.layers:
            h = layer(h)
        h_last = self.norm(
            h[:, -1, :]).reshape(B, N, -1)
        node_idx = torch.arange(
            N, device=x.device).clamp(
            max=self.node_emb
            .num_embeddings - 1)
        h_last   = (
            h_last
            + self.node_emb(node_idx)
            .unsqueeze(0))
        h_spatial = self.gcn(
            h_last, A_norm)
        h_cat = torch.cat(
            [h_last, h_spatial], dim=-1)
        return self.head(h_cat)

class SpatialFuseMoE(nn.Module):
    def __init__(self, input_dim,
                 d_model=128,
                 n_experts=4, top_k=2,
                 d_state=16,
                 n_ssm_layers=2,
                 n_nodes=256,
                 gnn_layers=2,
                 n_targets=1,
                 dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.d_model   = d_model
        self.embed = nn.Linear(
            input_dim, d_model)
        self.expert_mamba1 = MambaBlock(
            d_model, d_state,
            dropout=dropout)
        self.expert_gru = nn.GRU(
            d_model, d_model,
            batch_first=True)
        self.expert_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model,
                      kernel_size=7,
                      padding=3,
                      groups=d_model),
            nn.Conv1d(d_model, d_model,
                      kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1))
        self.expert_mamba2 = MambaBlock(
            d_model, d_state,
            dropout=dropout)
        self.expert_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(d_model,
                      d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2,
                      n_experts),
        )
        self.ssm_backbone = nn.ModuleList([
            MambaBlock(d_model, d_state,
                       dropout=dropout)
            for _ in range(n_ssm_layers)
        ])
        self.gcn = SpatialGCN(
            d_model, gnn_layers, dropout)
        self.node_emb = nn.Embedding(
            n_nodes, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )
        self.aux_loss = torch.tensor(0.0)

    def _run_experts(self, h):
        outs = []
        h0 = self.expert_mamba1(h)
        outs.append(self.expert_norms[0](
            h0.mean(dim=1)))
        _, h1 = self.expert_gru(h)
        outs.append(self.expert_norms[1](
            h1[-1]))
        h2 = self.expert_cnn(
            h.transpose(1, 2)
        ).squeeze(-1)
        outs.append(self.expert_norms[2](h2))
        h3 = self.expert_mamba2(h)
        outs.append(self.expert_norms[3](
            h3.mean(dim=1)))
        return outs

    def forward(self, x, A_norm):
        B, N, L, F = x.shape
        x_flat = x.reshape(B * N, L, F)
        h      = self.embed(x_flat)
        h_pool = h.mean(dim=1)
        logits = self.gate(h_pool)
        top_vals, top_idx = logits.topk(
            self.top_k, dim=-1)
        gate_scores = torch.nn.functional.softmax(
            top_vals, dim=-1)
        gate_soft  = torch.nn.functional.softmax(
            logits, dim=-1)
        importance = gate_soft.mean(dim=0)
        load = (gate_soft > (
            1.0 / self.n_experts)
        ).float().mean(dim=0)
        self.aux_loss = (
            (importance * load).sum()
            * self.n_experts)
        expert_outs = self._run_experts(h)
        E_stack = torch.stack(
            expert_outs, dim=1)
        selected = torch.gather(
            E_stack, 1,
            top_idx.unsqueeze(-1).expand(
                -1, -1, self.d_model))
        fused = (
            selected
            * gate_scores.unsqueeze(-1)
        ).sum(dim=1)
        fused_seq = (
            fused.unsqueeze(1)
            .expand(-1, L, -1) + h)
        for blk in self.ssm_backbone:
            fused_seq = blk(fused_seq)
        h_last = self.norm(
            fused_seq[:, -1, :]
        ).reshape(B, N, -1)
        node_idx = torch.arange(
            N, device=x.device).clamp(
            max=self.node_emb
            .num_embeddings - 1)
        h_last = (
            h_last
            + self.node_emb(node_idx)
            .unsqueeze(0))
        h_spatial = self.gcn(
            h_last, A_norm)
        h_cat = torch.cat(
            [h_last, h_spatial], dim=-1)
        return self.head(h_cat)

SPATIAL_DL_MODELS = [
    'SpatialBiGRU',
    'SpatialMamba',
    'SpatialS4',
    'SpatialFuseMoE',
]


def make_spatial_model(arch, n_targets,
                        n_features,
                        n_nodes):
    cfg = {
        'SpatialBiGRU': dict(
            input_dim=n_features,
            hidden_dim=128, n_layers=2,
            n_heads=4, n_nodes=n_nodes,
            gnn_layers=2,
            n_targets=n_targets,
            dropout=0.1),
        'SpatialMamba': dict(
            input_dim=n_features,
            d_model=128, n_layers=4,
            d_state=16, d_conv=4,
            n_nodes=n_nodes,
            gnn_layers=2,
            n_targets=n_targets,
            dropout=0.1),
        'SpatialS4': dict(
            input_dim=n_features,
            d_model=128, n_layers=4,
            d_state=64,
            n_nodes=n_nodes,
            gnn_layers=2,
            n_targets=n_targets,
            dropout=0.1),
        'SpatialFuseMoE': dict(
            input_dim=n_features,
            d_model=128, n_experts=4,
            top_k=2, d_state=16,
            n_ssm_layers=2,
            n_nodes=n_nodes,
            gnn_layers=2,
            n_targets=n_targets,
            dropout=0.1),
    }
    cls_map = {
        'SpatialBiGRU'  : SpatialBiGRU,
        'SpatialMamba'  : SpatialMamba,
        'SpatialS4'     : SpatialS4,
        'SpatialFuseMoE': SpatialFuseMoE,
    }
    return cls_map[arch](**cfg[arch])


def count_spatial_params(model):
    total = sum(
        p.numel()
        for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad)
    return total, trainable


def huber_loss(pred, target, delta=1.0):
    diff = pred - target
    abs_ = diff.abs()
    return torch.where(
        abs_ <= delta,
        0.5 * diff ** 2,
        delta * (abs_ - 0.5 * delta)
    ).mean()


def spatial_smoothness_loss(pred, A_norm):
    pred_smooth = torch.einsum(
        'nm,bmt->bnt',
        A_norm.to(pred.device), pred)
    return torch.nn.functional.mse_loss(pred, pred_smooth)


def combined_spatial_loss(
        pred, target, A_norm,
        delta=1.0,
        lambda_spatial=0.05,
        lambda_aux=0.01,
        aux_loss=None):
    l_huber   = huber_loss(
        pred, target, delta)
    l_spatial = spatial_smoothness_loss(
        pred, A_norm)
    total     = (l_huber
                 + lambda_spatial * l_spatial)
    if aux_loss is not None:
        total = total + lambda_aux * aux_loss
    return total, {
        'huber'  : l_huber.item(),
        'spatial': l_spatial.item(),
        'aux'    : (
            aux_loss.item()
            if aux_loss is not None
            else 0.0),
    }


def train_epoch_spatial(
        model, loader, opt,
        scheduler, scaler_amp,
        device, arch,
        lambda_spatial=0.05,
        lambda_aux=0.01):
    model.train()
    totals = {
        'total':0.0, 'huber':0.0,
        'spatial':0.0, 'aux':0.0}
    n_batches = 0
    use_amp   = (device.type == 'cuda')

    for batch in loader:
        (X, y_res,
         y_app, y_raw,
         A_norm) = [
            b.to(device) for b in batch]
        opt.zero_grad()
        with torch.cuda.amp.autocast(
                enabled=use_amp):
            pred     = model(X, A_norm)
            aux_loss = (
                model.aux_loss
                if arch == 'SpatialFuseMoE'
                and hasattr(model,
                            'aux_loss')
                else None)
            loss, components = (
                combined_spatial_loss(
                    pred, y_res, A_norm,
                    lambda_spatial=(
                        lambda_spatial),
                    lambda_aux=lambda_aux,
                    aux_loss=aux_loss))
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(opt)
        nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        scaler_amp.step(opt)
        scaler_amp.update()
        if scheduler is not None:
            scheduler.step()
        totals['total']  += loss.item()
        totals['huber']  += (
            components['huber'])
        totals['spatial']+= (
            components['spatial'])
        totals['aux']    += (
            components['aux'])
        n_batches += 1

    nb = max(n_batches, 1)
    return {k: round(v / nb, 6)
            for k, v in totals.items()}


@torch.no_grad()
def eval_epoch_spatial(
        model, loader, device, arch,
        tgt_sc_dict,
        entropy_tracker=None,
        epoch=0):
    model.eval()
    all_true   = []
    all_pred   = []
    all_approx = []
    use_amp    = (device.type == 'cuda')
    tgt_sc = list(
        tgt_sc_dict.values())[0]

    for batch in loader:
        (X, y_res,
         y_app, y_raw,
         A_norm) = [
            b.to(device) for b in batch]
        with torch.cuda.amp.autocast(
                enabled=use_amp):
            pred_sc = model(X, A_norm)
        B, N, T  = pred_sc.shape
        pred_np  = (
            pred_sc.cpu().float().numpy())
        app_np   = (
            y_app.cpu().float().numpy())
        raw_np   = (
            y_raw.cpu().float().numpy())
        pred_r   = tgt_sc.inverse_transform(
            pred_np.reshape(-1, T)
        ).reshape(B, N, T)
        all_true.append(raw_np)
        all_pred.append(app_np + pred_r)
        all_approx.append(app_np)

    y_true   = np.concatenate(
        all_true,   axis=0)
    y_pred   = np.concatenate(
        all_pred,   axis=0)
    y_approx = np.concatenate(
        all_approx, axis=0)

    yt = y_true[:,  :, 0].flatten()
    yp = y_pred[:,  :, 0].flatten()
    ya = y_approx[:, :, 0].flatten()
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt = yt[mask]
    yp = yp[mask]
    ya = ya[mask]

    if len(yt) < 10:
        return {
            'R2':0.0, 'RMSE':999.0,
            'MAE':999.0, 'Skill':-1.0,
            'KGE':-1.0, 'H_norm':0.0,
            'node_r2_mean':0.0,
            'node_r2_min':0.0,
            'node_r2_std':0.0,
            'spatial_var_ratio':0.0}

    rmse  = float(np.sqrt(
        np.mean((yt - yp) ** 2)))
    mae   = float(
        np.mean(np.abs(yt - yp)))
    ss    = float(
        np.sum((yt - yp) ** 2))
    st    = float(np.sum(
        (yt - yt.mean()) ** 2))
    r2    = float(
        1.0 - ss / (st + 1e-10))
    skill = float(1.0 - (
        np.nanmean((yt - yp) ** 2) /
        (np.nanmean(
            (yt - ya) ** 2) + 1e-10)))
    r     = float(
        np.corrcoef(yt, yp)[0, 1])
    alpha = float(
        np.std(yp) /
        (np.std(yt) + 1e-10))
    beta  = float(
        np.mean(yp) /
        (np.mean(yt) + 1e-10))
    kge   = float(1 - np.sqrt(
        (r-1)**2
        + (alpha-1)**2
        + (beta-1)**2))

    n_nodes = y_true.shape[1]
    node_r2 = []
    for n in range(n_nodes):
        yt_n = y_true[:, n, 0]
        yp_n = y_pred[:, n, 0]
        mk   = ~(np.isnan(yt_n) |
                  np.isnan(yp_n))
        if mk.sum() < 5:
            node_r2.append(np.nan)
            continue
        ss_n = np.sum(
            (yt_n[mk] - yp_n[mk]) ** 2)
        st_n = np.sum(
            (yt_n[mk]
             - yt_n[mk].mean()) ** 2)
        node_r2.append(float(
            1.0 - ss_n /
            (st_n + 1e-10)))

    H_norm = np.nan
    if entropy_tracker is not None:
        H_norm = entropy_tracker.compute(
            yp, epoch)

    pred_sv = float(np.nanvar(
        y_pred[:, :, 0], axis=1).mean())
    true_sv = float(np.nanvar(
        y_true[:, :, 0], axis=1).mean())
    sv_ratio = float(
        pred_sv / (true_sv + 1e-10))

    return {
        'R2'   : round(r2,    4),
        'RMSE' : round(rmse,  4),
        'MAE'  : round(mae,   4),
        'Skill': round(skill, 4),
        'KGE'  : round(kge,   4),
        'H_norm': round(float(
            H_norm
            if not np.isnan(H_norm)
            else 0.0), 4),
        'node_r2_mean': round(float(
            np.nanmean(node_r2)), 4),
        'node_r2_min' : round(float(
            np.nanmin(node_r2)), 4),
        'node_r2_std' : round(float(
            np.nanstd(node_r2)), 4),
        'spatial_var_ratio': round(
            sv_ratio, 4),
    }


def train_spatial_model(
        arch, n_targets,
        n_features, n_nodes,
        train_loader, val_loader,
        tgt_sc_dict,
        epochs=30, lr=3e-4,
        patience=7,
        lambda_spatial=0.05,
        lambda_aux=0.01,
        model_dir=Path('./models/dl'),
        ckpt_name=None):

    DEVICE = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu')

    model = make_spatial_model(
        arch, n_targets,
        n_features, n_nodes
    ).to(DEVICE)

    opt = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=lr,
        weight_decay=1e-4)

    n_steps = epochs * len(train_loader)
    sched   = OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=n_steps,
        pct_start=0.1,
        anneal_strategy='cos')

    amp_sc = torch.cuda.amp.GradScaler(
        enabled=(
            torch.cuda.is_available()))

    entropy_tracker = EntropyTracker(
        n_bins=50)
    best_val_r2  = -np.inf
    best_state   = None
    patience_cnt = 0
    history      = []
    t_start      = time.time()

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad)
    print(
        f'  Training {arch} | '
        f'{n_params:,} params | '
        f'{epochs} epochs | '
        f'nodes={n_nodes} | '
        f'device={DEVICE}')

    for epoch in range(1, epochs + 1):
        tr = train_epoch_spatial(
            model, train_loader,
            opt, sched, amp_sc,
            DEVICE, arch,
            lambda_spatial=lambda_spatial,
            lambda_aux=lambda_aux)

        val_m = eval_epoch_spatial(
            model, val_loader,
            DEVICE, arch,
            tgt_sc_dict,
            entropy_tracker=(
                entropy_tracker),
            epoch=epoch)

        history.append({
            'epoch'        : epoch,
            'train_total'  : tr['total'],
            'train_huber'  : tr['huber'],
            'train_spatial': tr['spatial'],
            **{f'val_{k}': v
               for k, v in val_m.items()}
        })

        if val_m['R2'] > best_val_r2:
            best_val_r2  = val_m['R2']
            best_state   = {
                k: v.cpu().clone()
                for k, v in
                model.state_dict()
                .items()
            }
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            diag    = (
                'SEASONAL'
                if entropy_tracker
                   .is_seasonal_fitting()
                else 'LEARNING')
            print(
                f'    E{epoch:03d} | '
                f'loss='
                f'{tr["total"]:.4f} '
                f'(h={tr["huber"]:.3f} '
                f's='
                f'{tr["spatial"]:.3f}) | '
                f'R2={val_m["R2"]:.4f} | '
                f'Skill='
                f'{val_m["Skill"]:.4f} | '
                f'H='
                f'{val_m["H_norm"]:.3f}'
                f'[{diag}] | '
                f'NodeR2='
                f'{val_m["node_r2_mean"]:.4f}'
                f' | {elapsed:.0f}s')

        if patience_cnt >= patience:
            print(
                f'    Early stop '
                f'@ epoch {epoch}')
            break

    elapsed = time.time() - t_start
    e_summ  = entropy_tracker.summary()
    diag    = e_summ.get(
        'diagnosis', 'UNKNOWN')

    print('  Physics Diagnosis:')
    print(
        f'    Initial H : '
        f'{e_summ.get("initial_H",0):.4f}')
    print(
        f'    Final H   : '
        f'{e_summ.get("final_H",0):.4f}')
    print(
        f'    Delta H   : '
        f'{e_summ.get("delta_H",0):.4f}')
    print(f'    Diagnosis : {diag}')
    if diag == 'SEASONAL_FITTING':
        print(
            '    WARNING: seasonal fitting.')
        print(
            '    Try: increase '
            'lambda_spatial, '
            'reduce lookback.')
    else:
        print(
            '    OK: learning '
            'physical dynamics.')

    if best_state:
        model.load_state_dict(best_state)
    print('  Training ' + arch + ' | ' + str(n_params) + ' params | ' + str(epochs) + ' epochs | nodes=' + str(n_nodes) + ' | device=' + str(DEVICE))
    save_name = (
        ckpt_name
        or f'{arch}_spatial_best.pt')
    ckpt_path = (
        Path(model_dir) / save_name)
    torch.save({
        'arch'           : arch,
        'state_dict'     : best_state,
        'val_r2'         : best_val_r2,
        'history'        : history,
        'epochs_run'     : epoch,
        'elapsed_s'      : elapsed,
        'entropy_summary': e_summ,
        'n_nodes'        : n_nodes,
        'n_features'     : n_features,
        'n_targets'      : n_targets,
    }, ckpt_path)

    print(
        f'  Saved: '
        f'val_r2={best_val_r2:.4f} | '
        f'time={elapsed:.0f}s | '
        f'{ckpt_path.name}')

    return (
        model, history,
        best_val_r2, elapsed)
