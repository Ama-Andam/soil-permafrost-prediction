
# ============================================================
# soil_dl_models.py
# All DL model classes + training engine
# Must live at: /home/emmanuel.keku/soil_dl_models.py
# ============================================================

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim           import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib               import Path

warnings.filterwarnings("ignore")


# ============================================================
# 1. BiGRU + Multi-Head Attention
# ============================================================
class BiGRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,
                 n_layers=2, n_heads=4,
                 n_targets=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.0,
        )
        self.attn  = nn.MultiheadAttention(
            hidden_dim * 2, n_heads,
            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
        )
        self.head = nn.Linear(hidden_dim * 2, n_targets)

    def forward(self, x):
        x    = self.input_proj(x)
        h, _ = self.gru(x)
        a, _ = self.attn(h, h, h)
        h    = self.norm1(h + a)
        h    = self.norm2(h + self.ffn(h))
        return self.head(h[:, -1, :])


# ============================================================
# 2. Mamba SSM Block
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16,
                 d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size = d_conv,
            padding     = d_conv - 1,
            groups      = self.d_inner,
            bias        = True)
        self.act     = nn.SiLU()
        self.x_proj  = nn.Linear(
            self.d_inner,
            d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(
            self.d_inner, self.d_inner, bias=True)
        A = torch.arange(
            1, d_state + 1,
            dtype=torch.float32
        ).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(
            torch.ones(self.d_inner))
        self.out_proj= nn.Linear(
            self.d_inner, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def ssm_scan(self, x):
        B, L, D = x.shape
        S       = self.d_state
        x_dbl   = self.x_proj(x)
        delta, B_p, C = x_dbl.split(
            [D, S, S], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A     = -torch.exp(self.A_log.float())
        dA    = torch.exp(
            torch.einsum("bld,ds->blds", delta, A))
        dB    = torch.einsum(
            "bld,bls->blds", delta, B_p)
        h     = torch.zeros(
            B, D, S,
            device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = (dA[:, i] * h
                 + dB[:, i] * x[:, i, :, None])
            y = torch.einsum(
                "bds,bs->bd", h, C[:, i, :])
            ys.append(y)
        return torch.stack(ys, dim=1) * self.D

    def forward(self, x):
        residual = x
        xz       = self.in_proj(x)
        x_, z    = xz.chunk(2, dim=-1)
        x_       = x_.transpose(1, 2)
        x_       = self.conv1d(x_)[..., :x.shape[1]]
        x_       = x_.transpose(1, 2)
        x_       = self.act(x_)
        y        = self.ssm_scan(x_)
        y        = y * self.act(z)
        y        = self.out_proj(self.drop(y))
        return self.norm(residual + y)


class MambaModel(nn.Module):
    def __init__(self, input_dim, d_model=128,
                 n_layers=4, d_state=16, d_conv=4,
                 n_targets=1, dropout=0.1):
        super().__init__()
        self.embed  = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state,
                       d_conv, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, -1, :]))


# ============================================================
# 3. Deep Echo State Network
# ============================================================
class DeepESN(nn.Module):
    def __init__(self, input_dim,
                 reservoir_dim=512, n_layers=3,
                 spectral_radius=0.9,
                 leaking_rate=0.3, sparsity=0.1,
                 n_targets=1, dropout=0.1):
        super().__init__()
        self.n_layers      = n_layers
        self.reservoir_dim = reservoir_dim
        self.leaking_rate  = leaking_rate
        self.W_in  = nn.ParameterList()
        self.W_res = nn.ParameterList()
        for i in range(n_layers):
            in_d = input_dim if i == 0 else reservoir_dim
            self.W_in.append(nn.Parameter(
                torch.randn(reservoir_dim, in_d) * 0.1,
                requires_grad=False))
            self.W_res.append(
                self._init_reservoir(
                    reservoir_dim,
                    spectral_radius,
                    sparsity))
        self.drop    = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(
            reservoir_dim * n_layers)
        self.readout = nn.Sequential(
            nn.Linear(reservoir_dim * n_layers, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
        )

    @staticmethod
    def _init_reservoir(dim, rho, sparsity):
        W    = torch.randn(dim, dim)
        mask = torch.rand(dim, dim) > sparsity
        W[mask] = 0.0
        try:
            eigs = torch.linalg.eigvals(W).abs()
            sr   = eigs.max().item()
            if sr > 1e-8:
                W = W * (rho / sr)
        except Exception:
            W = W * rho
        return nn.Parameter(W, requires_grad=False)

    def forward(self, x):
        B, L, _ = x.shape
        h_prev   = [
            torch.zeros(B, self.reservoir_dim,
                        device=x.device)
            for _ in range(self.n_layers)
        ]
        for t in range(L):
            xt = x[:, t, :]
            for i in range(self.n_layers):
                inp = xt if i == 0 else h_prev[i-1]
                pre = (inp @ self.W_in[i].T
                       + h_prev[i] @ self.W_res[i].T)
                h   = ((1 - self.leaking_rate)
                       * h_prev[i]
                       + self.leaking_rate
                       * torch.tanh(pre))
                h_prev[i] = h
        h_cat = torch.cat(h_prev, dim=-1)
        return self.readout(
            self.norm(self.drop(h_cat)))


# ============================================================
# 4. ST-GNN
# ============================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim,
                               bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, H, A_norm):
        return self.act(
            self.norm(A_norm @ self.W(self.drop(H))))


class STGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,
                 n_nodes=4, gnn_layers=2,
                 gru_layers=2, n_targets=1,
                 dropout=0.1):
        super().__init__()
        self.n_nodes  = n_nodes
        self.temporal = nn.GRU(
            input_dim, hidden_dim,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = (dropout
                           if gru_layers > 1
                           else 0.0))
        self.site_emb = nn.Embedding(n_nodes,
                                      hidden_dim)
        self.gcn      = nn.ModuleList([
            GraphConvLayer(hidden_dim,
                           hidden_dim, dropout)
            for _ in range(gnn_layers)
        ])
        A_init       = (torch.ones(n_nodes, n_nodes)
                        - torch.eye(n_nodes))
        self.A_raw   = nn.Parameter(A_init)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _norm_adj(self):
        A = torch.relu(self.A_raw)
        A = A + A.T
        A = A + torch.eye(self.n_nodes,
                           device=A.device)
        D = A.sum(-1).pow(-0.5)
        return D.unsqueeze(-1) * A * D.unsqueeze(-2)

    def forward(self, x_nodes):
        B, N, L, F = x_nodes.shape
        x_flat      = x_nodes.reshape(B * N, L, F)
        _, h        = self.temporal(x_flat)
        h_last      = h[-1].reshape(B, N, -1)
        idx         = torch.arange(
            N, device=x_nodes.device)
        h_s         = (h_last
                       + self.site_emb(idx)
                       .unsqueeze(0))
        A_n         = self._norm_adj()
        hg          = h_s
        for gcn_layer in self.gcn:
            hg = torch.stack([
                gcn_layer(hg[b], A_n)
                for b in range(B)
            ])
        h_cat = torch.cat([h_s, hg], dim=-1)
        return self.readout(h_cat)


# ============================================================
# 5. S4 SSM
# ============================================================
class S4Layer(nn.Module):
    def __init__(self, d_model, d_state=64,
                 dropout=0.1, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidir   = bidirectional

        def hippo_legs(N):
            A = torch.zeros(N, N)
            for n in range(N):
                for m in range(n):
                    A[n, m] = (-(2*n+1)**0.5
                                * (2*m+1)**0.5)
                A[n, n] = -(n + 1)
            return A

        A = hippo_legs(d_state)
        self.A      = nn.Parameter(A,
                                    requires_grad=False)
        self.B      = nn.Parameter(
            torch.randn(d_state, 1) * 0.01)
        self.C      = nn.Parameter(
            torch.randn(d_model, d_state))
        self.D      = nn.Parameter(
            torch.ones(d_model))
        self.log_dt = nn.Parameter(
            torch.zeros(d_model))
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)
        self.out    = nn.Linear(d_model, d_model)
        dirs        = 2 if bidirectional else 1
        self.mix    = nn.Linear(
            d_model * dirs, d_model)

    def _scan(self, u):
        B_seq, L, d = u.shape
        dA  = torch.matrix_exp(self.A)
        dB  = self.B.squeeze(-1)
        h   = torch.zeros(B_seq, d,
                          self.d_state,
                          device=u.device)
        ys  = []
        for t in range(L):
            ut = u[:, t, :]
            h  = h @ dA.T + ut.unsqueeze(-1) * dB
            y  = (h * self.C.unsqueeze(0)).sum(-1)
            ys.append(y + self.D * ut)
        return torch.stack(ys, dim=1)

    def forward(self, x):
        if self.bidir:
            yf = self._scan(x)
            yr = self._scan(x.flip(1)).flip(1)
            y  = self.mix(
                torch.cat([yf, yr], dim=-1))
        else:
            y = self._scan(x)
        return self.norm(x + self.drop(self.out(y)))


class S4SSMModel(nn.Module):
    def __init__(self, input_dim, d_model=128,
                 n_layers=4, d_state=64,
                 n_targets=1, dropout=0.1):
        super().__init__()
        self.embed  = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            S4Layer(d_model, d_state,
                    dropout=dropout,
                    bidirectional=True)
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x[:, -1, :]))


# ============================================================
# 6. FuseMoE
# ============================================================
class ExpertSSM(nn.Module):
    def __init__(self, d_model, d_state=16,
                 dropout=0.1):
        super().__init__()
        self.block = MambaBlock(
            d_model, d_state=d_state,
            dropout=dropout)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = self.block(x)
        return self.pool(
            h.transpose(1, 2)).squeeze(-1)


class ExpertGRU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.gru  = nn.GRU(
            d_model, d_model,
            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        _, h = self.gru(x)
        return self.norm(h[-1])


class ExpertCNN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model,
                      kernel_size=7, padding=3,
                      groups=d_model),
            nn.Conv1d(d_model, d_model,
                      kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(
            self.net(x.transpose(1, 2)).squeeze(-1))


class SpatialExpert(nn.Module):
    def __init__(self, d_model, coord_dim=4,
                 dropout=0.1):
        super().__init__()
        self.coord_proj = nn.Linear(coord_dim,
                                     d_model)
        self.gru        = nn.GRU(
            d_model, d_model, batch_first=True)
        self.fusion     = nn.Linear(
            d_model * 2, d_model)
        self.norm       = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x, coords):
        _, h   = self.gru(x)
        h_last = h[-1]
        c_emb  = self.coord_proj(coords)
        fused  = self.fusion(
            torch.cat([h_last, c_emb], dim=-1))
        return self.norm(self.drop(fused))


class FuseMoE(nn.Module):
    def __init__(self, input_dim, d_model=128,
                 n_experts=4, top_k=2,
                 d_state=16, n_ssm_layers=2,
                 n_targets=1, dropout=0.1,
                 coord_dim=4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.d_model   = d_model
        self.embed     = nn.Linear(input_dim,
                                    d_model)
        self.experts   = nn.ModuleList([
            ExpertSSM(d_model, d_state, dropout),
            ExpertGRU(d_model, dropout),
            ExpertCNN(d_model, dropout),
            SpatialExpert(d_model,
                          coord_dim, dropout),
        ])
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_experts),
        )
        self.ssm_backbone = nn.ModuleList([
            MambaBlock(d_model,
                       d_state=d_state,
                       dropout=dropout)
            for _ in range(n_ssm_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x, coords=None):
        if coords is None:
            coords = torch.zeros(
                x.shape[0], 4, device=x.device)
        h           = self.embed(x)
        h_pool      = h.mean(dim=1)
        logits      = self.gate(h_pool)
        top_vals, top_idx = logits.topk(
            self.top_k, dim=-1)
        gate_scores = F.softmax(top_vals, dim=-1)
        gate_soft   = F.softmax(logits, dim=-1)
        importance  = gate_soft.mean(dim=0)
        load        = (gate_soft > (
            1.0 / self.n_experts)
        ).float().mean(dim=0)
        aux_loss    = ((importance * load).sum()
                       * self.n_experts)
        expert_outs = []
        for i, expert in enumerate(self.experts):
            if isinstance(expert, SpatialExpert):
                out = expert(h, coords)
            else:
                out = expert(h)
            expert_outs.append(out)
        E_stack  = torch.stack(expert_outs, dim=1)
        selected = torch.gather(
            E_stack, 1,
            top_idx.unsqueeze(-1).expand(
                -1, -1, self.d_model))
        fused    = (selected
                    * gate_scores.unsqueeze(-1)
                    ).sum(dim=1)
        fused_seq = (fused.unsqueeze(1)
                     .expand(-1, x.shape[1], -1)
                     + h)
        for blk in self.ssm_backbone:
            fused_seq = blk(fused_seq)
        out = self.head(
            self.norm(fused_seq[:, -1, :]))
        return out, aux_loss


# ============================================================
# Model factory
# ============================================================
def make_model(arch, n_targets,
               n_features=36, n_sites=4):
    cfg = {
        "BiGRU"  : dict(
            input_dim=n_features,
            hidden_dim=128, n_layers=2,
            n_heads=4, n_targets=n_targets,
            dropout=0.1),
        "Mamba"  : dict(
            input_dim=n_features,
            d_model=128, n_layers=4,
            d_state=16, d_conv=4,
            n_targets=n_targets, dropout=0.1),
        "DeepESN": dict(
            input_dim=n_features,
            reservoir_dim=512, n_layers=3,
            spectral_radius=0.9,
            n_targets=n_targets, dropout=0.1),
        "STGNN"  : dict(
            input_dim=n_features,
            hidden_dim=128, n_nodes=n_sites,
            gnn_layers=2, gru_layers=2,
            n_targets=n_targets, dropout=0.1),
        "S4SSM"  : dict(
            input_dim=n_features,
            d_model=128, n_layers=4,
            d_state=64, n_targets=n_targets,
            dropout=0.1),
        "FuseMoE": dict(
            input_dim=n_features,
            d_model=128, n_experts=4,
            top_k=2, d_state=16,
            n_ssm_layers=2, coord_dim=4,
            n_targets=n_targets, dropout=0.1),
    }
    cls_map = {
        "BiGRU"  : BiGRUAttention,
        "Mamba"  : MambaModel,
        "DeepESN": DeepESN,
        "STGNN"  : STGNN,
        "S4SSM"  : S4SSMModel,
        "FuseMoE": FuseMoE,
    }
    return cls_map[arch](**cfg[arch])


def count_params(model):
    total     = sum(
        p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad)
    return total, trainable


# ============================================================
# Loss
# ============================================================
def huber_loss(pred, target, delta=1.0):
    diff = pred - target
    abs_ = diff.abs()
    return torch.where(
        abs_ <= delta,
        0.5 * diff ** 2,
        delta * (abs_ - 0.5 * delta)
    ).mean()


# ============================================================
# Training epoch
# ============================================================
def train_epoch(model, loader, opt,
                scheduler, scaler_amp,
                device, arch,
                aux_weight=0.01):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    use_amp    = (device.type == "cuda")

    for batch in loader:
        X, y_res, y_app, y_raw, coords = [
            b.to(device) for b in batch]
        opt.zero_grad()

        with torch.cuda.amp.autocast(
                enabled=use_amp):
            if arch == "STGNN":
                pred = model(
                    X.unsqueeze(1))[:, 0, :]
                aux  = torch.tensor(
                    0.0, device=device)
            elif arch == "FuseMoE":
                pred, aux = model(X, coords)
            else:
                pred = model(X)
                aux  = torch.tensor(
                    0.0, device=device)
            loss = (huber_loss(pred, y_res)
                    + aux_weight * aux)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(opt)
        nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        scaler_amp.step(opt)
        scaler_amp.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# Evaluation epoch
# ============================================================
@torch.no_grad()
def eval_epoch(model, loader, device,
               arch, tgt_sc):
    model.eval()
    all_true   = []
    all_pred   = []
    all_approx = []
    use_amp    = (device.type == "cuda")

    for batch in loader:
        X, y_res, y_app, y_raw, coords = [
            b.to(device) for b in batch]

        with torch.cuda.amp.autocast(
                enabled=use_amp):
            if arch == "STGNN":
                pred_sc = model(
                    X.unsqueeze(1))[:, 0, :]
            elif arch == "FuseMoE":
                pred_sc, _ = model(X, coords)
            else:
                pred_sc = model(X)

        pred_np   = pred_sc.cpu().float().numpy()
        pred_r    = tgt_sc.inverse_transform(
            pred_np)
        app_np    = y_app.cpu().float().numpy()
        raw_np    = y_raw.cpu().float().numpy()
        pred_full = app_np + pred_r

        all_true.append(raw_np)
        all_pred.append(pred_full)
        all_approx.append(app_np)

    y_true   = np.concatenate(all_true,   axis=0)
    y_pred   = np.concatenate(all_pred,   axis=0)
    y_approx = np.concatenate(all_approx, axis=0)

    yt   = y_true[:, 0]
    yp   = y_pred[:, 0]
    ya   = y_approx[:, 0]
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp, ya = yt[mask], yp[mask], ya[mask]

    if len(yt) < 10:
        return {"R2": 0.0, "RMSE": 999.0,
                "MAE": 999.0, "Skill": -1.0,
                "KGE": -1.0}

    rmse  = float(np.sqrt(
        np.mean((yt - yp) ** 2)))
    mae   = float(np.mean(np.abs(yt - yp)))
    ss    = float(np.sum((yt - yp) ** 2))
    st    = float(np.sum(
        (yt - yt.mean()) ** 2))
    r2    = float(1.0 - ss / (st + 1e-10))
    skill = float(1.0 - (
        np.nanmean((yt - yp) ** 2) /
        (np.nanmean((yt - ya) ** 2) + 1e-10)))
    r     = float(np.corrcoef(yt, yp)[0, 1])
    alpha = float(
        np.std(yp) / (np.std(yt) + 1e-10))
    beta  = float(
        np.mean(yp) / (np.mean(yt) + 1e-10))
    kge   = float(1 - np.sqrt(
        (r-1)**2 + (alpha-1)**2 + (beta-1)**2))

    return {
        "R2"   : round(r2,    4),
        "RMSE" : round(rmse,  4),
        "MAE"  : round(mae,   4),
        "Skill": round(skill, 4),
        "KGE"  : round(kge,   4),
    }


# ============================================================
# Full training loop
# ============================================================
def train_model(arch, n_targets,
                train_loader, val_loader,
                tgt_sc,
                epochs    = 30,
                lr        = 1e-3,
                patience  = 7,
                model_dir = Path("./models/dl"),
                ckpt_name = None):

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu")

    model = make_model(arch, n_targets).to(DEVICE)
    opt   = AdamW(
        filter(lambda p: p.requires_grad,
               model.parameters()),
        lr=lr, weight_decay=1e-4)

    n_steps = epochs * len(train_loader)
    sched   = OneCycleLR(
        opt,
        max_lr          = lr,
        total_steps     = n_steps,
        pct_start       = 0.1,
        anneal_strategy = "cos")

    amp_scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available())

    best_val_r2  = -np.inf
    best_state   = None
    patience_cnt = 0
    history      = []
    t_start      = time.time()

    n_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad)
    print(f"  Training {arch} | "
          f"{n_params:,} params | "
          f"{epochs} epochs | "
          f"device={DEVICE}")

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(
            model, train_loader,
            opt, sched, amp_scaler,
            DEVICE, arch)

        val_m = eval_epoch(
            model, val_loader,
            DEVICE, arch, tgt_sc)

        history.append({
            "epoch"      : epoch,
            "train_loss" : round(tr_loss, 6),
            **{f"val_{k}": v
               for k, v in val_m.items()}
        })

        if val_m["R2"] > best_val_r2:
            best_val_r2  = val_m["R2"]
            best_state   = {
                k: v.cpu().clone()
                for k, v in
                model.state_dict().items()
            }
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed_now = time.time() - t_start
            print(f"    E{epoch:03d} | "
                  f"loss={tr_loss:.4f} | "
                  f"R2={val_m['R2']:.4f} | "
                  f"Skill="
                  f"{val_m['Skill']:.4f} | "
                  f"RMSE="
                  f"{val_m['RMSE']:.4f} | "
                  f"elapsed="
                  f"{elapsed_now:.0f}s")

        if patience_cnt >= patience:
            print(f"    Early stop "
                  f"@ epoch {epoch}")
            break

    elapsed = time.time() - t_start

    if best_state:
        model.load_state_dict(best_state)

    save_name = ckpt_name or f"{arch}_best.pt"
    ckpt_path = Path(model_dir) / save_name
    torch.save({
        "arch"      : arch,
        "state_dict": best_state,
        "val_r2"    : best_val_r2,
        "history"   : history,
        "epochs_run": epoch,
        "elapsed_s" : elapsed,
    }, ckpt_path)

    print(f"  ✓ val R2={best_val_r2:.4f} | "
          f"time={elapsed:.0f}s | "
          f"saved → {ckpt_path.name}")

    return model, history, best_val_r2, elapsed
