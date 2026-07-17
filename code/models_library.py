"""
Model zoo -- all classes share the same interface:
    forward(x): x is (batch, seq_len, n_features) -> returns (batch, n_targets)

This lets one training harness run any of them interchangeably.

Includes: LSTM, BiLSTM, GRU, BiGRU, TCN, PatchTST (lite), Informer (lite),
TFT (lite). S4D and Mamba live in their own files (s4_model.py,
mamba_model.py) and share this same interface -- import them alongside
these for the full roster.

"Lite" on PatchTST/Informer/TFT means: the core mechanism from each paper
(patching, prob-sparse-style attention approximated with standard attention
plus a sparsity-inducing top-k mask, and gated variable selection,
respectively) is implemented, but not every architectural detail from the
original papers -- these are practical, readable versions meant to run
reliably on HPC without exotic dependencies, not exact paper replicas.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _head(d_model: int, n_targets: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model // 2, n_targets),
    )


# ---------------------------------------------------------------------------
# Recurrent family: LSTM, BiLSTM, GRU, BiGRU
# ---------------------------------------------------------------------------

class _RNNModel(nn.Module):
    def __init__(self, n_features, n_targets, cell="lstm", bidirectional=False,
                 hidden_size=128, n_layers=2, dropout=0.1):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = _head(out_dim, n_targets, dropout)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class LSTMModel(_RNNModel):
    def __init__(self, n_features, n_targets, **kw):
        super().__init__(n_features, n_targets, cell="lstm", bidirectional=False, **kw)


class BiLSTMModel(_RNNModel):
    def __init__(self, n_features, n_targets, **kw):
        super().__init__(n_features, n_targets, cell="lstm", bidirectional=True, **kw)


class GRUModel(_RNNModel):
    def __init__(self, n_features, n_targets, **kw):
        super().__init__(n_features, n_targets, cell="gru", bidirectional=False, **kw)


class BiGRUModel(_RNNModel):
    def __init__(self, n_features, n_targets, **kw):
        super().__init__(n_features, n_targets, cell="gru", bidirectional=True, **kw)


# ---------------------------------------------------------------------------
# TCN -- dilated causal 1D convolutions
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.padding = padding

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = F.gelu(self.norm1(self.conv1(x)[..., :-self.padding or None]))
        out = self.dropout(out)
        out = F.gelu(self.norm2(self.conv2(out)[..., :-self.padding or None]))
        out = self.dropout(out)
        return out + residual


class TCNModel(nn.Module):
    def __init__(self, n_features, n_targets, channels=(64, 64, 128, 128),
                 kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = _head(channels[-1], n_targets, dropout)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))
        h_last = h[:, :, -1]
        return self.head(h_last)


# ---------------------------------------------------------------------------
# Shared positional encoding for the transformer-family models
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# PatchTST (lite)
# ---------------------------------------------------------------------------

class PatchTSTModel(nn.Module):
    def __init__(self, n_features, n_targets, patch_len=4, d_model=128,
                 n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = _head(d_model, n_targets, dropout)

    def forward(self, x):
        b, l, f = x.shape
        n_patches = l // self.patch_len
        usable = n_patches * self.patch_len
        x = x[:, -usable:, :]
        patches = x.reshape(b, n_patches, self.patch_len * f)
        h = self.patch_proj(patches)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.head(h[:, -1, :])


# ---------------------------------------------------------------------------
# Informer (lite)
# ---------------------------------------------------------------------------

class _ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, top_k_ratio=0.5, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k_ratio = top_k_ratio
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        top_k = max(1, int(l * self.top_k_ratio))
        if top_k < l:
            topk_vals, _ = scores.topk(top_k, dim=-1)
            threshold = topk_vals[..., -1:].detach()
            scores = scores.masked_fill(scores < threshold, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.out_proj(out)


class _InformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = _ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class InformerModel(nn.Module):
    def __init__(self, n_features, n_targets, d_model=128, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        self.layers = nn.ModuleList([_InformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.head = _head(d_model, n_targets, dropout)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.pos_enc(h)
        for layer in self.layers:
            h = layer(h)
        return self.head(h[:, -1, :])


# ---------------------------------------------------------------------------
# TFT (lite)
# ---------------------------------------------------------------------------

class _GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(self.skip(x) + g * h)


class TFTModel(nn.Module):
    def __init__(self, n_features, n_targets, d_model=128, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.var_selection = nn.Sequential(
            nn.Linear(n_features, n_features), nn.Softmax(dim=-1)
        )
        self.feature_grn = _GatedResidualNetwork(n_features, d_model, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.final_grn = _GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.head = _head(d_model, n_targets, dropout)

    def forward(self, x):
        weights = self.var_selection(x)
        x_weighted = x * weights
        h = self.feature_grn(x_weighted)
        out, _ = self.lstm(h)
        out = self.final_grn(out)
        return self.head(out[:, -1, :])


MODEL_REGISTRY = {
    "LSTM": LSTMModel,
    "BiLSTM": BiLSTMModel,
    "GRU": GRUModel,
    "BiGRU": BiGRUModel,
    "TCN": TCNModel,
    "PatchTST": PatchTSTModel,
    "Informer": InformerModel,
    "TFT": TFTModel,
}


if __name__ == "__main__":
    x = torch.randn(4, 24, 16)
    for name, cls in MODEL_REGISTRY.items():
        model = cls(n_features=16, n_targets=2)
        out = model(x)
        assert out.shape == (4, 2), f"{name} produced wrong shape {out.shape}"
        print(f"{name}: OK, output shape {tuple(out.shape)}")