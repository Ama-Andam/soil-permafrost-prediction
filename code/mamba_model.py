"""
Minimal Mamba block -- pure PyTorch, no custom CUDA kernels (no
`mamba-ssm`/`causal-conv1d` dependency).

Mamba's key difference from S4: the state-space parameters (B, C, and the
step size dt) are input-dependent ("selective"), computed from the input at
each timestep rather than fixed. This lets the model selectively remember
or forget information based on content, which is closer to what an
attention mechanism does, while keeping linear-time sequential recurrence
(here implemented as an explicit scan, since we don't have the fused CUDA
selective-scan kernel).

This will be slower than the official CUDA implementation (the explicit
Python-level scan over timesteps is the bottleneck), but it will run
anywhere your other PyTorch models run -- no kernel compilation, no CUDA
version matching against TALON's toolchain. If `mamba-ssm` is confirmed
installable later, swapping in the official selective_scan_fn is a speed
optimization, not an interface change.

Fits the same (batch, seq_len, n_features) -> (batch, n_targets) convention
as the other models in the roster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape

        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :l].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbl = self.x_proj(x_conv)
        delta_in, Bmat, Cmat = torch.split(
            x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_in))

        A = -torch.exp(self.A_log)

        y = self._selective_scan(x_conv, delta, A, Bmat, Cmat)
        y = y + x_conv * self.D
        y = y * F.silu(res)

        return self.out_proj(y)

    @staticmethod
    def _selective_scan(u, delta, A, B, C):
        """
        Explicit sequential scan (not fused/parallelized). This is the part
        the official CUDA kernel accelerates; here it's a plain Python loop
        over timesteps, which is the main throughput cost of this pure-
        PyTorch version.

        u:     (batch, seq_len, d_inner)
        delta: (batch, seq_len, d_inner)
        A:     (d_inner, d_state)
        B, C:  (batch, seq_len, d_state)
        """
        b, l, d_inner = u.shape
        d_state = A.shape[1]
        device = u.device

        deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_inner, d_state))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        state = torch.zeros(b, d_inner, d_state, device=device)
        ys = []
        for t in range(l):
            state = deltaA[:, t] * state + deltaB_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", state, C[:, t])
            ys.append(y_t)
        return torch.stack(ys, dim=1)


class MambaModel(nn.Module):
    """
    Stack of Mamba blocks for the soil sequence-to-target regression task.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_targets)
    """

    def __init__(self, n_features: int, n_targets: int, d_model: int = 128,
                 n_layers: int = 4, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + self.dropout(block(norm(h)))
        h_last = h[:, -1, :]
        return self.head(h_last)


if __name__ == "__main__":
    # smoke test
    model = MambaModel(n_features=20, n_targets=8, d_model=64, n_layers=3)
    x = torch.randn(4, 24, 20)
    out = model(x)
    print("Mamba output shape:", out.shape)  # expect (4, 8)