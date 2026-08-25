"""
S4D (Diagonal State Space Model) -- pure PyTorch, no custom CUDA kernels.

This is the simplified/diagonal variant of S4 (Gu et al.), chosen over the
full S4 (HiPPO-initialized, Cauchy-kernel convolution) because it avoids the
custom CUDA extensions that made the original S4 implementation painful to
install, while keeping the core idea: a linear state-space layer that can
model very long-range dependencies in O(L log L) via FFT-based convolution,
rather than the O(L) sequential recurrence of an RNN/GRU/LSTM.

Fits your existing pipeline shape convention: input (batch, seq_len,
n_features) -> output (batch, n_targets), i.e. same interface as your
LSTM/GRU/TCN models, so it should drop into the same Ray training loop with
minimal changes to the data loader / training script.

NOTE: this is a from-scratch, standard-reference implementation intended to
be readable and portable across HPC environments. If TALON's CUDA/PyTorch
versions are confirmed compatible with the official `mamba-ssm` /
`s4-pytorch` packages later, swapping in the official CUDA-kernel version is
a drop-in speed optimization, not a rewrite of this interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DLayer(nn.Module):
    """
    Single S4D layer. Learns a diagonal state-space model per channel and
    applies it via FFT-based convolution over the sequence dimension.
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # HiPPO-inspired diagonal initialization (real + imaginary parts of
        # the state matrix eigenvalues). log_A_real kept in log-space so the
        # real part stays negative (stability) after exponentiating.
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        A_imag = torch.pi * torch.arange(d_state).float().repeat(d_model, 1)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        self.C = nn.Parameter(torch.randn(d_model, d_state, 2) * 0.5**0.5)
        self.D = nn.Parameter(torch.randn(d_model))
        self.log_dt = nn.Parameter(torch.rand(d_model) * (torch.log(torch.tensor(0.1)) -
                                                            torch.log(torch.tensor(0.001))) +
                                    torch.log(torch.tensor(0.001)))

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()

    def kernel(self, seq_len: int, device) -> torch.Tensor:
        dt = torch.exp(self.log_dt).unsqueeze(-1)                     # (d_model, 1)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag            # (d_model, d_state)
        C = torch.view_as_complex(self.C)                             # (d_model, d_state)

        dtA = A * dt                                                  # (d_model, d_state)
        t = torch.arange(seq_len, device=device).float()              # (seq_len,)
        # (d_model, d_state, seq_len)
        power = dtA.unsqueeze(-1) * t.view(1, 1, -1)
        K = 2 * (C.unsqueeze(-1) * torch.exp(power)).real.sum(dim=1)  # (d_model, seq_len)
        return K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        b, l, d = x.shape
        u = x.transpose(1, 2)  # (batch, d_model, seq_len)

        K = self.kernel(l, x.device)  # (d_model, seq_len)

        # FFT-based causal convolution
        n_fft = 2 * l
        K_f = torch.fft.rfft(K, n=n_fft)
        u_f = torch.fft.rfft(u, n=n_fft)
        y = torch.fft.irfft(u_f * K_f, n=n_fft)[..., :l]  # (batch, d_model, seq_len)

        y = y + u * self.D.unsqueeze(-1)  # skip connection
        y = y.transpose(1, 2)             # (batch, seq_len, d_model)
        y = self.activation(y)
        y = self.dropout(y)
        return self.out_proj(y)


class S4DModel(nn.Module):
    """
    Stack of S4D layers for the soil sequence-to-target regression task.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_targets)
    """

    def __init__(self, n_features: int, n_targets: int, d_model: int = 128,
                 n_layers: int = 4, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([
            S4DLayer(d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)  # (batch, seq_len, d_model)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))  # pre-norm residual
        h_last = h[:, -1, :]  # use final timestep's representation for prediction
        return self.head(h_last)


if __name__ == "__main__":
    # smoke test
    model = S4DModel(n_features=20, n_targets=8, d_model=64, n_layers=3)
    x = torch.randn(4, 24, 20)  # (batch=4, seq_len=24, n_features=20)
    out = model(x)
    print("S4D output shape:", out.shape)  # expect (4, 8)