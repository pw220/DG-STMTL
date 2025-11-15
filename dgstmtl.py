

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .modules import CTKE, HAMG, GSTGC


def build_synchronous_prior(
    A_S: torch.Tensor,
    A_T: torch.Tensor,
    A_ST: torch.Tensor,
    m: int = 3,
) -> torch.Tensor:
    """Construct a static synchronous spatio‑temporal prior adjacency.

    Given base adjacency matrices for spatial connectivity (`A_S`),
    temporal continuity (`A_T`) and spatio‑temporal similarity (`A_ST`), this
    helper builds a block‑structured matrix of size ``(m·N, m·N)``.  The
    diagonal blocks contain the sum ``A_S + A_ST`` while the immediate
    off‑diagonals contain ``A_T``【439269036305875†L560-L601】.

    Args:
        A_S: Spatial connectivity matrix of shape ``(N, N)``.
        A_T: Temporal self‑loop matrix of shape ``(N, N)`` (often identity).
        A_ST: Spatio‑temporal similarity matrix of shape ``(N, N)``.
        m: Number of time steps per group.

    Returns:
        Tensor of shape ``(m·N, m·N)`` representing the static prior.
    """
    assert A_S.shape == A_T.shape == A_ST.shape
    N = A_S.size(0)
    A_diag = A_S + A_ST
    zero = torch.zeros_like(A_S)
    blocks: List[List[torch.Tensor]] = [[zero for _ in range(m)] for _ in range(m)]
    for t in range(m):
        blocks[t][t] = A_diag.clone()
        if t < m - 1:
            blocks[t][t + 1] = A_T.clone()
            blocks[t + 1][t] = A_T.clone()
    rows = [torch.cat(row, dim=1) for row in blocks]
    A_P = torch.cat(rows, dim=0)
    return A_P


class DGSTMTLNet(nn.Module):
    """Dynamic Group‑wise Spatio‑Temporal Multi‑Task Learning network.

    Args:
        num_nodes: Number of spatial nodes ``N``.
        num_tasks: Number of tasks ``K``.
        in_channels: Number of input channels per task.
        num_time_steps: Length of the input history ``T``.
        A_S: Spatial connectivity prior (``N×N``) used to build the
            synchronous prior.
        A_T: Temporal self‑loop prior (``N×N``).
        A_ST: Spatio‑temporal similarity prior (``N×N``).
        m: Temporal grouping length (defaults to 3).
        d_ctke: Latent dimension used in CTKE.
        hidden_dim: Hidden dimension used throughout GSTGC and the
            output head.
    """

    def __init__(
        self,
        num_nodes: int,
        num_tasks: int,
        in_channels: int,
        num_time_steps: int,
        A_S: torch.Tensor,
        A_T: torch.Tensor,
        A_ST: torch.Tensor,
        m: int = 3,
        d_ctke: int = 96,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_tasks = num_tasks
        self.in_channels = in_channels
        self.num_time_steps = num_time_steps
        self.m = m
        self.hidden_dim = hidden_dim

        # Build static synchronous prior and register as buffer
        A_P = build_synchronous_prior(A_S, A_T, A_ST, m=m)
        self.register_buffer("A_P", A_P)

        # CTKE and HAMG
        self.ctke = CTKE(
            num_nodes=num_nodes,
            num_tasks=num_tasks,
            in_channels=in_channels,
            num_time_steps=num_time_steps,
            m=m,
            d_model=d_ctke,
        )
        self.hamg = HAMG(prior_adj=A_P, num_tasks=num_tasks)

        # Task‑specific input layers
        self.input_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, hidden_dim)
                for _ in range(num_tasks)
            ]
        )

        # GSTGC layers for each task
        self.gstgc_layers = nn.ModuleList(
            [
                GSTGC(
                    num_nodes=num_nodes,
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_time_steps=num_time_steps,
                    m=m,
                )
                for _ in range(num_tasks)
            ]
        )

        # Task‑agnostic output layers
        self.out_fc1 = nn.Linear(hidden_dim * num_tasks, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, num_tasks)
        self.residual_proj = nn.Linear(hidden_dim * num_tasks, num_tasks)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.input_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out_fc1.weight)
        nn.init.zeros_(self.out_fc1.bias)
        nn.init.xavier_uniform_(self.out_fc2.weight)
        nn.init.zeros_(self.out_fc2.bias)
        nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(self, x_tasks: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of DGSTMTLNet.

        Args:
            x_tasks: List of length ``K`` where each item is a tensor of
                shape ``(B, T, N, Cin)``.

        Returns:
            A tensor of shape ``(B, N, K)`` containing predictions for each
            task at each node.
        """
        assert len(x_tasks) == self.num_tasks, "len(x_tasks) must equal num_tasks."

        # 1) Task‑specific input projection
        projected: List[torch.Tensor] = []
        for k, x_k in enumerate(x_tasks):
            Bsz, T, N, C = x_k.shape
            assert T == self.num_time_steps
            assert N == self.num_nodes
            assert C == self.in_channels
            x_lin = self.input_layers[k](x_k)  # (B, T, N, hidden)
            x_lin = F.relu(x_lin)
            projected.append(x_lin)

        # 2) Dynamic adjacency from CTKE
        B_dyn = self.ctke(x_tasks)

        # 3) Hybrid adjacencies per task via HAMG
        A_tasks = self.hamg(B_dyn)

        # 4) GSTGC per task
        gstgc_outputs: List[torch.Tensor] = []
        for k in range(self.num_tasks):
            A_k = A_tasks[k]
            X_k = projected[k]
            M_out_k = self.gstgc_layers[k](X_k, A_k)
            gstgc_outputs.append(M_out_k)

        # 5) Task‑agnostic output head
        M_stack = torch.stack(gstgc_outputs, dim=-1)  # (B, N, hidden, K)
        Bsz, N, H, K = M_stack.shape
        M_flat = M_stack.permute(0, 1, 3, 2).contiguous().view(Bsz * N, H * K)
        h = F.relu(self.out_fc1(M_flat))
        out = self.out_fc2(h)
        res = self.residual_proj(M_flat)
        y = out + res
        y = y.view(Bsz, N, K)
        return y