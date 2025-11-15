"""
Core modules for DG‑STMTL
=========================

This module implements the three key components of the DG‑STMTL
architecture:

* :class:`CTKE` – Cross‑Task Knowledge Exchange unit.  It builds a
  dynamic adjacency matrix from multiple task inputs by aggregating
  spatio‑temporal task embeddings and projecting them to a latent
  space【439269036305875†L560-L601】.

* :class:`HAMG` – Hybrid Adjacency Matrix Generation module.  It
  combines a static spatio‑temporal prior matrix with the dynamic
  adjacency via a learnable, task‑specific gating matrix to produce
  task‑dependent hybrid adjacency matrices【439269036305875†L560-L601】.

* :class:`GSTGC` – Group‑wise Spatio‑Temporal Graph Convolution.  It
  performs multi‑stage graph convolutions on temporally grouped
  features, followed by feature grouping and pooling, to learn
  spatio‑temporal representations【439269036305875†L623-L699】.

The implementation closely follows the mathematical formulation in
Section 3 of the paper and includes extensive inline comments to
clarify tensor shapes.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class CTKE(nn.Module):
    """Cross‑Task Knowledge Exchange (CTKE).

    The CTKE unit constructs a dynamic adjacency matrix by first
    embedding each task’s spatio‑temporal features, then aggregating
    over the temporal dimension, projecting into a latent space, and
    finally computing the pairwise similarity via a dot product.  A
    row‑wise softmax yields a dense adjacency matrix `B` of size
    ``(m · N, m · N)``【439269036305875†L560-L601】.

    Args:
        num_nodes: Number of spatial nodes ``N``.
        num_tasks: Number of prediction tasks ``K``.
        in_channels: Input channels per task ``C``.
        num_time_steps: History length ``T``.
        m: Number of consecutive time steps per group (e.g., 3).
        d_model: Latent dimension ``D`` after projection.  Must be
            divisible by ``m``.
    """

    def __init__(
        self,
        num_nodes: int,
        num_tasks: int,
        in_channels: int,
        num_time_steps: int,
        m: int = 3,
        d_model: int = 96,
    ) -> None:
        super().__init__()
        assert d_model % m == 0, "d_model must be divisible by m."
        self.num_nodes = num_nodes
        self.num_tasks = num_tasks
        self.in_channels = in_channels
        self.num_time_steps = num_time_steps
        self.m = m
        self.d_model = d_model

        # Temporal task embedding ETK ∈ R^{1×T×K×1}
        self.temporal_task_embedding = nn.Parameter(
            torch.zeros(1, num_time_steps, num_tasks, 1)
        )
        # Spatial task embedding ESK ∈ R^{N×1×K×1}
        self.spatial_task_embedding = nn.Parameter(
            torch.zeros(num_nodes, 1, num_tasks, 1)
        )

        # Linear layer W ∈ R^{(K·C)×D}
        self.proj = nn.Linear(num_tasks * in_channels, d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.temporal_task_embedding)
        nn.init.xavier_uniform_(self.spatial_task_embedding)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    @torch.no_grad()
    def _check_shapes(self, xs: List[torch.Tensor]) -> None:
        """Validate that all task tensors have matching shapes."""
        t0 = xs[0].shape[1]
        n0 = xs[0].shape[2]
        c0 = xs[0].shape[3]
        for x in xs:
            assert x.shape[1] == t0, "All tasks must share the same T."
            assert x.shape[2] == n0, "All tasks must share the same N."
            assert x.shape[3] == c0, "All tasks must share the same C."
        assert t0 == self.num_time_steps
        assert n0 == self.num_nodes
        assert c0 == self.in_channels

    def forward(self, x_tasks: List[torch.Tensor]) -> torch.Tensor:
        """Compute the dynamic adjacency matrix.

        Args:
            x_tasks: List of length ``K``.  Each entry ``x_tasks[k]`` has
                shape ``(B, T, N, C)``.

        Returns:
            `B`: Tensor of shape ``(m·N, m·N)`` representing the dynamic
            adjacency matrix used by all tasks【439269036305875†L560-L601】.
        """
        assert len(x_tasks) == self.num_tasks
        self._check_shapes(x_tasks)

        # Stack along the task dimension -> (B, T, N, K, C)
        x_stack = torch.stack(x_tasks, dim=3)
        # Average over batch dimension -> (T, N, K, C)
        x_mean = x_stack.mean(dim=0)
        # Move dimensions to (N, T, K, C)
        x_mean = x_mean.permute(1, 0, 2, 3)

        # Add temporal & spatial task embeddings
        x_embedded = (
            x_mean
            + self.temporal_task_embedding  # broadcast over N
            + self.spatial_task_embedding   # broadcast over T
        )  # (N, T, K, C)

        # Max over temporal dimension to obtain a summary per node/task
        x_agg, _ = x_embedded.max(dim=1)  # (N, K, C)

        # Flatten the task and channel dimensions for projection: (N, K·C)
        x_flat = x_agg.reshape(self.num_nodes, self.num_tasks * self.in_channels)

        # Linear projection: (N, D)
        x_trans = F.relu(self.proj(x_flat))

        # Reshape into groups for similarity computation.  Let D' = D / m.
        D_prime = self.d_model // self.m
        x_grouped = x_trans.view(self.num_nodes, self.m, D_prime)
        # -> (m, N, D') -> (m·N, D')
        x_grouped = x_grouped.permute(1, 0, 2).reshape(self.m * self.num_nodes, D_prime)

        # Compute correlation matrix and apply row‑wise softmax
        C = torch.matmul(x_grouped, x_grouped.transpose(0, 1))  # (mN, mN)
        B = F.softmax(C, dim=-1)
        return B


class GCNAGG(nn.Module):
    """Basic GCN aggregation layer.

    Implements ``H' = ReLU(A · H · W)`` for a given adjacency matrix ``A``.
    The bias term is absorbed into the linear transformation.

    Args:
        in_dim: Input channel dimension.
        out_dim: Output channel dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # H: (B, mN, Cin), A: (mN, mN)
        support = self.lin(H)  # (B, mN, Cout)
        out = torch.einsum("ij,bjc->bic", A, support)  # Graph propagation
        return F.relu(out)


class GroupWiseBlock(nn.Module):
    """Group‑wise spatio‑temporal learning block.

    Each block applies three sequential GCN aggregations with residual
    fusion.  After convolution, the middle temporal slice of the input
    sequence is extracted to form the output【439269036305875†L623-L699】.

    Args:
        num_nodes: Number of spatial nodes ``N``.
        in_dim: Input channel dimension ``Cin``.
        hidden_dim: Hidden/Output channel dimension ``Cout``.
        m: Number of time steps per group.
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        m: int = 3,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.m = m

        self.gcn1 = GCNAGG(in_dim, hidden_dim)
        self.gcn2 = GCNAGG(hidden_dim, hidden_dim)
        self.gcn3 = GCNAGG(hidden_dim, hidden_dim)

        # Residual fusion weights w^(g) ∈ R^4 (including input)
        self.res_weights = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Apply the block to temporally grouped data.

        Args:
            x: Input tensor of shape ``(B, m, N, Cin)``.
            A: Hybrid adjacency matrix of shape ``(m·N, m·N)``.

        Returns:
            Tensor of shape ``(B, N, hidden_dim)``.
        """
        Bsz = x.size(0)
        # Flatten the temporal dimension: (B, m·N, Cin)
        H0 = x.view(Bsz, self.m * self.num_nodes, -1)

        H1 = self.gcn1(H0, A)
        H2 = self.gcn2(H1, A)
        H3 = self.gcn3(H2, A)

        weights = F.softmax(self.res_weights, dim=0)
        fused = (
            weights[0] * H0
            + weights[1] * H1
            + weights[2] * H2
            + weights[3] * H3
        )  # (B, m·N, hidden_dim)

        # Extract the middle temporal slice: take nodes in the middle group
        start = self.num_nodes
        end = 2 * self.num_nodes
        F_c = fused[:, start:end, :]
        return F_c  # (B, N, hidden_dim)


class GSTGC(nn.Module):
    """Group‑wise Spatio‑Temporal Graph Convolution (GSTGC).

    The GSTGC module first groups the input sequence along the temporal
    dimension into ``T/m`` non‑overlapping groups of length ``m``.  Each
    group is processed independently by a :class:`GroupWiseBlock`.  The
    resulting group features are then grouped again along the feature
    dimension (overlapping windows) and processed by another
    :class:`GroupWiseBlock` before a max‑pool across the overlapping
    groups【439269036305875†L623-L699】.

    Args:
        num_nodes: Number of spatial nodes ``N``.
        in_dim: Input channel dimension ``Cin``.
        hidden_dim: Hidden/Output channel dimension ``Cout``.
        num_time_steps: Input sequence length ``T``.
        m: Group size (number of time steps per group).
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        num_time_steps: int,
        m: int = 3,
    ) -> None:
        super().__init__()
        assert num_time_steps % m == 0, "T must be divisible by m."
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_time_steps = num_time_steps
        self.m = m
        self.num_temporal_groups = num_time_steps // m

        self.temporal_group_block = GroupWiseBlock(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            m=m,
        )
        self.feature_group_block = GroupWiseBlock(
            num_nodes=num_nodes,
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            m=m,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass of GSTGC.

        Args:
            x: Input tensor of shape ``(B, T, N, Cin)``.
            A: Hybrid adjacency matrix of shape ``(m·N, m·N)``.

        Returns:
            Tensor of shape ``(B, N, hidden_dim)``.
        """
        Bsz, T, N, C = x.shape
        assert T == self.num_time_steps
        assert N == self.num_nodes

        # 1) Temporal grouping into windows of length m
        temporal_outputs = []
        for g in range(self.num_temporal_groups):
            t_start = g * self.m
            t_end = (g + 1) * self.m
            x_g = x[:, t_start:t_end, :, :]  # (B, m, N, Cin)
            out_g = self.temporal_group_block(x_g, A)  # (B, N, hidden_dim)
            temporal_outputs.append(out_g)

        # Stack along temporal groups: (B, N, num_groups, hidden_dim)
        F_tensor = torch.stack(temporal_outputs, dim=2)  # (B, N, G, hidden_dim)

        # Permute to (B, N, hidden_dim, G)
        F_tensor = F_tensor.permute(0, 1, 3, 2)

        # 2) Feature grouping with overlapping windows of size 3
        assert self.num_temporal_groups >= 4, (
            "GSTGC requires at least 4 temporal groups for overlapping windows."
        )
        M1 = F_tensor[:, :, :, 0:3]
        M2 = F_tensor[:, :, :, 1:4]
        # Reinterpret last dim as temporal dimension for the second block
        M1_input = M1.permute(0, 3, 1, 2).contiguous()
        M2_input = M2.permute(0, 3, 1, 2).contiguous()

        M1_out = self.feature_group_block(M1_input, A)
        M2_out = self.feature_group_block(M2_input, A)

        # 3) Max pooling over the two overlapping groups
        stacked = torch.stack([M1_out, M2_out], dim=-1)
        M_out, _ = stacked.max(dim=-1)

        return M_out


class HAMG(nn.Module):
    """Hybrid Adjacency Matrix Generation (HAMG).

    This module fuses a static spatio‑temporal prior adjacency matrix
    ``A_P`` with a dynamic adjacency matrix ``B`` via a task‑specific
    gating mechanism【439269036305875†L560-L601】.  The gating matrices
    ``M_k`` are learnable and restricted to ``(0, 1)`` via a sigmoid.  Each
    task obtains its own hybrid adjacency matrix ``A*_k = M_k ⊙ (A_P + B)``.

    Args:
        prior_adj: Static prior adjacency matrix ``A_P`` of shape
            ``(m·N, m·N)``.
        num_tasks: Number of tasks ``K``.
    """

    def __init__(self, prior_adj: torch.Tensor, num_tasks: int) -> None:
        super().__init__()
        assert prior_adj.dim() == 2 and prior_adj.size(0) == prior_adj.size(1)
        self.num_tasks = num_tasks
        self.mN = prior_adj.size(0)
        # Register the static prior as a non‑trainable buffer
        self.register_buffer("A_prior", prior_adj)
        # Learnable gating matrices for each task
        self.task_gates = nn.Parameter(torch.ones(num_tasks, self.mN, self.mN))

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        # B: (mN, mN)
        hybrid = self.A_prior + B
        gates = torch.sigmoid(self.task_gates)  # (K, mN, mN)
        hybrid_expanded = hybrid.unsqueeze(0).expand_as(gates)
        A_tasks = gates * hybrid_expanded
        return A_tasks