"""
Example training script for DG‑STMTL.

This script demonstrates how to prepare multi‑task traffic data using the
utility functions from ``utility1_flow.py``, build simple prior
adjacency matrices from the data itself, instantiate
:class:`dgstmtl.model.DGSTMTLNet` and train it using a mean‑squared
error loss.  It is designed to be straightforward to adapt to new
datasets: replace the paths to your CSV files and adjust the command
line arguments as needed.

By default, the script constructs the static prior matrices as
follows:

* ``A_S`` (spatial connectivity) is estimated via the positive part of
  the Pearson correlation between node time series across all tasks and
  time steps in the training set.
* ``A_T`` (temporal continuity) is an identity matrix.
* ``A_ST`` (spatio‑temporal similarity) is set equal to ``A_S`` for
  simplicity.  You can modify the function ``compute_priors`` if you
  have a more principled way to derive these matrices from your data.

Refer to the DG‑STMTL paper【439269036305875†L560-L601】 for the
rationale behind these priors and the overall architecture.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dgstmtl.model import DGSTMTLNet
import utility1_flow as util


def compute_priors(X_train: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute simple prior adjacency matrices from training data.

    Args:
        X_train: Array of shape ``(num_samples, N, T, K)`` containing the
            multi‑task training data.

    Returns:
        Tuple of three PyTorch tensors ``(A_S, A_T, A_ST)`` of shape
        ``(N, N)``.
    """
    # Flatten across samples, time and tasks to build correlation matrix
    num_samples, N, T, K = X_train.shape
    flattened = X_train.transpose(1, 0, 2, 3).reshape(N, -1)
    # Pearson correlation across nodes
    corr = np.corrcoef(flattened)
    # Take positive correlations only to avoid unstable negative edges
    corr = np.maximum(corr, 0.0)
    # Normalise to [0, 1]
    if np.max(corr) > 0:
        corr = corr / np.max(corr)
    A_S = torch.from_numpy(corr.astype(np.float32))
    A_T = torch.eye(N, dtype=torch.float32)
    # For simplicity set A_ST equal to A_S.  You may use other
    # measures (e.g., cross‑task correlation) here.
    A_ST = A_S.clone()
    return A_S, A_T, A_ST


class MultiTaskDataset(Dataset):
    """Wrap the arrays returned by the utility loader into a PyTorch dataset."""
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.X = X  # (num_samples, N, T, K)
        self.y = y  # (num_samples, N, K)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    """Custom collate function to convert numpy arrays into task lists."""
    xs, ys = zip(*batch)
    xs = np.stack(xs)  # (B, N, T, K)
    ys = np.stack(ys)  # (B, N, K)
    # Convert to torch tensors
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
    # Build per‑task tensors: list of K tensors of shape (B, T, N, 1)
    B, N, T, K = xs.shape
    x_tasks = []
    for k in range(K):
        x_k = xs[:, :, :, k]  # (B, N, T)
        x_k = x_k.permute(0, 2, 1).unsqueeze(-1)  # (B, T, N, 1)
        x_tasks.append(x_k)
    return x_tasks, ys


def main(args: argparse.Namespace) -> None:
    # Load data using the existing utility function.  The loader returns
    # sequences of length ``timestep`` and labels at the next time step.
    data = util.load_dataset(
        url1=args.data1,
        url2=args.data2,
        normalizer=args.normalizer,
        timestep=args.timestep,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
    )

    X_train = data["X_train"]  # (num_samples, N, T, K)
    y_train = data["y_train"]  # (num_samples, N, K)

    # Compute static prior adjacency matrices from the training set
    A_S, A_T, A_ST = compute_priors(X_train)
    num_nodes = X_train.shape[1]
    num_tasks = X_train.shape[3]

    # Instantiate the model
    model = DGSTMTLNet(
        num_nodes=num_nodes,
        num_tasks=num_tasks,
        in_channels=1,
        num_time_steps=args.timestep,
        A_S=A_S,
        A_T=A_T,
        A_ST=A_ST,
        m=args.m,
        d_ctke=args.d_ctke,
        hidden_dim=args.hidden_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    # Build PyTorch dataset and dataloader
    train_dataset = MultiTaskDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for x_tasks, y in train_loader:
            # Move to device
            x_tasks = [x.to(device) for x in x_tasks]
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x_tasks)  # (B, N, K)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
        epoch_loss /= len(train_dataset)
        print(f"Epoch {epoch:03d} | Train MSE: {epoch_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DG‑STMTL on a multi‑task dataset")
    parser.add_argument("--data1", type=str, default="PEMS04_Flow.csv", help="Path to CSV file for task 1")
    parser.add_argument("--data2", type=str, default="PEMS04_Speed.csv", help="Path to CSV file for task 2")
    parser.add_argument("--normalizer", type=str, default="None", choices=["None", "max01", "max11", "std"], help="Normalization scheme")
    parser.add_argument("--timestep", type=int, default=12, help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini‑batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--m", type=int, default=3, help="Group size for GSTGC")
    parser.add_argument("--d_ctke", type=int, default=96, help="Latent dimension for CTKE")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU even if CUDA is available")
    args = parser.parse_args()
    main(args)