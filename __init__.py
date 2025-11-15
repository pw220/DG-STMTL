"""
dgstmtl package
=================

This package implements the Dynamic Group‑wise Spatio‑Temporal
Multi‑Task Learning (DG‑STMTL) framework described in the paper
“DG‑STMTL: A Novel Graph Convolutional Network for Multi‑Task
Spatio‑Temporal Traffic Forecasting”【439269036305875†L560-L601】.  The
core modules encapsulate the cross‑task knowledge exchange unit
(CTKE), the hybrid adjacency matrix generation module (HAMG) with
task‑specific gating, and the group‑wise spatio‑temporal graph
convolution (GSTGC).  These building blocks can be composed to
construct a high‑level DGSTMTL network for multi‑task forecasting.

To use this package, import :class:`dgstmtl.model.DGSTMTLNet` and
instantiate it with your own priors and dataset specifications.  See
``train_dgstmtl.py`` for a worked example using synthetic data.

The implementation strives to follow the formulation in the paper as
closely as possible: the CTKE unit builds a dynamic adjacency matrix
from multi‑task features, the HAMG module fuses static priors and
dynamic adjacency via a task‑specific gating matrix, and the GSTGC
module performs group‑wise spatio‑temporal graph convolutions with
residual fusion【439269036305875†L560-L601】【439269036305875†L623-L699】.

"""

from .model import DGSTMTLNet, build_synchronous_prior  # noqa: F401