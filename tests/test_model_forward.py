from __future__ import annotations

import torch

from models.mlp_surrogate import MLPSurrogate
from models.resmlp_surrogate import ResMLPSurrogate


def test_mlp_forward_shape():
    model = MLPSurrogate(input_dim=8, output_dim=96)
    y = model(torch.rand(4, 8))
    assert y.shape == (4, 96)


def test_resmlp_forward_shape():
    model = ResMLPSurrogate(input_dim=8, output_dim=96, hidden_dim=32, num_blocks=2)
    y = model(torch.rand(4, 8))
    assert y.shape == (4, 96)
