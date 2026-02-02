"""Property-based tests for optimizer stability using Hypothesis."""

import torch
from hypothesis import given, settings, strategies as st
from torch import nn, optim

from src.optimizer import SAM, ZSharp

# Hypothesis strategies for generating training parameters
batch_sizes = st.integers(min_value=1, max_value=32)
input_dims = st.integers(min_value=1, max_value=128)
output_dims = st.integers(min_value=1, max_value=10)
rhos = st.floats(min_value=0.01, max_value=0.5)
lrs = st.floats(min_value=1e-4, max_value=1e-1)
percentiles = st.integers(min_value=50, max_value=99)


class SimpleLinearModel(nn.Module):
    """Simple linear model for property testing."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@settings(max_examples=50, deadline=None)
@given(
    batch_size=batch_sizes,
    in_dim=input_dims,
    out_dim=output_dims,
    rho=rhos,
    lr=lrs,
)
def test_sam_step_stability(
    batch_size: int, in_dim: int, out_dim: int, rho: float, lr: float
) -> None:
    """Invariant: SAM steps should never produce NaN/Inf parameters given valid inputs."""
    model = SimpleLinearModel(in_dim, out_dim)
    optimizer = SAM(list(model.parameters()), optim.SGD, rho=rho, lr=lr)

    # Generate random input
    x = torch.randn(batch_size, in_dim)
    y = torch.randn(batch_size, out_dim)
    criterion = nn.MSELoss()

    # Step 1
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step()

    # Verify no NaNs after perturbation
    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaN found after SAM first_step"
        assert not torch.isinf(p).any(), "Inf found after SAM first_step"

    # Step 2
    criterion(model(x), y).backward()
    optimizer.second_step()

    # Verify no NaNs after update
    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaN found after SAM second_step"
        assert not torch.isinf(p).any(), "Inf found after SAM second_step"


@settings(max_examples=50, deadline=None)
@given(
    batch_size=batch_sizes,
    in_dim=input_dims,
    out_dim=output_dims,
    rho=rhos,
    lr=lrs,
    percentile=percentiles,
)
def test_zsharp_step_stability(
    batch_size: int,
    in_dim: int,
    out_dim: int,
    rho: float,
    lr: float,
    percentile: int,
) -> None:
    """Invariant: ZSharp steps should never produce NaN/Inf parameters."""
    model = SimpleLinearModel(in_dim, out_dim)
    optimizer = ZSharp(
        list(model.parameters()),
        optim.SGD,
        rho=rho,
        lr=lr,
        percentile=percentile,
    )

    x = torch.randn(batch_size, in_dim)
    y = torch.randn(batch_size, out_dim)
    criterion = nn.MSELoss()

    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step()

    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaN found after ZSharp first_step"

    criterion(model(x), y).backward()
    optimizer.second_step()

    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaN found after ZSharp second_step"
