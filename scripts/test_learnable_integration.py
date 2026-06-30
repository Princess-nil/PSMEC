#!/usr/bin/env python
"""
Test that learnable maturity weights are integrated correctly in JointMemoryRouter.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from new_task.streaming_memory import memory_candidate_feature_dim

# Import the router (this also tests the import path fix)
sys.path.insert(0, str(Path(__file__).parent))
from train_gpu_joint_memory_router import JointMemoryRouter


def test_learnable_maturity_integration():
    """Test that JointMemoryRouter correctly uses learnable maturity weights."""

    print("="*60)
    print("Testing Learnable Maturity Weights Integration")
    print("="*60)

    # Setup
    batch_size = 2
    top_k = 3
    aggregate_dim = 54  # From shortlist_aggregate_feature_dim
    candidate_dim = 35  # 24 original + 11 raw maturity = 35

    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Top-K: {top_k}")
    print(f"  Aggregate dim: {aggregate_dim}")
    print(f"  Candidate dim: {candidate_dim}")
    print(f"  Memory candidate feature dim: {memory_candidate_feature_dim()}")

    # Create model
    model = JointMemoryRouter(
        aggregate_dim=aggregate_dim,
        candidate_dim=candidate_dim,
        hidden_dim=160,
        dropout=0.15,
        enable_risk_heads=False,
    )

    print(f"\n✅ Model created successfully")
    print(f"   Maturity weights shape: {model.maturity_weights.shape}")
    print(f"   Initial maturity weights: {model.maturity_weights.data.numpy()}")

    # Create dummy input
    aggregate_x = torch.randn(batch_size, aggregate_dim)
    candidate_x = torch.randn(batch_size, top_k, candidate_dim)
    candidate_mask = torch.ones(batch_size, top_k, dtype=torch.bool)

    # Forward pass
    print(f"\n✅ Running forward pass...")
    create_logit, candidate_logits, joint_logits, novelty_adjustment = model(
        aggregate_x, candidate_x, candidate_mask
    )

    print(f"   Create logit shape: {create_logit.shape}")
    print(f"   Candidate logits shape: {candidate_logits.shape}")
    print(f"   Joint logits shape: {joint_logits.shape}")

    # Test backward pass
    print(f"\n✅ Testing backward pass...")
    loss = joint_logits.sum()
    loss.backward()

    # Check if maturity weights have gradients
    if model.maturity_weights.grad is not None:
        print(f"   ✅ Maturity weights have gradients!")
        print(f"   Gradient values: {model.maturity_weights.grad.numpy()}")
    else:
        print(f"   ❌ Maturity weights have NO gradients!")
        return False

    # Test that gradients are not zero
    grad_norm = model.maturity_weights.grad.norm().item()
    print(f"   Gradient norm: {grad_norm:.6f}")

    if grad_norm > 1e-8:
        print(f"   ✅ Gradients are non-zero (can learn)")
    else:
        print(f"   ❌ Gradients are effectively zero")
        return False

    # Test optimization step
    print(f"\n✅ Testing optimization step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    weights_before = model.maturity_weights.data.clone()
    optimizer.step()
    weights_after = model.maturity_weights.data

    weight_change = (weights_after - weights_before).abs().max().item()
    print(f"   Max weight change: {weight_change:.6f}")

    if weight_change > 1e-6:
        print(f"   ✅ Weights updated successfully")
    else:
        print(f"   ❌ Weights did not update")
        return False

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nConclusion:")
    print("  - Learnable maturity weights are correctly integrated")
    print("  - Gradients flow through maturity weight computation")
    print("  - Weights can be optimized via backpropagation")
    print("  - Paper claim 'learned via backpropagation' is validated")
    print("="*60)

    return True


if __name__ == "__main__":
    success = test_learnable_maturity_integration()
    sys.exit(0 if success else 1)
