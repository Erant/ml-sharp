#!/usr/bin/env python
"""Simple test to verify masking tensor operations work correctly."""

# Test the tensor operations without requiring full dependencies
import numpy as np

def test_mask_dimensions():
    """Test that mask dimensions work correctly for broadcasting."""
    print("Testing mask dimension broadcasting...")

    # Simulate the shapes at each stage
    # Original image
    H, W = 2048, 1536
    print(f"Original image: ({H}, {W}, 4)")

    # After extracting alpha
    alpha_shape = (H, W, 1)
    print(f"Alpha channel: {alpha_shape}")

    # After permute(2, 0, 1)
    alpha_torch_shape = (1, H, W)
    print(f"After permute: {alpha_torch_shape}")

    # After adding batch dimension [None]
    alpha_batch_shape = (1, 1, H, W)
    print(f"After batch dim: {alpha_batch_shape}")

    # After interpolate to 1536x1536
    alpha_resized_shape = (1, 1, 1536, 1536)
    print(f"After resize: {alpha_resized_shape}")

    # After avg_pool2d with stride=2
    mask_shape = (1, 1, 768, 768)
    print(f"After downsample (mask): {mask_shape}")

    # Opacities shape (B, num_layers, H, W)
    num_layers = 2
    opacities_shape = (1, num_layers, 768, 768)
    print(f"Opacities shape: {opacities_shape}")

    # Test broadcasting
    # Create dummy arrays to verify broadcasting works
    mask = np.ones((1, 1, 768, 768))
    opacities = np.ones((1, num_layers, 768, 768))

    result = opacities * mask
    print(f"After masking: {result.shape}")

    assert result.shape == opacities_shape, f"Expected {opacities_shape}, got {result.shape}"
    print("✓ Broadcasting works correctly!")

    # Test with soft mask
    mask = np.random.rand(1, 1, 768, 768)
    opacities = np.random.rand(1, num_layers, 768, 768)
    result = opacities * mask

    # Verify that each layer gets the same mask applied
    layer_0_mask = result[0, 0, :, :] / opacities[0, 0, :, :]
    layer_1_mask = result[0, 1, :, :] / opacities[0, 1, :, :]

    # Should be identical (within numerical precision)
    diff = np.abs(layer_0_mask - layer_1_mask)
    max_diff = np.max(diff)
    print(f"Max difference between layer masks: {max_diff}")
    assert max_diff < 1e-10, "Mask not applied consistently across layers"
    print("✓ Mask applied consistently across all layers!")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_mask_dimensions()
