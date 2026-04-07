#!/usr/bin/env python3
"""
AI/ML specific features examples for Aleam (C++ Core).

This example demonstrates the AI-focused random utilities including:
- Gradient noise for training
- Latent space sampling for generative models
- Dropout masks for regularization
- Data augmentation parameters
- Mini-batch sampling
- Reinforcement learning exploration noise

Note: The C++ API expects:
- gradient_noise(shape) where shape is total number of elements (not a tuple)
- All methods return numpy arrays
- interpolate() expects Python lists, not numpy arrays
- Vector dimensions must match latent_dim
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - AI/ML Feature Examples (C++ Core)")
    print("=" * 70)
    
    # AI-specific random utilities
    ai = al.AIRandom()
    
    print("\n Gradient Noise for Training:")
    print("  Note: gradient_noise() takes total elements, not a tuple shape")
    total_elements = 16  # 4x4 = 16
    noise = ai.gradient_noise(total_elements, scale=0.1)
    print(f"  Total elements: {total_elements}")
    print(f"  Mean: {np.mean(noise):.4f}")
    print(f"  Std: {np.std(noise):.4f}")
    print(f"  Reshaped to 4x4:\n{noise.reshape(4, 4)}")
    
    print("\n Latent Space Vector:")
    latent = ai.latent_vector(512, "normal")
    print(f"  Dimension: {len(latent)}")
    print(f"  Mean: {np.mean(latent):.4f}")
    print(f"  Variance: {np.var(latent):.4f}")
    
    print("\n Dropout Mask (keep_prob=0.3):")
    mask = ai.dropout_mask(20, 0.3)
    print(f"  Mask: {mask}")
    print(f"  Active: {sum(mask)}/20 ({sum(mask)/20*100:.0f}%)")
    
    print("\n Data Augmentation Parameters:")
    params = ai.augmentation_params()
    print(f"  Rotation: {params.rotation:.2f} deg")
    print(f"  Scale: {params.scale:.3f}")
    print(f"  Brightness: {params.brightness:.3f}")
    print(f"  Contrast: {params.contrast:.3f}")
    print(f"  Flip horizontal: {params.flip_horizontal}")
    print(f"  Flip vertical: {params.flip_vertical}")
    
    print("\n Mini-batch Sampling:")
    dataset_size = 10000
    batch_size = 64
    batch = ai.mini_batch(dataset_size, batch_size)
    print(f"  Dataset size: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  First 10 indices: {batch[:10]}")
    
    print("\n Reinforcement Learning Exploration Noise:")
    action_dim = 4
    noise_rl = ai.exploration_noise(action_dim, scale=0.2)
    print(f"  Action space: {action_dim}")
    print(f"  Noise: {[f'{x:.4f}' for x in noise_rl]}")
    print(f"  Mean: {np.mean(noise_rl):.4f}")
    print(f"  Std: {np.std(noise_rl):.4f}")
    
    print("\n Gradient Noise with Decay:")
    grad_noise = al.GradientNoise(initial_scale=0.1, decay=0.95)
    gradients = np.ones(9)  # 3x3 flattened to 1D
    print(f"  Note: add_noise() expects 1D array, got shape {gradients.shape}")
    for i in range(5):
        noisy = grad_noise.add_noise(gradients.tolist())
        diff = np.mean(np.abs(np.array(noisy) - gradients))
        print(f"  Step {i+1}: scale={grad_noise.get_current_scale():.6f}, mean diff={diff:.6f}")
    
    grad_noise.reset()
    print(f"  Reset step: {grad_noise.get_step()}")
    
    print("\n Latent Space Sampler (dim=32):")
    sampler = al.LatentSampler(latent_dim=32, distribution="normal")
    samples = sampler.sample(5)
    print(f"  Sampled {len(samples)} vectors of dimension {len(samples[0])}")
    sample_means = [np.mean(s) for s in samples[:3]]
    sample_stds = [np.std(s) for s in samples[:3]]
    print(f"  Sample means: {[f'{m:.4f}' for m in sample_means]}")
    print(f"  Sample stds: {[f'{s:.4f}' for s in sample_stds]}")
    
    print("\n Latent Interpolation (dim=3):")
    # Create a separate sampler for 3D vectors
    sampler_3d = al.LatentSampler(latent_dim=3, distribution="normal")
    z1 = [1.0, 0.0, 0.0]
    z2 = [0.0, 1.0, 0.0]
    interp = sampler_3d.interpolate(z1, z2, steps=5)
    print(f"  Interpolation between {z1} and {z2}:")
    for i, vec in enumerate(interp):
        print(f"    Step {i}: {[f'{x:.2f}' for x in vec]}")
    
    print("\n" + "=" * 70)
    print(" AI/ML features demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()