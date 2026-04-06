#!/usr/bin/env python3
"""
AI/ML specific features examples for Aleam.

This example demonstrates the AI-focused random utilities including:
- Gradient noise for training
- Latent space sampling for generative models
- Dropout masks for regularization
- Data augmentation parameters
- Mini-batch sampling
- Reinforcement learning exploration noise
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - AI/ML Feature Examples")
    print("=" * 70)
    
    # AI-specific random utilities
    ai = al.AIRandom()
    
    print("\n🧠 Gradient Noise for Training:")
    shape = (4, 4)
    noise = ai.gradient_noise(shape, scale=0.1)
    print(f"  Shape: {shape}")
    print(f"  Mean: {np.mean(noise):.4f}")
    print(f"  Std: {np.std(noise):.4f}")
    print(f"  Sample:\n{noise[:2, :2]}")
    
    print("\n🎨 Latent Space Vector:")
    latent = ai.latent_vector(512, "normal")
    print(f"  Dimension: 512")
    print(f"  Mean: {np.mean(latent):.4f}")
    print(f"  Variance: {np.var(latent):.4f}")
    
    print("\n🎭 Dropout Mask (p=0.3):")
    mask = ai.dropout_mask(20, 0.3)
    print(f"  Mask: {mask}")
    print(f"  Active: {sum(mask)}/20 ({sum(mask)/20*100:.0f}%)")
    
    print("\n🖼️ Data Augmentation Parameters:")
    params = ai.augmentation_params()
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n📦 Mini-batch Sampling:")
    dataset_size = 10000
    batch_size = 64
    batch = ai.mini_batch(dataset_size, batch_size)
    print(f"  Dataset size: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  First 10 indices: {batch[:10]}")
    
    print("\n🤖 Reinforcement Learning Exploration Noise:")
    action_dim = 4
    noise_rl = ai.exploration_noise(action_dim, scale=0.2)
    print(f"  Action space: {action_dim}")
    print(f"  Noise: {[f'{x:.4f}' for x in noise_rl]}")
    print(f"  Mean: {np.mean(noise_rl):.4f}")
    print(f"  Std: {np.std(noise_rl):.4f}")
    
    print("\n📉 Gradient Noise with Decay:")
    grad_noise = al.GradientNoise(scale=0.1, decay=0.95)
    gradients = np.ones((3, 3))
    for i in range(5):
        noisy = grad_noise.add_noise(gradients)
        diff = np.mean(np.abs(noisy - gradients))
        print(f"  Step {i+1}: scale={grad_noise.get_current_scale():.6f}, mean diff={diff:.6f}")
    
    grad_noise.reset()
    print(f"  Reset step: {grad_noise.get_step()}")
    
    print("\n🎲 Latent Space Sampler:")
    sampler = al.LatentSampler(latent_dim=32, distribution="normal")
    samples = sampler.sample(5)
    print(f"  Sampled {len(samples)} vectors of dimension {len(samples[0])}")
    print(f"  Sample means: {[f'{np.mean(s):.4f}' for s in samples[:3]]}")
    print(f"  Sample stds: {[f'{np.std(s):.4f}' for s in samples[:3]]}")
    
    print("\n🔄 Latent Interpolation:")
    z1 = np.array([1.0, 0.0, 0.0])
    z2 = np.array([0.0, 1.0, 0.0])
    interp = sampler.interpolate(z1, z2, steps=5)
    print(f"  Interpolation between {z1} and {z2}:")
    for i, vec in enumerate(interp):
        print(f"    Step {i}: {[f'{x:.2f}' for x in vec]}")
    
    print("\n" + "=" * 70)
    print("✅ AI/ML features demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()