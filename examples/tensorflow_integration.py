"""
TensorFlow integration examples for Aleam.
"""

import tensorflow as tf
import aleam as al


def main():
    print("=" * 70)
    print("ALEAM - TensorFlow Integration Examples")
    print("=" * 70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n🔧 GPUs available: {len(gpus)}")
    if gpus:
        print(f"   GPU: {gpus[0]}")
    
    # Create generator
    gen = al.TFGenerator()
    
    print("\n📊 Tensor Generation:")
    
    # Random normal tensor
    t_normal = gen.normal((3, 3), mean=0, stddev=1)
    print(f"  normal((3,3)):\n{t_normal.numpy()}")
    
    # Random uniform tensor
    t_uniform = gen.uniform((2, 4), minval=0, maxval=1)
    print(f"\n  uniform((2,4)):\n{t_uniform.numpy()}")
    
    # Random integer tensor
    t_int = gen.randint((3, 5), minval=0, maxval=10)
    print(f"\n  randint((3,5),0,10):\n{t_int.numpy()}")
    
    print("\n📈 Large Tensor Generation:")
    
    # Large tensor benchmark
    import time
    sizes = [(1000, 1000), (5000, 5000)]
    
    for size in sizes:
        total = size[0] * size[1]
        start = time.time()
        t = gen.normal(size)
        elapsed = time.time() - start
        print(f"  normal{size}: {total/elapsed/1e6:.2f}M elements/sec")
    
    print("\n🔬 Statistical Properties:")
    
    # Generate large sample and verify statistics
    samples = gen.normal((100000,))
    mean = tf.reduce_mean(samples).numpy()
    std = tf.math.reduce_std(samples).numpy()
    print(f"  normal(100000): mean={mean:.4f}, std={std:.4f}")
    print(f"  Expected: mean=0, std=1")
    
    print("\n🔀 Shuffling Tensors:")
    
    # Create a tensor and shuffle
    original = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    shuffled = gen.shuffle(original)
    print(f"  Original: {original.numpy()}")
    print(f"  Shuffled: {shuffled.numpy()}")
    
    print("\n" + "=" * 70)
    print("✅ TensorFlow integration demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()