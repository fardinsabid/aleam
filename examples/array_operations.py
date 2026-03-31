"""
Array operations examples for Aleam.
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("ALEAM - Array Operations Examples")
    print("=" * 70)
    
    rng = al.Aleam()
    
    print("\n📊 Random Array Generation:")
    
    # 1D array
    arr_1d = rng.random_array(10)
    print(f"  random_array(10): {[f'{x:.4f}' for x in arr_1d]}")
    
    # 2D array
    arr_2d = rng.random_array((3, 4))
    print(f"  random_array((3,4)):\n{np.array(arr_2d)}")
    
    # 3D array
    arr_3d = rng.random_array((2, 3, 2))
    print(f"  random_array((2,3,2)) shape: {np.array(arr_3d).shape}")
    
    print("\n📈 Normal Array Generation:")
    
    # Normal distribution array
    norm_arr = rng.randn_array((1000,), mu=0, sigma=1)
    print(f"  randn_array(1000): mean={np.mean(norm_arr):.4f}, std={np.std(norm_arr):.4f}")
    
    # 2D normal array
    norm_2d = rng.randn_array((5, 5), mu=0, sigma=1)
    print(f"  randn_array((5,5)):\n{np.array(norm_2d):.4f}")
    
    print("\n🔢 Integer Array Generation:")
    
    # Integer array
    int_arr = rng.randint_array((10,), low=0, high=10)
    print(f"  randint_array(10, 0, 10): {int_arr}")
    
    # 2D integer array
    int_2d = rng.randint_array((4, 5), low=0, high=100)
    print(f"  randint_array((4,5), 0, 100):\n{np.array(int_2d)}")
    
    print("\n🎲 Choice Array:")
    
    # Choice from range
    choice_arr = al.choice_array(10, size=(20,), replace=True)
    print(f"  choice_array(10, size=20): {choice_arr}")
    
    # Choice without replacemal
    try:
        choice_no_replace = al.choice_array(10, size=(5,), replace=False)
        print(f"  choice_array(10, size=5, replace=False): {choice_no_replace}")
    except ValueError as e:
        print(f"  choice_array without replacement: {e}")
    
    # Choice from list
    fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    fruit_choices = al.choice_array(fruits, size=(10,))
    print(f"  choice_array from fruits: {fruit_choices}")
    
    # Weighted choice
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    weighted_choices = al.choice_array(fruits, size=(100,), p=weights)
    from collections import Counter
    counts = Counter(weighted_choices)
    print(f"  Weighted choice counts: {dict(counts)}")
    
    print("\n" + "=" * 70)
    print("✅ Array operations demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()