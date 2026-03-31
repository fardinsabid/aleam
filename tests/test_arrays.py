"""
Tests for array operations.
"""

import pytest
import numpy as np
from aleam import random_array, randn_array, randint_array, choice_array


class TestArrayFunctions:
    """Test array operation functions"""
    
    def test_random_array_1d(self):
        arr = random_array(100)
        assert len(arr) == 100
        assert all(0 <= x < 1 for x in arr)
    
    def test_random_array_2d(self):
        arr = random_array((10, 10))
        assert len(arr) == 10
        assert len(arr[0]) == 10
    
    def test_random_array_3d(self):
        arr = random_array((5, 5, 5))
        assert len(arr) == 5
        assert len(arr[0]) == 5
        assert len(arr[0][0]) == 5
    
    def test_randn_array(self):
        arr = randn_array((1000,), mu=0, sigma=1)
        mean = sum(arr) / len(arr)
        assert abs(mean) < 0.1
    
    def test_randint_array(self):
        arr = randint_array((100,), 0, 10)
        assert all(0 <= x <= 10 for x in arr)
        assert len(set(arr)) > 1  # Should have some variety
    
    def test_choice_array_with_replacement(self):
        arr = choice_array(10, size=(20,), replace=True)
        assert len(arr) == 20
        assert all(0 <= x <= 9 for x in arr)
    
    def test_choice_array_without_replacement(self):
        arr = choice_array(10, size=(5,), replace=False)
        assert len(arr) == 5
        assert len(set(arr)) == 5
        assert all(0 <= x <= 9 for x in arr)
    
    def test_choice_array_with_weights(self):
        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        arr = choice_array(10, size=(100,), p=weights)
        assert len(arr) == 100
    
    def test_choice_array_from_list(self):
        items = ['a', 'b', 'c', 'd']
        arr = choice_array(items, size=(10,))
        assert all(x in items for x in arr)
    
    def test_choice_array_no_size(self):
        val = choice_array([1, 2, 3])
        assert val in [1, 2, 3]
    
    def test_choice_array_invalid_size(self):
        with pytest.raises(ValueError):
            choice_array(10, size=(20,), replace=False)