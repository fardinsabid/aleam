"""
PyTorch integration for Aleam.
Provides true random tensors for GPU/CPU training.
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple
from .core import Aleam, AleamBase


class TorchGenerator:
    """
    PyTorch-compatible random generator using true randomness.
    
    Drop-in replacement for torch.Generator with true entropy.
    
    Example:
        >>> from aleam.torch_integration import TorchGenerator
        >>> gen = TorchGenerator()
        >>> tensor = torch.randn(100, 100, generator=gen)
    """
    
    def __init__(self, device: Optional[str] = None, rng: Optional[AleamBase] = None):
        """
        Args:
            device: PyTorch device ('cpu', 'cuda', etc.)
            rng: Optional Aleam instance
        """
        self.rng = rng or Aleam()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._seed = None  # No seed - true randomness
    
    def rand(self, *size, **kwargs) -> torch.Tensor:
        """
        Generate tensor of true random floats in [0, 1).
        
        Args:
            *size: Tensor dimensions
            **kwargs: Additional torch.rand arguments
        
        Returns:
            Tensor of shape size with values in [0, 1)
        """
        shape = size
        total = 1
        for dim in shape:
            total *= dim
        
        # Generate random values
        values = [self.rng.random() for _ in range(total)]
        tensor = torch.tensor(values, device=self.device, **kwargs)
        return tensor.reshape(*shape)
    
    def randn(self, *size, **kwargs) -> torch.Tensor:
        """
        Generate tensor of true random values from N(0, 1).
        
        Args:
            *size: Tensor dimensions
            **kwargs: Additional torch.randn arguments
        
        Returns:
            Tensor of shape size with values from N(0, 1)
        """
        shape = size
        total = 1
        for dim in shape:
            total *= dim
        
        # Generate random values
        values = [self.rng.gauss() for _ in range(total)]
        tensor = torch.tensor(values, device=self.device, **kwargs)
        return tensor.reshape(*shape)
    
    def randint(self, low: int, high: int, size: Union[int, Tuple[int, ...]], **kwargs) -> torch.Tensor:
        """
        Generate tensor of true random integers in [low, high).
        
        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            size: Tensor dimensions
            **kwargs: Additional torch.randint arguments
        
        Returns:
            Tensor of shape size with random integers
        """
        if isinstance(size, int):
            size = (size,)
        
        total = 1
        for dim in size:
            total *= dim
        
        # Generate random values
        values = [self.rng.randint(low, high - 1) for _ in range(total)]
        tensor = torch.tensor(values, device=self.device, dtype=torch.long, **kwargs)
        return tensor.reshape(*size)
    
    def rand_like(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate tensor of true random floats with same shape as input.
        
        Args:
            input_tensor: Reference tensor
        
        Returns:
            Tensor with same shape as input, values in [0, 1)
        """
        return self.rand(*input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    def randn_like(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate tensor of true random values from N(0,1) with same shape as input.
        
        Args:
            input_tensor: Reference tensor
        
        Returns:
            Tensor with same shape as input, values from N(0,1)
        """
        return self.randn(*input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    def normal(self, mean: float, std: float, size: Union[int, Tuple[int, ...]], **kwargs) -> torch.Tensor:
        """
        Generate tensor of true random values from N(mean, std).
        
        Args:
            mean: Mean of distribution
            std: Standard deviation
            size: Tensor dimensions
            **kwargs: Additional torch.normal arguments
        
        Returns:
            Tensor of shape size with values from N(mean, std)
        """
        if isinstance(size, int):
            size = (size,)
        
        total = 1
        for dim in size:
            total *= dim
        
        # Generate random values
        values = [self.rng.gauss(mean, std) for _ in range(total)]
        return torch.tensor(values, device=self.device, **kwargs).reshape(*size)
    
    def uniform(self, low: float, high: float, size: Union[int, Tuple[int, ...]], **kwargs) -> torch.Tensor:
        """
        Generate tensor of true random values from Uniform(low, high).
        
        Args:
            low: Lower bound
            high: Upper bound
            size: Tensor dimensions
            **kwargs: Additional torch.uniform arguments
        
        Returns:
            Tensor of shape size with values from Uniform(low, high)
        """
        if isinstance(size, int):
            size = (size,)
        
        total = 1
        for dim in size:
            total *= dim
        
        # Generate random values
        values = [self.rng.uniform(low, high) for _ in range(total)]
        return torch.tensor(values, device=self.device, **kwargs).reshape(*size)
    
    def random(self) -> float:
        """Generate single random float (compatibility with Python random)"""
        return self.rng.random()
    
    def randint_scalar(self, low: int, high: int) -> int:
        """Generate single random integer (compatibility with Python random)"""
        return self.rng.randint(low, high - 1)
    
    def choice(self, seq: List, size: Optional[int] = None) -> Union[any, List]:
        """Choose random elements from sequence"""
        if size is None:
            return self.rng.choice(seq)
        else:
            return [self.rng.choice(seq) for _ in range(size)]
    
    def manual_seed(self, seed: int):
        """
        Raise error explaining that Aleam doesn't use seeds.
        
        This method exists for compatibility with PyTorch's Generator API.
        """
        raise NotImplementedError(
            "TorchGenerator uses true randomness and does not support seeding. "
            "Each call is independent and stateless."
        )
    
    @property
    def device(self) -> torch.device:
        """Get current device"""
        return self._device
    
    @device.setter
    def device(self, device: str):
        """Set device"""
        self._device = torch.device(device)
    
    def get_state(self):
        """Get generator state (not supported)"""
        raise NotImplementedError("TorchGenerator is stateless")
    
    def set_state(self, state):
        """Set generator state (not supported)"""
        raise NotImplementedError("TorchGenerator is stateless")


def torch_rand(*size, generator: Optional[TorchGenerator] = None, **kwargs) -> torch.Tensor:
    """
    Generate true random tensor in [0, 1) using Aleam.
    
    Drop-in replacement for torch.rand with true randomness.
    
    Example:
        >>> from aleam.torch_integration import torch_rand
        >>> tensor = torch_rand(100, 100)
    """
    if generator is None:
        generator = TorchGenerator()
    return generator.rand(*size, **kwargs)


def torch_randn(*size, generator: Optional[TorchGenerator] = None, **kwargs) -> torch.Tensor:
    """
    Generate true random tensor from N(0,1) using Aleam.
    
    Drop-in replacement for torch.randn with true randomness.
    
    Example:
        >>> from aleam.torch_integration import torch_randn
        >>> tensor = torch_randn(100, 100)
    """
    if generator is None:
        generator = TorchGenerator()
    return generator.randn(*size, **kwargs)


def torch_randint(low: int, high: int, size: Union[int, Tuple[int, ...]], 
                  generator: Optional[TorchGenerator] = None, **kwargs) -> torch.Tensor:
    """
    Generate true random integer tensor using Aleam.
    
    Drop-in replacement for torch.randint with true randomness.
    
    Example:
        >>> from aleam.torch_integration import torch_randint
        >>> tensor = torch_randint(0, 10, (100, 100))
    """
    if generator is None:
        generator = TorchGenerator()
    return generator.randint(low, high, size, **kwargs)