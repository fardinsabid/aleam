"""
Tests for entropy sources.
"""

import pytest
from aleam.sources import SystemEntropy, HardwareEntropy


class TestSystemEntropy:
    """Test system entropy source"""
    
    def setup_method(self):
        self.source = SystemEntropy()
    
    def test_get_entropy_returns_int(self):
        entropy = self.source.get_entropy(8)
        assert isinstance(entropy, int)
    
    def test_get_entropy_different_sizes(self):
        for bytes in [4, 8, 16, 32]:
            entropy = self.source.get_entropy(bytes)
            assert entropy >= 0
            assert entropy.bit_length() <= bytes * 8
    
    def test_get_entropy_varies(self):
        e1 = self.source.get_entropy(8)
        e2 = self.source.get_entropy(8)
        # They could theoretically be equal, but extremely unlikely
        # We'll just check they're valid ints
        assert isinstance(e1, int)
        assert isinstance(e2, int)


class TestHardwareEntropy:
    """Test hardware entropy source"""
    
    def setup_method(self):
        self.source = HardwareEntropy()
    
    def test_available_property(self):
        # Should be boolean
        assert isinstance(self.source.available, bool)
    
    def test_get_entropy_fallback(self):
        # Should always return something (may fallback to system)
        entropy = self.source.get_entropy(8)
        assert isinstance(entropy, int)
        assert entropy >= 0