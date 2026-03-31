"""
Tests for utility functions.
"""

import pytest
from aleam.utils import seed_free


class TestUtils:
    """Test utility functions"""
    
    def test_seed_free_raises(self):
        """Test that seed_free raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            seed_free()
    
    def test_seed_free_message(self):
        """Test the error message explains why"""
        with pytest.raises(NotImplementedError) as exc_info:
            seed_free()
        assert "Aleam uses true randomness" in str(exc_info.value)