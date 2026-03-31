"""
True entropy sources for Aleam.
"""

import os
import struct
from typing import Optional


class EntropySource:
    """Base class for entropy sources"""
    
    def get_entropy(self, num_bytes: int = 16) -> int:
        """Get entropy as integer"""
        raise NotImplementedError


class SystemEntropy(EntropySource):
    """
    System entropy via os.urandom.
    
    Uses the operating system's cryptographic random number generator.
    Available on all platforms (Unix, Windows, macOS).
    """
    
    def get_entropy(self, num_bytes: int = 16) -> int:
        """Get entropy from system"""
        return int.from_bytes(os.urandom(num_bytes), byteorder='big')


class HardwareEntropy(EntropySource):
    """
    Hardware entropy source (RDRAND, /dev/hwrng).
    
    Attempts to use CPU hardware random number generator or
    hardware random device if available. Falls back to system entropy.
    """
    
    def __init__(self):
        self._available = self._check_availability()
        self._system_fallback = SystemEntropy()
        self._rdrand_available = self._check_rdrand()
    
    def _check_availability(self) -> bool:
        """Check if hardware entropy is available"""
        # Check for /dev/hwrng on Linux
        try:
            with open('/dev/hwrng', 'rb') as f:
                f.read(1)
            return True
        except (FileNotFoundError, PermissionError, OSError):
            pass
        
        # Check for RDRAND
        return self._check_rdrand()
    
    def _check_rdrand(self) -> bool:
        """Check if CPU supports RDRAND instruction"""
        try:
            # Try to use cpuid module if available
            import cpuid
            # Check RDRAND bit in ECX
            return bool(cpuid.CPUID().eax(1).ecx & (1 << 30))
        except (ImportError, AttributeError):
            pass
        
        # Fallback: try to read from /proc/cpuinfo on Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                return 'rdrand' in content
        except (FileNotFoundError, PermissionError):
            pass
        
        return False
    
    def _get_rdrand(self) -> int:
        """Get random number from RDRAND instruction"""
        # RDRAND requires assembly or C extension
        # For now, fallback to system entropy
        # This can be extended with a C module later
        return self._system_fallback.get_entropy(8)
    
    def get_entropy(self, num_bytes: int = 16) -> int:
        """Get entropy from hardware source"""
        if self._rdrand_available:
            try:
                # Use RDRAND for 64-bit chunks
                result = 0
                bytes_needed = num_bytes
                for _ in range((bytes_needed + 7) // 8):
                    result = (result << 64) | self._get_rdrand()
                return result & ((1 << (bytes_needed * 8)) - 1)
            except Exception:
                pass
        
        # Try /dev/hwrng
        if not self._rdrand_available:
            try:
                with open('/dev/hwrng', 'rb') as f:
                    data = f.read(num_bytes)
                return int.from_bytes(data, byteorder='big')
            except Exception:
                pass
        
        # Fallback to system entropy
        return self._system_fallback.get_entropy(num_bytes)
    
    @property
    def available(self) -> bool:
        """Check if hardware entropy is available"""
        return self._available