# core/similarity_engine/vector_io.py
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from core.utilities.gpu_utils import recommend_max_batch_size
from config import VRAM_SAFETY_FACTOR, PathConfig
import struct

class VectorReaderGPU:
    """GPU-optimized vector reader with VRAM-aware batch sizing and mask support."""
    
    VECTOR_DIMENSIONS = 32
    BYTES_PER_VECTOR = 128  # 32 * 4 bytes
    BYTES_PER_MASK = 4      # 32-bit unsigned integer
    DTYPE = torch.float32
    
    def __init__(self, vectors_path: Optional[str] = None, 
                 masks_path: Optional[str] = None,
                 device="cuda", vram_scaling_factor_mb=None):
        # Use PathConfig defaults if not provided
        self.vectors_path = Path(vectors_path) if vectors_path else PathConfig.get_vector_file()
        self.masks_path = Path(masks_path) if masks_path else PathConfig.get_mask_file()
        self.device = device
        
        if not self.vectors_path.exists():
            raise FileNotFoundError(f"Vector file not found: {self.vectors_path}")
        if not self.masks_path.exists():
            raise FileNotFoundError(f"Mask file not found: {self.masks_path}")
        
        # Calculate vector count
        self.file_size = self.vectors_path.stat().st_size
        self.total_vectors = self.file_size // self.BYTES_PER_VECTOR
        
        # Auto-configure batch size based on VRAM
        self.max_batch_size = self._calculate_max_batch_size(vram_scaling_factor_mb)
        
        # Memory map on CPU
        self.mmap_cpu = torch.from_file(
            str(self.vectors_path),
            size=self.total_vectors * self.VECTOR_DIMENSIONS,
            dtype=self.DTYPE
        ).reshape(self.total_vectors, self.VECTOR_DIMENSIONS)
        
        # Memory map masks as int64 to prevent overflow
        self.masks_mmap = torch.from_file(
            str(self.masks_path),
            size=self.total_vectors,
            dtype=torch.int64  # Changed to int64
        )
        
        print(f"   GPU Vector Reader Initialized:")
        print(f"     Total track vectors: {self.total_vectors:,}")
        print(f"     Vector file: {self.vectors_path.name}")
        print(f"     Mask file: {self.masks_path.name}")
        print(f"     Max batch size: {self.max_batch_size:,} vectors")
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """
        Read a chunk of vectors from file.
        
        Args:
            start_idx: Starting vector index (0-based)
            num_vectors: Number of vectors to read
            
        Returns:
            NumPy array of shape (num_vectors, 32)
        """
        return self.vectors_mmap[start_idx:start_idx+num_vectors]
    
    def read_masks(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """
        Read a chunk of masks from file.
        
        Args:
            start_idx: Starting vector index (0-based)
            num_vectors: Number of masks to read
            
        Returns:
            NumPy array of uint32 masks
        """
        return self.masks_mmap[start_idx:start_idx+num_vectors]
    
    def read_vector_and_mask(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Read a single vector and its mask by index.
        
        Args:
            index: Vector index (0-based)
            
        Returns:
            Tuple (vector, mask)
        """
        return self.vectors_mmap[index], self.masks_mmap[index]
    
    def get_total_vectors(self) -> int:
        """Get total number of vectors in the file."""
        return self.total_vectors
    
    def read_single_vector(self, index: int) -> np.ndarray:
        """Read a single vector by index."""
        return self.read_chunk(index, 1)[0]
    
    def unpack_mask(self, mask: int) -> np.ndarray:
        """
        Unpack a bitmask into a boolean array.
        
        Args:
            mask: 32-bit unsigned integer mask
            
        Returns:
            Boolean array of shape (32,) where True indicates valid dimension
        """
        return np.array([(mask >> i) & 1 for i in range(32)], dtype=bool)
    
    def get_valid_dimensions(self, mask: int, query_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get valid dimension indices considering both vector mask and query validity.
        
        Args:
            mask: 32-bit mask for the vector
            query_vector: Optional query vector to consider its validity
            
        Returns:
            Array of valid dimension indices
        """
        vector_valid = self.unpack_mask(mask)
        
        if query_vector is None:
            return np.where(vector_valid)[0]
        
        query_valid = (query_vector != -1)
        return np.where(vector_valid & query_valid)[0]
