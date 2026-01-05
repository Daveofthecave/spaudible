# core/similarity_engine/vector_io_gpu.py
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
        
        # Memory map masks as uint32 (4 bytes per mask)
        self.masks_mmap = torch.from_file(
            str(self.masks_path),
            size=self.total_vectors,
            dtype=torch.int32  # Corrected to int32
        )
        
        print(f"   GPU Vector Reader Initialized:")
        print(f"     Total track vectors: {self.total_vectors:,}")
        print(f"     Vector file: {self.vectors_path.name}")
        print(f"     Mask file: {self.masks_path.name}")
        print(f"     Max batch size: {self.max_batch_size:,} vectors")
    
    def _calculate_max_batch_size(self, vram_scaling_factor_mb):
        """Calculate optimal batch size based on available VRAM."""
        if vram_scaling_factor_mb is None:
            # Auto-detect VRAM
            max_batch = recommend_max_batch_size(
                vector_dim=self.VECTOR_DIMENSIONS,
                dtype_size=4,  # float32 = 4 bytes
                safety_factor=VRAM_SAFETY_FACTOR
            )
            return max_batch
        
        # Use manual configuration
        bytes_per_vector = 32 * 4
        return int((vram_scaling_factor_mb * 1024**2) / bytes_per_vector)
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read full chunk by breaking into sub-batches"""
        vectors = []
        processed = 0
        
        while processed < num_vectors:
            # Calculate remaining vectors
            remaining = num_vectors - processed
            read_size = min(remaining, self.max_batch_size)
            
            # Read sub-batch
            sub_start = start_idx + processed
            sub_vectors = self.mmap_cpu[sub_start:sub_start+read_size].to(self.device)
            vectors.append(sub_vectors)
            
            processed += read_size
        
        # Combine all sub-batches
        return torch.cat(vectors)
    
    def read_masks(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read masks for a chunk of vectors"""
        # Read from CPU memory map
        masks_cpu = self.masks_mmap[start_idx:start_idx+num_vectors]
        
        # Convert to GPU tensor
        return masks_cpu.to(self.device)
    
    def read_vector_and_mask(self, index: int) -> Tuple[torch.Tensor, int]:
        """Read a single vector and mask by index"""
        vector = self.mmap_cpu[index].to(self.device)
        mask = self.masks_mmap[index].item()
        return vector, mask
    
    def get_total_vectors(self) -> int:
        return self.total_vectors
    
    def get_max_batch_size(self) -> int:
        return self.max_batch_size
    
    def unpack_mask(self, mask: int) -> torch.Tensor:
        """
        Unpack a bitmask into a boolean tensor.
        
        Args:
            mask: 32-bit unsigned integer mask
            
        Returns:
            Boolean tensor of shape (32,) where True indicates valid dimension
        """
        return torch.tensor([(mask >> i) & 1 for i in range(32)], 
                           dtype=torch.bool, device=self.device)
    
    def get_valid_dimensions(self, mask: int, query_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get valid dimension indices considering both vector mask and query validity.
        
        Args:
            mask: 32-bit mask for the vector
            query_vector: Optional query vector to consider its validity
            
        Returns:
            Tensor of valid dimension indices
        """
        vector_valid = self.unpack_mask(mask)
        
        if query_vector is None:
            return torch.where(vector_valid)[0]
        
        query_valid = (query_vector != -1)
        return torch.where(vector_valid & query_valid)[0]
