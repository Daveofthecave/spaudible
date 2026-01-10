# core/similarity_engine/vector_io_gpu.py
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from config import PathConfig

class VectorReaderGPU:
    """GPU-optimized vector reader with VRAM-aware batch sizing and mask support."""
    
    VECTOR_DIMENSIONS = 32
    BYTES_PER_VECTOR = 128  # 32 * 4 bytes
    BYTES_PER_MASK = 4      # 4 bytes per mask (uint32)
    DTYPE = torch.float32
    
    def __init__(self, vectors_path: Optional[str] = None, 
                 masks_path: Optional[str] = None,
                 device="cuda", vram_scaling_factor_mb=None):
        # Use PathConfig defaults if not provided
        self.vectors_path = Path(vectors_path) if vectors_path else PathConfig.get_vector_file()
        self.masks_path = Path(masks_path) if masks_path else PathConfig.get_mask_file()
        self.device = device
        
        # Verify paths exist
        if not self.vectors_path.exists():
            raise FileNotFoundError(f"Vector file not found: {self.vectors_path}")
        if not self.masks_path.exists():
            raise FileNotFoundError(f"Mask file not found: {self.masks_path}")
        
        # Get actual file sizes
        vector_file_size = self.vectors_path.stat().st_size
        mask_file_size = self.masks_path.stat().st_size
        
        # Calculate vector count from vectors file
        self.total_vectors = vector_file_size // self.BYTES_PER_VECTOR
        
        # Verify mask file size matches vector count
        expected_mask_size = self.total_vectors * self.BYTES_PER_MASK
        if mask_file_size != expected_mask_size:
            raise ValueError(
                f"Mask file size mismatch: Expected {expected_mask_size} bytes, "
                f"got {mask_file_size} bytes. Vector count: {self.total_vectors}"
            )
        
        # Auto-configure batch size based on VRAM
        self.max_batch_size = self._calculate_max_batch_size(vram_scaling_factor_mb)
        
        # Memory map vectors on CPU
        self.mmap_cpu = torch.from_file(
            str(self.vectors_path),
            size=self.total_vectors * self.VECTOR_DIMENSIONS,
            dtype=self.DTYPE
        ).reshape(self.total_vectors, self.VECTOR_DIMENSIONS)
        
        # Memory map masks as uint32 (4 bytes per mask)
        # Use memory-mapped file instead of loading entire file
        self.masks_fd = os.open(str(self.masks_path), os.O_RDONLY)
        self.masks_size = mask_file_size
        
        # print(f"   GPU Vector Reader Initialized:")
        # print(f"     Total track vectors: {self.total_vectors:,}")
        # print(f"     Vector file: {self.vectors_path.name} ({vector_file_size/(1024**3):.2f} GB)")
        # print(f"     Mask file: {self.masks_path.name} ({mask_file_size/(1024**3):.2f} GB)")
        # print(f"     Max batch size: {self.max_batch_size:,} vectors")
    
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
        # Calculate byte positions
        start_byte = start_idx * self.BYTES_PER_MASK
        num_bytes = num_vectors * self.BYTES_PER_MASK
        
        # Validate bounds
        if start_byte + num_bytes > self.masks_size:
            raise ValueError(
                f"Mask read out of bounds: {start_byte} + {num_bytes} > {self.masks_size}"
            )
        
        # Read directly from file descriptor
        os.lseek(self.masks_fd, start_byte, os.SEEK_SET)
        mask_data = os.read(self.masks_fd, num_bytes)
        
        # Convert to tensor
        masks_np = np.frombuffer(mask_data, dtype=np.uint32)
        masks_tensor = torch.tensor(masks_np, dtype=torch.int64, device=self.device)
        
        return masks_tensor
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'masks_fd'):
            os.close(self.masks_fd)
    
    def read_vector_and_mask(self, index: int) -> Tuple[torch.Tensor, int]:
        """Read a single vector and mask by index"""
        vector = self.mmap_cpu[index].to(self.device)
        
        # Read single mask
        mask_byte = index * self.BYTES_PER_MASK
        os.lseek(self.masks_fd, mask_byte, os.SEEK_SET)
        mask_data = os.read(self.masks_fd, self.BYTES_PER_MASK)
        mask = struct.unpack('I', mask_data)[0]
        
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

class RegionReaderGPU:
    """GPU region reader with robust bit unpacking and validation"""
    
    def __init__(self, region_path: str, total_vectors: int, device="cuda"):
        self.device = device
        self.region_path = Path(region_path)
        self.total_vectors = total_vectors
        self.regions = None
        
        if not self.region_path.exists():
            print(f"  ⚠️  Region file not found: {self.region_path}")
            return
        
        try:
            # Load entire file into GPU memory
            with open(self.region_path, 'rb') as f:
                data = f.read()
                self.regions = torch.tensor(
                    np.frombuffer(data, dtype=np.uint8),
                    device=device
                )
            
            # Validate region data size
            expected_size = (self.total_vectors * 3 + 7) // 8
            if len(self.regions) != expected_size:
                print(f"  ⚠️  Region file size mismatch: expected {expected_size} bytes, got {len(self.regions)}")
                print(f"  ⚠️  Region data may be incomplete. Using default regions.")
                self.regions = None
            else:
                print(f"  ✅ Preloaded region data: {len(self.regions)/1e6:.1f}M regions")
        except Exception as e:
            print(f"  ❗ Error loading region data: {e}")
            self.regions = None
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read a chunk of region indices with guaranteed size and bounds checking"""
        # Create default regions (Anglo)
        result = torch.zeros(num_vectors, dtype=torch.uint8, device=self.device)
        
        if self.regions is None:
            return result
        
        # Calculate byte range needed
        start_byte = (start_idx * 3) // 8
        end_byte = ((start_idx + num_vectors) * 3 + 7) // 8
        byte_count = end_byte - start_byte
        
        # Validate bounds
        if start_byte >= len(self.regions):
            return result
        
        # Adjust for partial read at end of file
        actual_byte_count = min(byte_count, len(self.regions) - start_byte)
        if actual_byte_count <= 0:
            return result
        
        # Get packed bytes
        packed_chunk = self.regions[start_byte:start_byte+actual_byte_count]
        
        # Create bit offsets tensor
        indices = torch.arange(num_vectors, device=self.device)
        bit_offsets = (indices * 3) % 8
        byte_offsets = (indices * 3) // 8 - start_byte
        
        # Create valid mask (within packed chunk bounds)
        valid_mask = (byte_offsets >= 0) & (byte_offsets < actual_byte_count)
        
        # Extract valid regions
        if torch.any(valid_mask):
            valid_byte_offsets = byte_offsets[valid_mask].long()
            valid_bit_offsets = bit_offsets[valid_mask]
            
            # Ensure byte offsets are within bounds
            valid_byte_offsets = torch.clamp(valid_byte_offsets, 0, actual_byte_count - 1)
            
            words = packed_chunk[valid_byte_offsets]
            
            # Ensure shift amounts are positive
            shift_amounts = 5 - valid_bit_offsets
            shift_amounts = torch.clamp(shift_amounts, 0, 7)
            
            regions_valid = (words >> shift_amounts) & 0x07
            result[valid_mask] = regions_valid.byte()
        
        return result
