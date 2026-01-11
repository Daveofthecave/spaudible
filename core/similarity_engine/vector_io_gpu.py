# core/similarity_engine/vector_io_gpu.py
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from core.utilities.gpu_utils import recommend_max_batch_size
from config import VRAM_SAFETY_FACTOR, PathConfig
import os

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
    """GPU-optimized region data reader with packed bit storage"""
    
    BITS_PER_REGION = 3
    BYTES_PER_REGION = BITS_PER_REGION / 8  # 0.375 bytes
    REGION_COUNT = 8  # 0-7 regions
    
    def __init__(self, region_path: Optional[Path] = None):
        """
        Initialize region reader.
        
        Args:
            region_path: Path to track_regions.bin file
        """
        self.region_path = region_path or PathConfig.VECTORS / "track_regions.bin"
        self.packed_data = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.total_regions = 0
        self.is_loaded = False
        
    def load(self) -> bool:
        """Load packed region data into GPU memory."""
        if not self.region_path.exists():
            print(f"Region file not found: {self.region_path}")
            return False
        
        try:
            # Get file size to calculate total regions
            file_size = self.region_path.stat().st_size
            self.total_regions = int(file_size / self.BYTES_PER_REGION)
            
            # Read entire file into memory
            with open(self.region_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Transfer to GPU tensor
            self.packed_data = torch.tensor(data, dtype=torch.uint8, device=self.device)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading region data: {e}")
            return False
    
    def get_region_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get regions for a batch of indices.
        
        Args:
            indices: Tensor of vector indices (0-based)
            
        Returns:
            Tensor of region codes (0-7) with same shape as indices
        """
        if not self.is_loaded:
            raise RuntimeError("Region data not loaded. Call load() first.")
        
        # Convert indices to long if needed
        indices = indices.to(dtype=torch.long, device=self.device)
        
        # Calculate byte positions
        byte_offsets = (indices * self.BITS_PER_REGION) // 8
        bit_offsets = (indices * self.BITS_PER_REGION) % 8
        
        # Convert to long
        byte_offsets = byte_offsets.long()
        bit_offsets = bit_offsets.long()
        
        # Get packed bytes containing the regions
        packed_bytes = self.packed_data[byte_offsets]
        
        # Create mask for different extraction cases
        mask_low = bit_offsets <= 5  # Bits within single byte
        mask_high = ~mask_low        # Bits span two bytes
        
        # Initialize output tensor
        regions = torch.zeros_like(indices, dtype=torch.long, device=self.device)  # CHANGED TO LONG
        
        # Case 1: All bits within a single byte
        if torch.any(mask_low):
            # Calculate shift amount (5 - bit_offset)
            shifts = 5 - bit_offsets[mask_low]
            
            # Extract 3-bit region value
            regions[mask_low] = (packed_bytes[mask_low].long() >> shifts) & 0x07
        
        # Case 2: Bits span two bytes
        if torch.any(mask_high):
            # Get next byte for cross-byte regions
            next_bytes = self.packed_data[byte_offsets[mask_high] + 1]
            
            # Calculate bits from first byte
            first_part = packed_bytes[mask_high].long() << (bit_offsets[mask_high] - 5)
            
            # Calculate bits from second byte
            second_part = next_bytes.long() >> (13 - bit_offsets[mask_high])
            
            # Combine and mask to 3 bits
            regions[mask_high] = (first_part | second_part) & 0x07
        
        return regions.to(torch.uint8)  # Convert to uint8 at the end

    def get_region(self, index: int) -> int:
        """
        Get region for a single index.
        
        Args:
            index: Vector index (0-based)
            
        Returns:
            Region code (0-7)
        """
        if not self.is_loaded:
            self.load()
        
        # Create tensor for single index
        index_tensor = torch.tensor([index], dtype=torch.long, device=self.device)
        return self.get_region_batch(index_tensor).item()
    
    def get_total_regions(self) -> int:
        """Get total number of regions (should match vector count)."""
        return self.total_regions
    
    def is_available(self) -> bool:
        """Check if region data is available."""
        return self.is_loaded
    
    def get_region_distribution(self, sample_size: int = 1000000) -> dict:
        """Get distribution of regions in the dataset."""
        if not self.is_loaded:
            self.load()
        
        # Create random sample indices
        indices = torch.randint(0, self.total_regions, (sample_size,), device=self.device)
        
        # Get regions for sample
        regions = self.get_region_batch(indices)
        
        # Count distribution
        counts = torch.bincount(regions, minlength=self.REGION_COUNT)
        return {
            region: count.item() / sample_size
            for region, count in enumerate(counts)
        }
    
    def __del__(self):
        """Clean up resources."""
        if self.packed_data is not None:
            del self.packed_data
