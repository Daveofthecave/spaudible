# core/similarity_engine/vector_io.py
"""
Vector Input/Output Operations (CPU)
"""
import numpy as np
import os
from pathlib import Path
import mmap

class VectorReader:
    """Optimized CPU vector reader with persistent memory mapping"""
    
    VECTOR_DIMENSIONS = 32
    BYTES_PER_VECTOR = 128  # 32 * 4 bytes
    BYTES_PER_MASK = 4      # 32-bit unsigned integer
    DTYPE = np.float32

    def __init__(self, vectors_path: str, masks_path: str):
        self.vectors_path = vectors_path
        self.masks_path = masks_path
        
        # Open files and create persistent memory maps
        self.vectors_file = open(vectors_path, 'rb')
        self.masks_file = open(masks_path, 'rb')
        
        self.vector_file_size = os.path.getsize(vectors_path)
        self.mask_file_size = os.path.getsize(masks_path)
        self.total_vectors = self.vector_file_size // self.BYTES_PER_VECTOR
        
        # Create memory maps
        self.vectors_mmap = mmap.mmap(
            self.vectors_file.fileno(), 
            self.vector_file_size, 
            access=mmap.ACCESS_READ
        )
        
        self.masks_mmap = mmap.mmap(
            self.masks_file.fileno(),
            self.mask_file_size,
            access=mmap.ACCESS_READ
        )
        
        # print(f"  📊 Memory-mapped vector file: {self.vector_file_size/(1024**3):.1f} GB")
        # print(f"  📊 Memory-mapped mask file: {self.mask_file_size/(1024**3):.1f} GB")

    def read_chunk(self, start_idx: int, num_vectors: int) -> np.ndarray:
        # Adjust num_vectors to not exceed file bounds
        actual_num = min(num_vectors, self.total_vectors - start_idx)
        if actual_num <= 0:
            return np.empty((0, self.VECTOR_DIMENSIONS), dtype=self.DTYPE)

        offset = start_idx * self.BYTES_PER_VECTOR
        return np.frombuffer(
            self.vectors_mmap,
            dtype=self.DTYPE,
            count=actual_num * self.VECTOR_DIMENSIONS,
            offset=offset
        ).reshape(actual_num, self.VECTOR_DIMENSIONS)

    def read_masks(self, start_idx: int, num_vectors: int) -> np.ndarray:
        actual_num = min(num_vectors, self.total_vectors - start_idx)
        if actual_num <= 0:
            return np.empty(0, dtype=np.uint32)

        offset = start_idx * self.BYTES_PER_MASK
        return np.frombuffer(
            self.masks_mmap,
            dtype=np.uint32,
            count=actual_num,
            offset=offset
        )

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'vectors_mmap'):
            self.vectors_mmap.close()
        if hasattr(self, 'masks_mmap'):
            self.masks_mmap.close()
        if hasattr(self, 'vectors_file'):
            self.vectors_file.close()
        if hasattr(self, 'masks_file'):
            self.masks_file.close()

    def get_total_vectors(self) -> int:
        return self.total_vectors

class RegionReader:
    """CPU-based region reader with full preloading and validation"""
    
    def __init__(self, region_path: str, total_vectors: int):
        self.region_path = Path(region_path)
        self.total_vectors = total_vectors
        self.regions = None
        
        if not self.region_path.exists():
            print(f"  ⚠️  Region file not found: {self.region_path}")
            return
        
        try:
            # Load entire file into memory
            with open(self.region_path, 'rb') as f:
                data = f.read()
                self.regions = np.frombuffer(data, dtype=np.uint8)
            
            # Validate region data size
            expected_size = (self.total_vectors * 3 + 7) // 8  # 3 bits per vector
            if len(self.regions) != expected_size:
                print(f"  ⚠️  Region file size mismatch: expected {expected_size} bytes, got {len(self.regions)}")
                print(f"  ⚠️  Region data may be incomplete. Using default regions.")
                self.regions = None
            else:
                print(f"  ✅ Preloaded region data: {len(self.regions)/1e6:.1f}M regions")
        except Exception as e:
            print(f"  ❗ Error loading region data: {e}")
            self.regions = None
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Read a chunk of region indices with guaranteed size"""
        # Create default regions (Anglo)
        result = np.zeros(num_vectors, dtype=np.uint8)
        
        if self.regions is None:
            return result
        
        # Calculate byte range needed
        start_byte = (start_idx * 3) // 8
        end_byte = ((start_idx + num_vectors) * 3 + 7) // 8
        byte_count = end_byte - start_byte
        
        # Validate bounds
        if start_byte >= len(self.regions):
            return result
        
        # Read packed bytes
        packed_bytes = self.regions[start_byte:start_byte+byte_count]
        
        # Unpack regions
        for i in range(num_vectors):
            byte_offset = (start_idx + i) * 3 // 8 - start_byte
            bit_offset = (start_idx + i) * 3 % 8
            
            if byte_offset < len(packed_bytes):
                byte_val = packed_bytes[byte_offset]
                region = (byte_val >> (5 - bit_offset)) & 0x07
                result[i] = region
            else:
                result[i] = 0  # Default to Anglo
        
        return result
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()
