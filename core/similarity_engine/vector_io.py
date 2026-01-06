# core/similarity_engine/vector_io.py
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
