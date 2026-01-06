# core/similarity_engine/vector_io.py
import numpy as np
from pathlib import Path
import os

class VectorReader:  # CHANGED FROM VectorReaderGPU
    """CPU-based vector reader using memory mapping."""

    VECTOR_DIMENSIONS = 32
    BYTES_PER_VECTOR = 128  # 32 * 4 bytes
    BYTES_PER_MASK = 4      # 32-bit unsigned integer
    DTYPE = np.float32

    def __init__(self, vectors_path: str, masks_path: str):
        self.vectors_path = vectors_path
        self.masks_path = masks_path
        self.vector_file_size = os.path.getsize(vectors_path)
        self.mask_file_size = os.path.getsize(masks_path)
        self.total_vectors = self.vector_file_size // self.BYTES_PER_VECTOR

    def read_chunk(self, start_idx: int, num_vectors: int) -> np.ndarray:
        # Adjust num_vectors to not exceed file bounds
        actual_num = min(num_vectors, self.total_vectors - start_idx)
        if actual_num <= 0:
            return np.empty((0, self.VECTOR_DIMENSIONS), dtype=self.DTYPE)

        # Memory-map the vectors file
        mmap = np.memmap(self.vectors_path, dtype=self.DTYPE, mode='r', 
                        offset=start_idx * self.BYTES_PER_VECTOR,
                        shape=(actual_num, self.VECTOR_DIMENSIONS))
        # Copy to avoid leaving the file open
        return np.array(mmap)

    def read_masks(self, start_idx: int, num_vectors: int) -> np.ndarray:
        actual_num = min(num_vectors, self.total_vectors - start_idx)
        if actual_num <= 0:
            return np.empty(0, dtype=np.uint32)

        mmap = np.memmap(self.masks_path, dtype=np.uint32, mode='r',
                        offset=start_idx * self.BYTES_PER_MASK,
                        shape=(actual_num,))
        return np.array(mmap)

    def get_total_vectors(self) -> int:
        return self.total_vectors