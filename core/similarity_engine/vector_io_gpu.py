# core/similarity_engine/vector_io_gpu.py
import torch
import numpy as np
from pathlib import Path

class VectorReaderGPU:
    """GPU-optimized vector reader with direct CUDA loading."""
    
    VECTOR_DIMENSIONS = 32
    BYTES_PER_VECTOR = 128  # 32 * 4 bytes
    DTYPE = torch.float32
    
    def __init__(self, vectors_path: str, device="cuda", max_gpu_mb=2048):
        self.vectors_path = Path(vectors_path)
        self.device = device
        self.max_gpu_mb = max_gpu_mb  # Max GPU memory to use (MB)
        
        if not self.vectors_path.exists():
            raise FileNotFoundError(f"Vector file not found: {self.vectors_path}")
        
        # Calculate vector count
        self.file_size = self.vectors_path.stat().st_size
        self.total_vectors = self.file_size // self.BYTES_PER_VECTOR
        
        # Calculate max vectors per batch based on GPU memory
        bytes_per_vector = 32 * 4  # 32 floats * 4 bytes each
        self.max_batch_size = int((max_gpu_mb * 1024**2) / bytes_per_vector)
        
        # Memory map on CPU
        self.mmap_cpu = torch.from_file(
            str(self.vectors_path),
            size=self.total_vectors * self.VECTOR_DIMENSIONS,
            dtype=self.DTYPE
        ).reshape(self.total_vectors, self.VECTOR_DIMENSIONS)
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read chunk and transfer to GPU."""
        end_idx = start_idx + num_vectors
        # Limit batch size to GPU capacity
        actual_size = min(num_vectors, self.max_batch_size)
        chunk = self.mmap_cpu[start_idx:start_idx+actual_size]
        return chunk.to(self.device)
    
    def get_total_vectors(self) -> int:
        return self.total_vectors
    
    def get_max_batch_size(self) -> int:
        return self.max_batch_size
