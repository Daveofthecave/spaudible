# core/similarity_engine/vector_io.py
"""
Reading vectors from binary files.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import struct

class VectorReader:
    """Read vectors from binary files efficiently."""
    
    VECTOR_DIMENSIONS = 32
    BYTES_PER_VECTOR = 128  # 32 floats * 4 bytes each
    DTYPE = np.float32
    
    def __init__(self, vectors_path: str = "data/vectors/track_vectors.bin"):
        """
        Initialize vector reader.
        
        Args:
            vectors_path: Path to track_vectors.bin file
        """
        self.vectors_path = Path(vectors_path)
        
        if not self.vectors_path.exists():
            raise FileNotFoundError(f"Vector file not found: {self.vectors_path}")
        
        # Calculate total vectors
        file_size = self.vectors_path.stat().st_size
        self.total_vectors = file_size // self.BYTES_PER_VECTOR
        
        # print(f"   Vector Reader Initialized:")
        # print(f"     Total track vectors: {self.total_vectors:,}")
        # print(f"     File size: {file_size / (1024**3):.1f} GB")
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """
        Read a chunk of vectors from file.
        
        Args:
            start_idx: Starting vector index (0-based)
            num_vectors: Number of vectors to read
            
        Returns:
            NumPy array of shape (num_vectors, 32)
        """
        with open(self.vectors_path, 'rb') as f:
            offset = start_idx * self.BYTES_PER_VECTOR
            f.seek(offset)
            
            bytes_to_read = num_vectors * self.BYTES_PER_VECTOR
            chunk_bytes = f.read(bytes_to_read)
            
            # Convert to NumPy array efficiently
            return np.frombuffer(chunk_bytes, dtype=self.DTYPE).reshape(-1, self.VECTOR_DIMENSIONS)
    
    def get_total_vectors(self) -> int:
        """Get total number of vectors in the file."""
        return self.total_vectors
    
    def read_single_vector(self, index: int) -> np.ndarray:
        """Read a single vector by index."""
        return self.read_chunk(index, 1)[0]
