# core/similarity_engine/vector_math.py
"""
Vector math operations for similarity calculations.
"""
import numpy as np
from typing import List

class VectorOps:
    """Mathematical operations on vectors."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = np.float32
    
    @staticmethod
    def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and a batch of vectors.
        
        Args:
            query: Query vector shape (32,)
            vectors: Batch of vectors shape (N, 32)
            
        Returns:
            Array of similarities shape (N,)
        """
        # Ensure consistent float32 precision
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)

        # Vectorized dot product
        dots = np.dot(vectors, query)
        
        # Vectorized norms
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query)
        
        # Avoid division by zero
        mask = (vector_norms > 0) & (query_norm > 0)
        similarities = np.zeros(vectors.shape[0], dtype=VectorOps.DTYPE)
        similarities[mask] = dots[mask] / (vector_norms[mask] * query_norm)
        
        return similarities
    
    @staticmethod
    def validate_vector(vector: List[float]) -> bool:
        """Validate vector dimensions and values."""
        if len(vector) != VectorOps.VECTOR_DIMENSIONS:
            return False
        for v in vector:
            if not (-1 <= v <= 1):  # Validate range
                return False
        return True
    
    @staticmethod
    def to_numpy_array(vector: List[float]) -> np.ndarray:
        """Convert Python list to NumPy array."""
        return np.array(vector, dtype=VectorOps.DTYPE)
