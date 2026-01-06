# core/similarity_engine/vector_math.py
"""
Vector math operations for similarity calculations.
"""
import numpy as np
import simsimd as simd
from numba import njit, prange
from typing import List
from .weight_layers import WeightLayers

class VectorOps:
    """Optimized vector operations using SIMD and parallel processing."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = np.float32

    def __init__(self, algorithm='cosine-euclidean'):
        self.algorithm = algorithm
        self.weight_layers = WeightLayers()
        # Precompute constants
        self.baseline_weights = self.weight_layers.baseline_weights.astype(np.float32)
        self.availability_boost = np.float32(self.weight_layers.availability_boost)
        self.genre_reduction = np.float32(self.weight_layers.genre_reduction)
        self.user_weights = np.ones(32, dtype=np.float32)  # Default weights

    def set_user_weights(self, weights: List[float]):
        """Set user-defined weights at runtime."""
        if len(weights) != 32:
            raise ValueError(f"Expected 32 weights, got {len(weights)}")
        self.user_weights = np.array(weights, dtype=np.float32)

    def reset_weights(self):
        """Reset weights to baseline values."""
        self.user_weights = np.ones(32, dtype=np.float32)

    def compute_similarity(self, query: np.ndarray, vectors, masks) -> np.ndarray:
        """Main entry point for similarity calculations"""
        if self.algorithm == 'cosine':
            return self.masked_weighted_cosine_similarity(query, vectors, masks)
        elif self.algorithm == 'cosine-euclidean':
            return self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks)
        elif self.algorithm == 'euclidean':
            return self.masked_euclidean_similarity(query, vectors, masks)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def masked_weighted_cosine_similarity(self, query: np.ndarray, vectors, masks) -> np.ndarray:
        """
        Optimized cosine similarity using SIMD instructions and parallel processing.
        """
        # Apply weights to query
        weighted_query = query * self.user_weights
        
        # Process masks in bulk
        valid_masks = self._unpack_masks_bulk(masks)
        
        # Ensure arrays are contiguous
        vectors_cont = np.ascontiguousarray(vectors)
        valid_masks_cont = np.ascontiguousarray(valid_masks)
        
        # Use SIMD acceleration for cosine similarity
        return self._simd_cosine_similarity(weighted_query, vectors_cont, valid_masks_cont)

    def _simd_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray, valid_masks: np.ndarray) -> np.ndarray:
        """SIMD-accelerated cosine similarity calculation"""
        # Precompute query norm once
        query_valid = np.where(query != -1.0, query, 0.0)
        query_norm = np.sqrt(np.sum(query_valid**2))
        
        # Prepare vectors with masking
        vectors_masked = vectors * valid_masks
        
        # Compute dot products using SIMD - avoid transposing non-contiguous arrays
        dots = np.empty(vectors_masked.shape[0], dtype=np.float32)
        for i in range(vectors_masked.shape[0]):
            # Ensure vector is contiguous
            vec = np.ascontiguousarray(vectors_masked[i])
            dots[i] = simd.dot(query, vec)
        
        # Compute vector norms
        norms = np.sqrt(np.sum(vectors_masked**2, axis=1))
        
        # Avoid division by zero
        norms[norms == 0] = 1e-9
        similarities = dots / (query_norm * norms)
        
        return similarities

    def _unpack_masks_bulk(self, masks: np.ndarray) -> np.ndarray:
        """Bulk unpack masks to boolean arrays"""
        # Optimized with Numba for speed
        return self._numba_unpack_masks(masks)
    
    @staticmethod
    @njit(parallel=True)
    def _numba_unpack_masks(masks: np.ndarray) -> np.ndarray:
        """Numba-accelerated mask unpacking"""
        result = np.empty((len(masks), 32), dtype=np.bool_)
        for i in prange(len(masks)):
            mask = masks[i]
            for j in range(32):
                result[i, j] = (mask >> j) & 1
        return result

    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, vectors, masks) -> np.ndarray:
        """
        Optimized hybrid similarity metric.
        """
        cosine_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        euclidean_sim = self.masked_euclidean_similarity(query, vectors, masks)
        return cosine_sim * euclidean_sim

    def masked_euclidean_similarity(self, query: np.ndarray, vectors, masks) -> np.ndarray:
        """Euclidean similarity with masking"""
        # Apply weights
        weighted_query = query * self.user_weights
        
        # Process masks in bulk
        valid_masks = self._unpack_masks_bulk(masks)
        
        # Ensure arrays are contiguous
        vectors_cont = np.ascontiguousarray(vectors)
        valid_masks_cont = np.ascontiguousarray(valid_masks)
        
        # Compute squared Euclidean distance using SIMD
        distances = np.empty(vectors_cont.shape[0], dtype=np.float32)
        for i in range(vectors_cont.shape[0]):
            # Ensure vector is contiguous
            vec = np.ascontiguousarray(vectors_cont[i])
            distances[i] = simd.l2sq(weighted_query, vec)
        
        # Convert to similarity
        similarities = 1.0 / (1.0 + np.sqrt(distances))
        
        return similarities

    @staticmethod
    def validate_vector(vector: List[float]) -> bool:
        """Validate vector dimensions and values."""
        if len(vector) != 32:
            return False
        for v in vector:
            if not (-1 <= v <= 1):  # Validate range
                return False
        return True
    
    @staticmethod
    def to_numpy_array(vector: List[float]) -> np.ndarray:
        """Convert Python list to NumPy array."""
        return np.array(vector, dtype=VectorOps.DTYPE)
