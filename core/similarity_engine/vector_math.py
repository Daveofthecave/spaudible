# core/similarity_engine/vector_math.py
"""
Vector math operations for similarity calculations.
"""
import numpy as np
import numba as nb
from typing import List
from .weight_layers import WeightLayers

class VectorOps:
    """Mathematical operations on vectors."""
    
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
        Compute masked cosine similarity between query and batch of vectors
        with validity masks and user-defined weights.
        """
        # Apply weights to query
        weighted_query = query * self.user_weights
        
        # Process in batches for cache efficiency
        batch_size = 10000
        n = vectors.shape[0]
        similarities = np.empty(n, dtype=np.float32)
        
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_vectors = vectors[start_idx:end_idx]
            batch_masks = masks[start_idx:end_idx]
            
            # Use optimized function
            similarities[start_idx:end_idx] = self._batch_masked_cosine(
                weighted_query, batch_vectors, batch_masks
            )
            
        return similarities

    def _batch_masked_cosine(self, weighted_query, vectors, masks):
        """Compute cosine similarity for a batch of vectors"""
        n = vectors.shape[0]
        results = np.zeros(n, dtype=np.float32)
        
        # Precompute weights for vectors
        weighted_vectors = vectors * self.user_weights
        
        # Process each vector
        for i in range(n):
            mask = masks[i]
            valid_dims = self._get_valid_dims(mask)
            
            if valid_dims.size > 0:
                q_valid = weighted_query[valid_dims]
                v_valid = weighted_vectors[i, valid_dims]
                
                # Compute cosine similarity
                dot = 0.0
                norm_q = 0.0
                norm_v = 0.0
                
                for j in range(q_valid.shape[0]):
                    dot += q_valid[j] * v_valid[j]
                    norm_q += q_valid[j] * q_valid[j]
                    norm_v += v_valid[j] * v_valid[j]
                
                norm_q = np.sqrt(norm_q)
                norm_v = np.sqrt(norm_v)
                
                if norm_q > 0 and norm_v > 0:
                    results[i] = dot / (norm_q * norm_v)
        
        return results

    @staticmethod
    @nb.njit(fastmath=True)
    def _get_valid_dims(mask: int) -> np.ndarray:
        """Get valid dimensions from mask using Numba acceleration"""
        valid_dims = np.empty(32, dtype=np.bool_)
        for j in range(32):
            valid_dims[j] = (mask >> j) & 1
        return np.where(valid_dims)[0]

    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, vectors, masks) -> np.ndarray:
        """
        Compute hybrid cosine-euclidean similarity with validity masks.
        """
        # Compute both metrics
        cosine_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        euclidean_sim = self.masked_euclidean_similarity(query, vectors, masks)
        
        # Combine results
        return cosine_sim * euclidean_sim

    def masked_euclidean_similarity(self, query: np.ndarray, vectors, masks) -> np.ndarray:
        """Euclidean similarity with masking"""
        # Apply weights to query
        weighted_query = query * self.user_weights
        
        # Process in batches for cache efficiency
        batch_size = 10000
        n = vectors.shape[0]
        similarities = np.empty(n, dtype=np.float32)
        
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_vectors = vectors[start_idx:end_idx]
            batch_masks = masks[start_idx:end_idx]
            
            # Use optimized function
            similarities[start_idx:end_idx] = self._batch_masked_euclidean(
                weighted_query, batch_vectors, batch_masks
            )
            
        return similarities

    def _batch_masked_euclidean(self, weighted_query, vectors, masks):
        """Compute Euclidean similarity for a batch of vectors"""
        n = vectors.shape[0]
        results = np.zeros(n, dtype=np.float32)
        
        # Precompute weights for vectors
        weighted_vectors = vectors * self.user_weights
        
        # Process each vector
        for i in range(n):
            mask = masks[i]
            valid_dims = self._get_valid_dims(mask)
            
            if valid_dims.size > 0:
                q_valid = weighted_query[valid_dims]
                v_valid = weighted_vectors[i, valid_dims]
                
                # Compute Euclidean distance
                sq_diff = 0.0
                for j in range(q_valid.shape[0]):
                    diff = q_valid[j] - v_valid[j]
                    sq_diff += diff * diff
                
                distance = np.sqrt(sq_diff)
                results[i] = 1.0 / (1.0 + distance)
        
        return results
    
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
