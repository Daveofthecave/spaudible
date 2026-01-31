# core/similarity_engine/vector_math.py
import numpy as np
from numba import njit, prange
from typing import List
from .weight_layers import WeightLayers
from core.utilities.config_manager import config_manager

class VectorOps:
    """Optimized vector operations using Numba with mask support."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = np.float32

    def __init__(self, algorithm='cosine-euclidean'):
        self.algorithm = algorithm
        self.weight_layers = WeightLayers()
        
        # Load weights directly from ConfigManager
        self.user_weights = np.array(config_manager.get_weights(), dtype=np.float32)
        
        self.genre_reduction = np.float32(self.weight_layers.genre_reduction)
        self.availability_boost = np.float32(self.weight_layers.availability_boost)
        self.genre_mask = np.zeros(32, dtype=np.bool_)
        self.genre_mask[19:32] = True  # Dimensions 20-32

    def set_user_weights(self, weights: List[float]):
        """Set user-defined weights at runtime."""
        if len(weights) != 32:
            raise ValueError(f"Expected 32 weights, got {len(weights)}")
        self.user_weights = np.array(weights, dtype=np.float32)

    def reset_weights(self):
        """Reset weights to baseline values (from Config defaults)."""
        self.user_weights = np.array(config_manager.DEFAULT_VECTOR_WEIGHTS, dtype=np.float32)        

    def compute_similarity(self, query: np.ndarray, vectors, masks) -> np.ndarray:
        """Main entry point with mask support"""
        if self.algorithm == 'cosine':
            return self.masked_weighted_cosine_similarity(query, vectors, masks)
        elif self.algorithm == 'cosine-euclidean':
            return self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks)
        elif self.algorithm == 'euclidean':
            return self.masked_euclidean_similarity(query, vectors, masks)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    @staticmethod
    def to_numpy_array(vector: List[float]) -> np.ndarray:
        """Convert Python list to NumPy array."""
        return np.array(vector, dtype=np.float32)

    def masked_weighted_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Optimized cosine similarity with Numba and mask handling"""
        return self._numba_cosine_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            masks.astype(np.uint32),
            self.user_weights,
            self.genre_mask,
            self.availability_boost,
            self.genre_reduction
        )

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_cosine_similarity(query: np.ndarray, vectors: np.ndarray, masks: np.ndarray, 
                                 user_weights: np.ndarray,
                                 genre_mask: np.ndarray, availability_boost: float, 
                                 genre_reduction: float) -> np.ndarray:
        n = vectors.shape[0]
        similarities = np.empty(n, dtype=np.float32)
        query_has_genre = np.any(query[genre_mask] != -1)
        
        for i in prange(n):
            mask = masks[i]
            vector = vectors[i]
            dot = 0.0
            norm_u = 0.0
            norm_v = 0.0
            vector_has_genre = False
            
            # Check genre presence
            for j in range(32):
                if genre_mask[j] and vector[j] != -1:
                    vector_has_genre = True
                    break
            
            adj_factor = genre_reduction
            if query_has_genre and vector_has_genre:
                adj_factor = 1.0
            
            for j in range(32):
                # Skip if dimension is invalid in mask
                if not ((mask >> j) & 1):
                    continue
                    
                q_val = query[j]
                v_val = vector[j]
                
                if q_val == -1 or v_val == -1:
                    continue
                
                weight = user_weights[j] * availability_boost
                
                if genre_mask[j] and not vector_has_genre:
                    weight *= adj_factor
                
                w_q = q_val * weight
                w_v = v_val * weight
                
                dot += w_q * w_v
                norm_u += w_q * w_q
                norm_v += w_v * w_v
            
            norm_product = max(np.sqrt(norm_u) * np.sqrt(norm_v), 1e-9)
            similarities[i] = dot / norm_product
        
        return similarities

    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, vectors: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Optimized hybrid similarity with Numba and mask handling"""
        return self._numba_hybrid_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            masks.astype(np.uint32),
            self.user_weights,
            self.genre_mask,
            self.availability_boost,
            self.genre_reduction
        )

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_hybrid_similarity(
        query: np.ndarray, 
        vectors: np.ndarray, 
        masks: np.ndarray,
        user_weights: np.ndarray,
        genre_mask: np.ndarray,
        availability_boost: float, 
        genre_reduction: float
    ) -> np.ndarray:
        n = vectors.shape[0]
        similarities = np.empty(n, dtype=np.float32)
        query_has_genre = np.any(query[genre_mask] != -1)
        
        for i in prange(n):
            mask = masks[i]
            vector = vectors[i]
            dot = 0.0
            norm_u = 0.0
            norm_v = 0.0
            sq_diff = 0.0
            vector_has_genre = False
            
            # Check genre presence
            for j in range(32):
                if genre_mask[j] and vector[j] != -1:
                    vector_has_genre = True
                    break
            
            adj_factor = genre_reduction
            if query_has_genre and vector_has_genre:
                adj_factor = 1.0
            
            for j in range(32):
                # Skip if dimension is invalid in mask
                if not ((mask >> j) & 1):
                    continue
                    
                q_val = query[j]
                v_val = vector[j]
                
                if q_val == -1 or v_val == -1:
                    continue
                
                weight = user_weights[j] * availability_boost
                if genre_mask[j] and not vector_has_genre:
                    weight *= adj_factor
                
                w_q = q_val * weight
                w_v = v_val * weight
                
                dot += w_q * w_v
                norm_u += w_q * w_q
                norm_v += w_v * w_v
                sq_diff += (w_q - w_v) ** 2
            
            norm_product = max(np.sqrt(norm_u) * np.sqrt(norm_v), 1e-9)
            cosine_sim = dot / norm_product
            euclidean_sim = 1.0 / (1.0 + np.sqrt(sq_diff))
            similarities[i] = cosine_sim * euclidean_sim
        
        return similarities

    def masked_euclidean_similarity(self, query: np.ndarray, vectors: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Optimized Euclidean similarity with Numba and mask handling"""
        return self._numba_euclidean_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            masks.astype(np.uint32),
            self.user_weights
        )

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _numba_euclidean_similarity(query: np.ndarray, vectors: np.ndarray, masks: np.ndarray,
                                    user_weights: np.ndarray) -> np.ndarray:
        n = vectors.shape[0]
        similarities = np.empty(n, dtype=np.float32)
        
        for i in prange(n):
            mask = masks[i]
            vector = vectors[i]
            sq_diff = 0.0
            valid_dims = 0
            
            for j in range(32):
                # Skip if dimension is invalid in mask
                if not ((mask >> j) & 1):
                    continue
                    
                q_val = query[j]
                v_val = vector[j]
                
                if q_val == -1 or v_val == -1:
                    continue
                    
                diff = (q_val - v_val) * user_weights[j]
                sq_diff += diff * diff
                valid_dims += 1
            
            if valid_dims == 0:
                similarities[i] = 0.0
            else:
                distance = np.sqrt(sq_diff)
                similarities[i] = 1.0 / (1.0 + distance)
        
        return similarities

    def fused_similarity(self, query: np.ndarray, vectors: np.ndarray, 
                        masks: np.ndarray, regions: np.ndarray,
                        query_region: int, region_strength: float, 
                        algorithm: str = 'cosine-euclidean') -> np.ndarray:
        """Region-aware similarity for CPU."""
        # Get base similarity
        if algorithm == 'cosine':
            feature_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        elif algorithm == 'euclidean':
            feature_sim = self.masked_euclidean_similarity(query, vectors, masks)
        else:  # cosine-euclidean
            feature_sim = self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks)
        
        # Apply region penalty
        if query_region >= 0 and region_strength > 0.0:
            region_match = (regions == query_region).astype(np.float32)
            region_penalty = np.where(
                region_match == 1.0,
                1.0,
                1.0 - region_strength
            )
            return feature_sim * region_penalty
        
        return feature_sim
