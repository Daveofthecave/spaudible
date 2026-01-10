# core/similarity_engine/vector_math.py
import numpy as np
from numba import njit, prange
from typing import List
from .weight_layers import WeightLayers

class VectorOps:
    """Optimized vector operations using Numba with mask and region support."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = np.float32

    def __init__(self, algorithm='cosine-euclidean', region_filter=1.0):
        self.algorithm = algorithm
        self.weight_layers = WeightLayers()
        self.baseline_weights = self.weight_layers.baseline_weights.astype(np.float32)
        self.genre_reduction = np.float32(self.weight_layers.genre_reduction)
        self.availability_boost = np.float32(self.weight_layers.availability_boost)
        self.genre_mask = np.zeros(32, dtype=np.bool_)
        self.genre_mask[19:32] = True  # Dimensions 20-32
        self.user_weights = np.ones(32, dtype=np.float32)
        self.region_filter = region_filter
        self.query_region = None  # Will be set before search

    def set_query_region(self, region: int):
        """Set the query region for filtering"""
        self.query_region = region

    def set_user_weights(self, weights: List[float]):
        """Set user-defined weights at runtime."""
        if len(weights) != 32:
            raise ValueError(f"Expected 32 weights, got {len(weights)}")
        self.user_weights = np.array(weights, dtype=np.float32)

    def reset_weights(self):
        """Reset weights to baseline values."""
        self.user_weights = np.ones(32, dtype=np.float32)        

    def compute_similarity(self, query: np.ndarray, vectors, masks, regions) -> np.ndarray:
        """Main entry point with mask and region support"""
        if self.algorithm == 'cosine':
            return self.masked_weighted_cosine_similarity(query, vectors, masks, regions)
        elif self.algorithm == 'cosine-euclidean':
            return self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks, regions)
        elif self.algorithm == 'euclidean':
            return self.masked_euclidean_similarity(query, vectors, masks, regions)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    @staticmethod
    def to_numpy_array(vector: List[float]) -> np.ndarray:
        """Convert Python list to NumPy array."""
        return np.array(vector, dtype=np.float32)

    def masked_weighted_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray, 
                                         masks: np.ndarray, regions: np.ndarray) -> np.ndarray:
        """Optimized cosine similarity with Numba and mask/region handling"""
        return self._numba_cosine_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            masks.ast(np.uint32),
            regions.astype(np.uint8),
            self.baseline_weights,
            self.genre_mask,
            self.availability_boost,
            self.genre_reduction,
            self.user_weights,
            self.region_filter,
            self.query_region if self.query_region is not None else 0
        )

    @staticmethod
    @njit(parallel=True)
    def _numba_cosine_similarity(query, vectors, masks, regions, 
                                baseline_weights, genre_mask, availability_boost, 
                                genre_reduction, user_weights, region_filter, query_region):
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
                
                weight = baseline_weights[j] * availability_boost * user_weights[j]
                if genre_mask[j] and not vector_has_genre:
                    weight *= adj_factor
                
                w_q = q_val * weight
                w_v = v_val * weight
                
                dot += w_q * w_v
                norm_u += w_q * w_q
                norm_v += w_v * w_v
            
            norm_product = max(np.sqrt(norm_u) * np.sqrt(norm_v), 1e-9)
            similarity = dot / norm_product
            
            # Apply region filtering LAST
            if region_filter > 0 and regions[i] != query_region:
                if region_filter == 1.0:
                    similarity = -10000.0  # Hard exclusion
                else:
                    similarity *= (1 - region_filter)  # Soft penalty
            
            similarities[i] = similarity
        
        return similarities

    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, vectors: np.ndarray, 
                                                   masks: np.ndarray, regions: np.ndarray) -> np.ndarray:
        """Optimized hybrid similarity with Numba and mask/region handling"""
        return self._numba_hybrid_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            masks.astype(np.uint32),
            regions.astype(np.uint8),
            self.baseline_weights,
            self.user_weights,
            self.genre_mask,
            self.availability_boost,
            self.genre_reduction,
            self.region_filter,
            self.query_region if self.query_region is not None else 0
        )

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_hybrid_similarity(
        query: np.ndarray, 
        vectors: np.ndarray, 
        masks: np.ndarray,
        regions: np.ndarray,
        baseline_weights: np.ndarray, 
        user_weights: np.ndarray,
        genre_mask: np.ndarray,
        availability_boost: float, 
        genre_reduction: float,
        region_filter: float,
        query_region: int
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
            
            # Apply region filtering first
            if region_filter > 0 and regions[i] != query_region:
                if region_filter == 1.0:
                    # Hard exclusion
                    similarities[i] = -10000.0
                    continue
                else:
                    # Will apply penalty at the end
                    penalty_factor = 1 - region_filter
            else:
                penalty_factor = 1.0
            
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
                
                # Include user_weights in calculation
                weight = baseline_weights[j] * availability_boost * user_weights[j]
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
            similarity = cosine_sim * euclidean_sim
            
            # Apply penalty if needed
            similarity *= penalty_factor
            
            similarities[i] = similarity
        
        return similarities

    def masked_euclidean_similarity(self, query: np.ndarray, vectors: np.ndarray, 
                                   masks: np.ndarray, regions: np.ndarray) -> np.ndarray:
        """Optimized Euclidean similarity with Numba and mask/region handling"""
        return self._numba_euclidean_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            masks.astype(np.uint32),
            regions.astype(np.uint8),
            self.user_weights,
            self.region_filter,
            self.query_region if self.query_region is not None else 0
        )

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _numba_euclidean_similarity(query: np.ndarray, vectors: np.ndarray, masks: np.ndarray,
                                   regions: np.ndarray, user_weights: np.ndarray, 
                                   region_filter: float, query_region: int) -> np.ndarray:
        n = vectors.shape[0]
        similarities = np.empty(n, dtype=np.float32)
        
        for i in prange(n):
            mask = masks[i]
            vector = vectors[i]
            sq_diff = 0.0
            valid_dims = 0
            
            # Apply region filtering first
            if region_filter > 0 and regions[i] != query_region:
                if region_filter == 1.0:
                    # Hard exclusion
                    similarities[i] = -10000.0
                    continue
                else:
                    # Will apply penalty at the end
                    penalty_factor = 1 - region_filter
            else:
                penalty_factor = 1.0
            
            for j in range(32):
                # Skip if dimension is invalid in mask
                if not ((mask >> j) & 1):
                    continue
                    
                q_val = query[j]
                v_val = vector[j]
                
                if q_val == -1 or v_val == -1:
                    continue
                    
                diff = q_val - v_val
                sq_diff += (diff * diff) * user_weights[j]
                valid_dims += 1
            
            if valid_dims == 0:
                similarity = 0.0
            else:
                distance = np.sqrt(sq_diff)
                similarity = 1.0 / (1.0 + distance)
            
            # Apply penalty if needed
            similarity *= penalty_factor
            
            similarities[i] = similarity
        
        return similarities
