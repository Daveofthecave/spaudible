# core/similarity_engine/vector_math_gpu.py
"""
GPU-accelerated vector operations using PyTorch.
Returns tensors (versus numpy arrays) for pipeline compatibility.
"""
import torch
import numpy as np
from .weight_layers import WeightLayers
from typing import List
from core.utilities.config_manager import config_manager

class VectorOpsGPU:
    """GPU-accelerated vector operations using PyTorch."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = torch.float32

    def __init__(self, device="cuda"):
        self.device = device
        
        # Load weights
        self.weight_layers = WeightLayers()
        
        # Load weights directly from ConfigManager
        self.user_weights = torch.tensor(
            config_manager.get_weights(), 
            dtype=self.DTYPE,
            device=self.device
        )
        self.availability_boost = torch.tensor(
            self.weight_layers.availability_boost,
            dtype=self.DTYPE,
            device=self.device
        )
        self.genre_reduction = torch.tensor(
            self.weight_layers.genre_reduction,
            dtype=self.DTYPE,
            device=self.device
        )
        
        self.genre_mask = torch.zeros(32, dtype=torch.bool, device=self.device)
        self.genre_mask[19:32] = True
    
    def set_user_weights(self, weights: List[float]):
        """Set user-defined weights."""
        if len(weights) != 32:
            raise ValueError(f"Expected 32 weights, got {len(weights)}")
        self.user_weights = torch.tensor(weights, dtype=self.DTYPE, device=self.device)
    
    def reset_weights(self):
        """Reset weights to baseline (from Config defaults)."""
        self.user_weights = torch.tensor(
            config_manager.DEFAULT_VECTOR_WEIGHTS, 
            dtype=self.DTYPE, 
            device=self.device
        )

    # Add cached bitmask to avoid recreation
    @staticmethod
    def _get_bitmask(device):
        """Cache the bitmask tensor to avoid recreation."""
        if not hasattr(VectorOpsGPU, '_bitmask_cache'):
            VectorOpsGPU._bitmask_cache = torch.tensor(
                [1 << i for i in range(32)], 
                dtype=torch.int64, 
                device=device
            )
        return VectorOpsGPU._bitmask_cache
    
    def masked_weighted_cosine_similarity(self, query: np.ndarray, vectors: torch.Tensor, 
                                          masks: torch.Tensor) -> torch.Tensor:
        """Compute masked cosine similarity with proper NaN handling."""
        # Convert query once and expand once
        query_t = torch.as_tensor(query, dtype=torch.float32, device=vectors.device)
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0)
        
        # Expand query to match batch size
        batch_size = vectors.shape[0]
        query_expanded = query_t.expand(batch_size, -1)
        
        # Validity masks
        query_valid = (query_expanded != -1)
        vector_valid = self._unpack_masks(masks)
        valid_mask = query_valid & vector_valid
        
        # Check for genre presence
        query_has_genre = (query_expanded[:, self.genre_mask] != -1).any(dim=1)
        vector_has_genre = (vectors[:, self.genre_mask] != -1).any(dim=1)
        
        # Apply genre reduction
        genre_adj = torch.where(
            query_has_genre & vector_has_genre,
            torch.tensor(1.0, device=self.device),
            self.genre_reduction
        ).unsqueeze(1)
        
        weights = self.user_weights * self.availability_boost
        
        # Apply genre adjustment
        weights_expanded = weights.unsqueeze(0).expand(batch_size, -1)
        weights_adj = torch.where(
            self.genre_mask.unsqueeze(0) & ~vector_has_genre.unsqueeze(1),
            weights_expanded * genre_adj,
            weights_expanded
        )
        
        # Apply weights
        weighted_query = query_expanded * weights.unsqueeze(0)
        weighted_vectors = vectors * weights_adj
        
        # Apply validity mask to zero out invalid dimensions
        weighted_query = weighted_query * valid_mask.float()
        weighted_vectors = weighted_vectors * valid_mask.float()
        
        # Compute dot product (now only valid dimensions contribute)
        dot = torch.sum(weighted_query * weighted_vectors, dim=1)
        
        # Compute norms
        norm_q = torch.norm(weighted_query, dim=1)
        norm_v = torch.norm(weighted_vectors, dim=1)
        
        # Prevent division by zero and NaN
        norm_product = norm_q * norm_v
        valid_norms = norm_product > 1e-9
        
        sim = torch.where(
            valid_norms,
            dot / torch.clamp(norm_product, min=1e-9),
            torch.tensor(0.0, device=self.device)
        )
        
        # Handle any remaining NaNs
        sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0)
        
        return sim
    
    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, 
                                                   vectors: torch.Tensor, 
                                                   masks: torch.Tensor) -> torch.Tensor:
        """Hybrid cosine-euclidean similarity."""
        # Compute cosine part
        cosine_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        
        # Convert query once and expand once
        query_t = torch.as_tensor(query, dtype=torch.float32, device=vectors.device)
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0)
        
        batch_size = vectors.shape[0]
        query_expanded = query_t.expand(batch_size, -1)
        
        # Validity masks
        query_valid = (query_expanded != -1)
        vector_valid = self._unpack_masks(masks)
        valid_mask = query_valid & vector_valid
        
        # Compute weighted differences
        weights = self.user_weights * self.availability_boost
        weighted_diff = (query_expanded - vectors) * weights.unsqueeze(0)
        
        # Apply validity mask to zero out invalid dimensions
        weighted_diff = weighted_diff * valid_mask.float()
        
        # Compute Euclidean distance
        euclidean_dist = torch.norm(weighted_diff, dim=1)
        
        # Convert to similarity
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine
        hybrid_sim = cosine_sim * euclidean_sim
        
        # Ensure no NaNs
        hybrid_sim = torch.nan_to_num(hybrid_sim, nan=0.0, posinf=1.0, neginf=0.0)
        
        return hybrid_sim
    
    def masked_euclidean_similarity(self, query: np.ndarray, vectors: torch.Tensor, 
                                  masks: torch.Tensor) -> torch.Tensor:
        """Euclidean similarity with NaN protection."""
        query_t = torch.as_tensor(query, dtype=torch.float32, device=vectors.device)
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0)
        
        batch_size = vectors.shape[0]
        query_expanded = query_t.expand(batch_size, -1)
        
        # Validity masks
        query_valid = (query_expanded != -1)
        vector_valid = self._unpack_masks(masks)
        valid_mask = query_valid & vector_valid
        
        # Compute weighted differences (no genre adjustment for pure Euclidean)
        weights = self.user_weights
        weighted_diff = (query_expanded - vectors) * weights.unsqueeze(0)
        
        # Apply validity mask to zero out invalid dimensions
        weighted_diff = weighted_diff * valid_mask.float()
        
        # Compute Euclidean distance
        euclidean_dist = torch.norm(weighted_diff, dim=1)
        
        # Convert to similarity
        sim = 1.0 / (1.0 + euclidean_dist)
        
        # Handle NaNs
        sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0)
        
        return sim
    
    def fused_similarity(self, query: np.ndarray, vectors: torch.Tensor, 
                        masks: torch.Tensor, regions: torch.Tensor,
                        query_region: int, region_strength: float, 
                        algorithm: str = 'cosine-euclidean') -> torch.Tensor:
        """Combined feature + region similarity."""
        # Get base similarity
        if algorithm == 'cosine':
            feature_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        elif algorithm == 'euclidean':
            feature_sim = self.masked_euclidean_similarity(query, vectors, masks)
        else:  # cosine-euclidean
            feature_sim = self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks)
        
        # Region match penalty
        if query_region >= 0 and region_strength > 0.0:
            region_match = (regions == query_region).float()
            region_penalty = torch.where(
                region_match == 1,
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0 - region_strength, device=self.device)
            )
            return feature_sim * region_penalty
        
        return feature_sim
    
    # Memory-efficient mask unpacking with caching
    def _unpack_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Unpack batch of int32 masks into boolean tensors of shape [n, 32].
        Each bit in the int32 becomes a boolean True/False in the output.
        """
        # Use cached bitmask to avoid recreation
        bitmask = self._get_bitmask(masks.device)
        
        # Reshape for broadcasting and use memory-efficient bitwise operations
        masks_reshaped = masks.view(-1, 1)  # Shape: [n, 1]
        
        # torch.bitwise_and produces int64 result, convert directly to bool
        # This avoids storing a large int64 intermediate
        result = torch.bitwise_and(masks_reshaped, bitmask).to(torch.bool)
        
        return result
