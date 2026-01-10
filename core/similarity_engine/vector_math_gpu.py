# core/similarity_engine/vector_math_gpu.py
import torch
import numpy as np
from .weight_layers import WeightLayers
from typing import List

class VectorOpsGPU:
    """GPU-accelerated vector operations with efficient region filtering."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = torch.float32
    
    def __init__(self, device="cuda", region_filter=1.0):
        self.device_str = device
        self.device = torch.device(device)
        self.weight_layers = WeightLayers()
        self.baseline_weights = torch.tensor(
            self.weight_layers.baseline_weights, 
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
        self.user_weights = torch.ones(32, dtype=self.DTYPE, device=self.device)
        self.region_filter = region_filter
        self.query_region = None
        
        # Create genre mask
        self.genre_mask = torch.zeros(32, dtype=torch.bool, device=self.device)
        self.genre_mask[19:32] = True  # Dimensions 20-32
    
    def set_query_region(self, region: int):
        """Set query region for filtering."""
        self.query_region = region

    def set_user_weights(self, weights: List[float]):
        """Set user-defined weights at runtime."""
        if len(weights) != 32:
            raise ValueError(f"Expected 32 weights, got {len(weights)}")
        self.user_weights = torch.tensor(weights, dtype=self.DTYPE, device=self.device)

    def reset_weights(self):
        """Reset weights to baseline values."""
        self.user_weights = torch.ones(32, dtype=self.DTYPE, device=self.device)

    def masked_weighted_cosine_similarity(self, query: np.ndarray, vectors: torch.Tensor, 
                                         masks: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated cosine similarity with mask and region support"""
        return self._compute_similarity(query, vectors, masks, regions, "cosine")
    
    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, vectors: torch.Tensor, 
                                                   masks: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated hybrid cosine-euclidean similarity with region support"""
        return self._compute_similarity(query, vectors, masks, regions, "hybrid")
    
    def masked_euclidean_similarity(self, query: np.ndarray, vectors: torch.Tensor, 
                                   masks: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated Euclidean similarity with masking and region support"""
        return self._compute_similarity(query, vectors, masks, regions, "euclidean")
    
    def _compute_similarity(self, query: np.ndarray, vectors: torch.Tensor, 
                            masks: torch.Tensor, regions: torch.Tensor, 
                            algorithm: str) -> torch.Tensor:
        """
        Unified similarity computation with optimized region filtering
        """
        # Convert query to PyTorch tensor on the same device as vectors
        query_t = torch.tensor(query, dtype=vectors.dtype, device=vectors.device)
        
        # Ensure query has correct shape [1, 32]
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0)
        
        # Expand query to match batch size [batch_size, 32]
        if query_t.shape[0] != vectors.shape[0]:
            query_t = query_t.expand(vectors.shape[0], -1)
        
        # Unpack masks to boolean tensors [batch_size, 32]
        vector_valid = self._unpack_masks(masks)
        
        # Create valid mask for query [batch_size, 32]
        query_valid = (query_t != -1)
        
        # Combine validity masks [batch_size, 32]
        valid_mask = query_valid & vector_valid
        
        # Compute genre flags [batch_size]
        query_has_genre = torch.any(query_t[:, 19:32] != -1, dim=1)
        vector_has_genre = torch.any(vectors[:, 19:32] != -1, dim=1)
        
        # Compute adjustment factor [batch_size]
        adj_factor = torch.where(
            query_has_genre & vector_has_genre,
            torch.tensor(1.0, device=vectors.device),
            self.genre_reduction
        )
        
        # Create weight matrix [32]
        weights = self.baseline_weights * self.availability_boost * self.user_weights
        
        # Apply genre adjustment
        # Expand genre mask to [1, 32] for broadcasting
        genre_mask_exp = self.genre_mask.unsqueeze(0)
        # Expand vector_has_genre to [batch_size, 1] for broadcasting
        no_genre_mask = ~vector_has_genre.unsqueeze(1)
        
        # Create condition [batch_size, 32]
        condition = genre_mask_exp & no_genre_mask
        
        # Apply adjustment where condition is True
        weights_exp = weights.unsqueeze(0)
        adj_factor_exp = adj_factor.unsqueeze(1)
        adjusted_weights = torch.where(
            condition,
            weights_exp * adj_factor_exp,
            weights_exp
        )
        
        # Apply weights and mask invalid values
        weighted_query = query_t * adjusted_weights * valid_mask.float()
        weighted_vectors = vectors * adjusted_weights * valid_mask.float()
        
        # Compute cosine similarity
        dot = torch.sum(weighted_query * weighted_vectors, dim=1)
        norm_query = torch.norm(weighted_query, dim=1)
        norm_vectors = torch.norm(weighted_vectors, dim=1)
        
        # Avoid division by zero
        cosine_sim = dot / (norm_query * norm_vectors + 1e-9)
        cosine_sim = torch.nan_to_num(cosine_sim, nan=0.0)
        
        # Compute Euclidean distance if needed
        if algorithm in ["hybrid", "euclidean"]:
            diff = weighted_query - weighted_vectors
            sq_diff = diff ** 2
            sum_sq = torch.sum(sq_diff, dim=1)
            euclidean_dist = torch.sqrt(sum_sq)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine similarities based on algorithm
        if algorithm == "cosine":
            similarity = cosine_sim
        elif algorithm == "hybrid":
            similarity = cosine_sim * euclidean_sim
        else:  # euclidean
            similarity = euclidean_sim
        
        # Apply region filtering LAST to ensure it overrides everything
        if self.region_filter > 0 and self.query_region is not None:
            # Create region mask [batch_size]
            region_mask = (regions == self.query_region)
            
            if self.region_filter == 1.0:
                # Strict exclusion: set non-matching regions to very low similarity
                similarity = torch.where(
                    region_mask,
                    similarity,
                    torch.tensor(-10000.0, device=similarity.device)
                )
            else:
                # Apply penalty to non-matching regions
                similarity = torch.where(
                    region_mask,
                    similarity,
                    similarity * (1 - self.region_filter)
                )
        
        return similarity

    def _unpack_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Unpack batch of masks into boolean tensors.
        
        Args:
            masks: Tensor of uint32 masks (shape: [n])
            
        Returns:
            Boolean tensor of shape [n, 32]
        """
        # Create bitmask tensor with int64 dtype
        bitmask = torch.tensor([(1 << i) for i in range(32)], 
                            dtype=torch.int64,
                            device=masks.device)
        
        # Expand dimensions for broadcasting
        masks_exp = masks.unsqueeze(-1)  # [n, 1]
        bitmask_exp = bitmask.unsqueeze(0)  # [1, 32]
        
        # Compute validity
        return (masks_exp & bitmask_exp) != 0
