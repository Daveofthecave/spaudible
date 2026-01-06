# core/similarity_engine/vector_math_gpu.py
import torch
import numpy as np
from .weight_layers import WeightLayers
from typing import List

class VectorOpsGPU:
    """GPU-accelerated vector operations using PyTorch with unified mask and weight support."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = torch.float32
    
    def __init__(self, device="cuda"):
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
    
    def set_user_weights(self, weights: List[float]):
        """Set user-defined weights at runtime."""
        if len(weights) != 32:
            raise ValueError(f"Expected 32 weights, got {len(weights)}")
        self.user_weights = torch.tensor(weights, dtype=self.DTYPE, device=self.device)

    def reset_weights(self):
        """Reset weights to baseline values."""
        self.user_weights = torch.ones(32, dtype=self.DTYPE, device=self.device)

    def masked_weighted_cosine_similarity(self, query: np.ndarray, vectors: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
        masks = masks.to(torch.int64)

        # Convert query to PyTorch tensor on the same device as vectors
        query_t = torch.tensor(query, dtype=torch.float32, device=vectors.device)
        
        # Ensure query has correct shape
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0).expand(vectors.shape[0], -1)
        
        # Create valid mask for query
        query_valid = (query_t != -1)
        
        # Unpack masks to boolean tensors (using int64)
        vector_valid = self._unpack_masks(masks.to(torch.int64))  # Convert to int64
        
        # Combine validity masks
        valid_mask = query_valid & vector_valid
        
        # Compute genre flags
        query_has_genre = torch.any(query_t[:, 19:32] != -1, dim=1)
        vector_has_genre = torch.any(vectors[:, 19:32] != -1, dim=1)
        
        # Compute adjustment factor
        adj_factor = torch.where(
            query_has_genre & vector_has_genre,
            torch.tensor(1.0, device=vectors.device),
            self.genre_reduction
        )
        
        # Create weight matrix
        weights = self.baseline_weights * self.availability_boost * self.user_weights
        
        # Apply genre adjustment
        genre_condition = torch.zeros_like(weights, dtype=torch.bool)
        genre_condition[19:32] = True
        weights = torch.where(
            genre_condition & ~vector_has_genre.unsqueeze(1),
            weights * adj_factor.unsqueeze(1),
            weights
        )
        
        # Apply weights and mask invalid values
        weighted_query = query_t * weights * valid_mask.float()
        weighted_vectors = vectors * weights * valid_mask.float()
        
        # Compute cosine similarity
        dot = torch.sum(weighted_query * weighted_vectors, dim=1)
        norm_query = torch.norm(weighted_query, dim=1)
        norm_vectors = torch.norm(weighted_vectors, dim=1)
        
        # Avoid division by zero
        sim = dot / (norm_query * norm_vectors + 1e-9)
        sim[torch.isnan(sim)] = 0
        
        # Synchronize GPU before returning
        if self.device_str == 'cuda':
            torch.cuda.synchronize()
        
        return sim.cpu().numpy()  # Convert to CPU numpy array

    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, vectors: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
        """
        Optimized GPU implementation of hybrid cosine-euclidean similarity.
        Computes: hybrid_sim = cosine_sim * euclidean_sim
        """
        # Convert masks to int64 if needed
        if masks.dtype != torch.int64:
            masks = masks.to(torch.int64)
        
        # Convert query to tensor with matching device
        query_t = torch.tensor(query, dtype=vectors.dtype, device=vectors.device)
        
        # Ensure query has correct shape
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0).expand(vectors.shape[0], -1)
        
        # Create valid mask for query
        query_valid = (query_t != -1)
        
        # Unpack masks to boolean tensors
        vector_valid = self._unpack_masks(masks)
        
        # Combine validity masks
        valid_mask = query_valid & vector_valid
        
        # Compute genre flags
        query_has_genre = torch.any(query_t[:, 19:32] != -1, dim=1)
        vector_has_genre = torch.any(vectors[:, 19:32] != -1, dim=1)
        
        # Compute adjustment factor
        adj_factor = torch.where(
            query_has_genre & vector_has_genre,
            torch.tensor(1.0, device=vectors.device),
            self.genre_reduction
        )
        
        # Create weight matrix
        weights = self.baseline_weights * self.availability_boost * self.user_weights
        
        # Apply genre adjustment
        genre_condition = torch.zeros_like(weights, dtype=torch.bool)
        genre_condition[19:32] = True
        weights = torch.where(
            genre_condition & ~vector_has_genre.unsqueeze(1),
            weights * adj_factor.unsqueeze(1),
            weights
        )
        
        # Apply weights and mask invalid values
        weighted_query = query_t * weights * valid_mask.float()
        weighted_vectors = vectors * weights * valid_mask.float()
        
        # Compute cosine similarity
        dot = torch.sum(weighted_query * weighted_vectors, dim=1)
        norm_query = torch.norm(weighted_query, dim=1)
        norm_vectors = torch.norm(weighted_vectors, dim=1)
        cosine_sim = dot / (norm_query * norm_vectors + 1e-9)
        cosine_sim = torch.nan_to_num(cosine_sim, nan=0.0)
        
        # Compute Euclidean distance
        diff = weighted_query - weighted_vectors
        sq_diff = diff ** 2
        sum_sq = torch.sum(sq_diff, dim=1)
        euclidean_dist = torch.sqrt(sum_sq)
        
        # Compute Euclidean similarity
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine into hybrid similarity
        hybrid_sim = cosine_sim * euclidean_sim
        
        # Synchronize GPU before returning
        if self.device_str == 'cuda':
            torch.cuda.synchronize()
        
        return hybrid_sim.cpu().numpy()

    def masked_euclidean_similarity(self, query: np.ndarray, vectors: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
        """GPU-accelerated Euclidean similarity with masking"""
        # Convert query to PyTorch tensor on the same device as vectors
        query_t = torch.tensor(query, dtype=self.DTYPE, device=vectors.device)
        
        # Ensure query has correct shape
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0).expand(vectors.shape[0], -1)
        
        # Create valid mask for query
        query_valid = (query_t != -1)
        
        # Unpack masks to boolean tensors
        vector_valid = self._unpack_masks(masks)
        
        # Combine validity masks
        valid_mask = query_valid & vector_valid
        
        # Apply weights
        weighted_query = query_t * self.user_weights * valid_mask.float()
        weighted_vectors = vectors * self.user_weights * valid_mask.float()
        
        # Compute differences only where valid
        diff = torch.where(valid_mask, weighted_query - weighted_vectors, torch.zeros_like(query_t))
        
        # Compute Euclidean distance
        squared = diff ** 2
        sum_squared = torch.sum(squared, dim=1)
        distances = torch.sqrt(sum_squared)
        
        # Convert to similarity
        similarities = 1.0 / (1.0 + distances)
        
        # Synchronize GPU before returning
        if self.device_str == 'cuda':
            torch.cuda.synchronize()
        
        return similarities.cpu().numpy()

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
