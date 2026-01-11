# core/similarity_engine/vector_math_gpu.py
import torch
import numpy as np
from .weight_layers import WeightLayers
from typing import List

class VectorOpsGPU:
    """GPU-accelerated vector operations using PyTorch."""
    
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

    def fused_similarity(
        self,
        query: np.ndarray,
        vectors: torch.Tensor,
        masks: torch.Tensor,
        regions: torch.Tensor,
        query_region: int,
        region_strength: float,
        algorithm: str = 'cosine-euclidean'
    ) -> torch.Tensor:
        """
        Combined feature + region similarity with multiplicative penalty
        """
        # Convert query to tensor
        query_t = torch.tensor(query, dtype=vectors.dtype, device=vectors.device)
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0).expand(vectors.shape[0], -1)
        
        # Compute base similarity
        if algorithm == 'cosine':
            feature_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        elif algorithm == 'euclidean':
            feature_sim = self.masked_euclidean_similarity(query, vectors, masks)
        else:  # Default to hybrid
            feature_sim = self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks)
        
        # Convert to tensor if needed
        if not isinstance(feature_sim, torch.Tensor):
            feature_sim = torch.tensor(feature_sim, dtype=torch.float32, device=vectors.device)
        
        # Calculate region match (1 if same region, 0 otherwise)
        region_match = (regions == query_region).float()
        
        # For matching regions: region_compensation_factor = 1
        # For non-matching regions: region_compensation_factor = (1 - region_strength)
        region_compensation_factor = torch.where(
            region_match == 1,
            torch.tensor(1.0, device=feature_sim.device),
            torch.tensor(1.0 - region_strength, device=feature_sim.device)
        )
        
        # Apply multiplicative penalty
        blended_sim = feature_sim * region_compensation_factor
        
        # Synchronize GPU before returning
        if self.device_str == 'cuda':
            torch.cuda.synchronize()
            
        return blended_sim

    def fused_similarity_batch(
        self,
        query: np.ndarray,
        vectors: torch.Tensor,
        masks: torch.Tensor,
        regions: torch.Tensor,
        query_region: int,
        region_strength: float,
        algorithm: str
    ) -> torch.Tensor:
        """
        Combined feature + region similarity with multiplicative penalty
        """
        # Compute base similarity using existing function
        if algorithm == 'cosine':
            feature_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        elif algorithm == 'euclidean':
            feature_sim = self.masked_euclidean_similarity(query, vectors, masks)
        else:
            feature_sim = self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks)
        
        # Convert to tensor if needed
        if not isinstance(feature_sim, torch.Tensor):
            feature_sim = torch.tensor(feature_sim, dtype=torch.float32, device=vectors.device)
        
        # Calculate region match (1 if same region, 0 otherwise)
        region_match = (regions == query_region).float()
        
        # For matching regions: region_compensation_factor = 1
        # For non-matching regions: region_compensation_factor = (1 - region_strength)
        region_compensation_factor = torch.where(
            region_match == 1,
            torch.tensor(1.0, device=feature_sim.device),
            torch.tensor(1.0 - region_strength, device=feature_sim.device)
        )
        
        # Apply multiplicative penalty
        blended_sim = feature_sim * region_compensation_factor
        
        # Debug output
        # print(f"  Region strength: {region_strength}")
        # print(f"  Matching regions: {torch.sum(region_match == 1).item()}")
        # print(f"  Non-matching regions: {torch.sum(region_match == 0).item()}")
        # print(f"  Feature sim range: {feature_sim.min().item():.4f} - {feature_sim.max().item():.4f}")
        # print(f"  Blended sim range: {blended_sim.min().item():.4f} - {blended_sim.max().item():.4f}")
        
        return blended_sim

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

    def _extract_region_bits(self, packed_regions: torch.Tensor, index: int) -> int:
        """
        Extract region bits for a single index
        """
        byte_offset = (index * 3) // 8
        bit_offset = (index * 3) % 8
        
        byte_val = packed_regions[byte_offset].item()
        
        if bit_offset <= 5:
            return (byte_val >> (5 - bit_offset)) & 0x07
        else:
            next_byte_val = packed_regions[byte_offset + 1].item()
            return ((byte_val << (bit_offset - 5)) | 
                   (next_byte_val >> (13 - bit_offset))) & 0x07

    def _batch_extract_regions(self, packed_regions: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Batch extract regions from packed data
        """
        byte_offsets = (indices * 3) // 8
        bit_offsets = (indices * 3) % 8
        
        # Get packed bytes
        packed_bytes = packed_regions[byte_offsets]
        
        # Create masks for extraction cases
        mask_low = bit_offsets <= 5
        mask_high = ~mask_low
        
        regions = torch.zeros_like(indices, dtype=torch.uint8)
        
        # Case 1: Bits within single byte
        if torch.any(mask_low):
            shift = 5 - bit_offsets[mask_low]
            regions[mask_low] = (packed_bytes[mask_low] >> shift) & 0x07
        
        # Case 2: Bits span two bytes
        if torch.any(mask_high):
            next_bytes = packed_regions[byte_offsets[mask_high] + 1]
            shift_amount = bit_offsets[mask_high] - 5
            first_part = packed_bytes[mask_high] << shift_amount
            second_part = next_bytes >> (8 - shift_amount)
            regions[mask_high] = (first_part | second_part) & 0x07
        
        return regions

    def region_aware_similarity(
        self,
        query: np.ndarray,
        vectors: torch.Tensor,
        masks: torch.Tensor,
        packed_regions: torch.Tensor,
        indices: torch.Tensor,
        query_region: int,
        region_strength: float,
        algorithm: str = 'cosine-euclidean'
    ) -> torch.Tensor:
        """
        Fully integrated region-aware similarity with bit extraction
        """
        # Extract regions for this batch
        regions = self._batch_extract_regions(packed_regions, indices)
        
        # Compute base similarity
        if algorithm == 'cosine':
            feature_sim = self.masked_weighted_cosine_similarity(query, vectors, masks)
        elif algorithm == 'euclidean':
            feature_sim = self.masked_euclidean_similarity(query, vectors, masks)
        else:  # Default to hybrid
            feature_sim = self.masked_weighted_cosine_euclidean_similarity(query, vectors, masks)
        
        # Convert to tensor if needed
        if not isinstance(feature_sim, torch.Tensor):
            feature_sim = torch.tensor(feature_sim, dtype=torch.float32, device=vectors.device)
        
        # Blend with region similarity
        region_match = (regions == query_region).float()
        blended_sim = (1 - region_strength) * feature_sim + region_strength * region_match
        
        return blended_sim
