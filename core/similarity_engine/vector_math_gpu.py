# core/similarity_engine/vector_math_gpu.py
import torch
import numpy as np
from .weight_layers import WeightLayers

class VectorOpsGPU:
    """GPU-accelerated vector operations using PyTorch with unified logic."""
    
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
        self.genre_mask = torch.zeros(32, dtype=torch.bool, device=self.device)
        self.genre_mask[19:32] = True  # Dimensions 20-32 (0-indexed 19-31)
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
    
    def masked_cosine_similarity_batch(self, query: np.ndarray, vectors: torch.Tensor) -> np.ndarray:
        # Convert query to PyTorch tensor on the same device as vectors
        query_t = torch.tensor(query, dtype=self.DTYPE, device=vectors.device)
        
        # Ensure query has correct shape
        if query_t.ndim == 1:
            query_t = query_t.unsqueeze(0).expand(vectors.shape[0], -1)
        
        # Create valid mask for both query and vectors
        valid_mask = (query_t != -1) & (vectors != -1)
        
        # Compute genre flags
        query_has_genre = torch.any(query_t[:, self.genre_mask] != -1, dim=1)
        vector_has_genre = torch.any(vectors[:, self.genre_mask] != -1, dim=1)
        
        # Compute adjustment factor (same as CPU)
        adj_factor = torch.where(
            query_has_genre & vector_has_genre,
            torch.tensor(1.0, device=vectors.device),
            self.genre_reduction
        )
        
        # Create weight matrix
        weights = self.baseline_weights * self.availability_boost
        weights = weights.to(vectors.device)  # Ensure weights are on correct device
        
        weights = torch.where(
            self.genre_mask.to(vectors.device) & ~vector_has_genre.unsqueeze(1),
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

