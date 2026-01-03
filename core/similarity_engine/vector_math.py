# core/similarity_engine/vector_math.py
"""
Vector math operations for similarity calculations.
"""
import numpy as np
from numba import njit, prange
from typing import List
import torch
from .weight_layers import WeightLayers

class VectorOps:
    """Mathematical operations on vectors."""
    
    VECTOR_DIMENSIONS = 32
    DTYPE = np.float32
    GENRE_START = 19 # index
    GENRE_END = 31

    def __init__(self):
        self.weight_layers = WeightLayers()
        # Precompute constants
        self.baseline_weights = self.weight_layers.baseline_weights.astype(np.float32)
        self.genre_mask = np.zeros(32, dtype=np.bool_)
        self.genre_mask[self.GENRE_START:self.GENRE_END+1] = True
        self.availability_boost = np.float32(self.weight_layers.availability_boost)
        self.genre_reduction = np.float32(self.weight_layers.genre_reduction)
        # Warm up JIT compiler
        self._warmup_jit()

    def _warmup_jit(self):
        """Warm up the JIT compiler"""
        warmup_vector = np.random.rand(32).astype(np.float32)
        warmup_vectors = np.random.rand(100, 32).astype(np.float32)
        self.masked_cosine_similarity_batch(warmup_vector, warmup_vectors)
      
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

    def masked_cosine_similarity_batch(self, query: np.ndarray, vectors) -> np.ndarray:
        """
        Compute masked cosine similarity between query and batch of vectors
        with feature weighting
        
        Handles both NumPy arrays and PyTorch tensors efficiently
        """      
        # Convert PyTorch tensors to NumPy arrays if needed
        if isinstance(vectors, torch.Tensor):
            # Only convert if necessary - GPU tensors need conversion
            if vectors.is_cuda:
                vectors = vectors.cpu().numpy()
            else:
                vectors = vectors.numpy()
            
        # Handle single query vector by broadcasting to batch size
        if query.ndim == 1:
            query = np.tile(query, (vectors.shape[0], 1))
        
        # Call optimized Numba function
        return self._numba_optimized_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            self.baseline_weights,
            self.genre_mask,
            self.availability_boost,
            self.genre_reduction
        )

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_optimized_similarity(query, vectors, baseline_weights, genre_mask, availability_boost, genre_reduction):
        """Optimized similarity computation with cache-blocking"""
        n = vectors.shape[0]
        dim = vectors.shape[1]
        similarities = np.empty(n, dtype=np.float32)
        
        # Precompute query genre status once
        query_has_genre = False
        for j in range(dim):
            if genre_mask[j] and query[0, j] != -1:
                query_has_genre = True
                break
        
        # Vectorized constants
        adj_factor = genre_reduction
        
        # Determine optimal block size (512KB blocks)
        block_size = 4096  # 4096 vectors × 128B = 512KB (L2 cache size)
        num_blocks = (n + block_size - 1) // block_size
        
        # Process in parallel blocks
        for block_idx in prange(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, n)
            
            # Process vectors in current block
            for i in range(start_idx, end_idx):
                dot = 0.0
                norm_u = 0.0
                norm_v = 0.0
                vector_has_genre = False
                
                # First pass: Check genre presence
                for j in range(dim):
                    if genre_mask[j] and vectors[i, j] != -1:
                        vector_has_genre = True
                        break
                
                # Unified adjustment factor (same as GPU)
                if query_has_genre and vector_has_genre:
                    adj_factor = 1.0
                
                # Second pass: Compute similarity
                for j in range(dim):
                    q_val = query[i, j]
                    v_val = vectors[i, j]
                    
                    # Skip invalid dimensions
                    if q_val == -1 or v_val == -1:
                        continue
                    
                    # Calculate weight
                    weight = baseline_weights[j] * availability_boost
                    
                    # Genre-specific adjustment
                    if genre_mask[j] and not vector_has_genre:
                        weight *= adj_factor
                    
                    # Weighted values
                    w_q = q_val * weight
                    w_v = v_val * weight
                    
                    # Accumulate
                    dot += w_q * w_v
                    norm_u += w_q * w_q
                    norm_v += w_v * w_v
                
                # Compute final similarity
                norm_product = max(np.sqrt(norm_u) * np.sqrt(norm_v), 1e-9)
                similarities[i] = dot / norm_product
        
        return similarities

    @staticmethod
    def euclidean_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Euclidean distance converted to similarity (1 / (1 + distance))"""
        diff = vectors - query
        squared = np.square(diff)
        distances = np.sqrt(np.sum(squared, axis=1))
        return 1 / (1 + distances)
    
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
