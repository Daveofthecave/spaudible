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

    def __init__(self, algorithm='cosine-euclidean'):
        self.algorithm = algorithm
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
        self.masked_weighted_cosine_similarity(warmup_vector, warmup_vectors)
        self.masked_weighted_cosine_euclidean_similarity(warmup_vector, warmup_vectors)
      
    def compute_similarity(self, query: np.ndarray, vectors) -> np.ndarray:
        """Main entry point for similarity calculations"""
        if self.algorithm == 'cosine':
            return self.masked_weighted_cosine_similarity(query, vectors)
        elif self.algorithm == 'cosine-euclidean':
            return self.masked_weighted_cosine_euclidean_similarity(query, vectors)
        elif self.algorithm == 'euclidean':
            return self.masked_euclidean_similarity(query, vectors)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    @staticmethod
    def simple_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
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

    def masked_weighted_cosine_similarity(self, query: np.ndarray, vectors) -> np.ndarray:
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
        return self._numba_masked_weighted_cosine_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            self.baseline_weights,
            self.genre_mask,
            self.availability_boost,
            self.genre_reduction
        )

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_masked_weighted_cosine_similarity(query, vectors, baseline_weights, genre_mask, availability_boost, genre_reduction):
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

    def masked_weighted_cosine_euclidean_similarity(self, query: np.ndarray, vectors) -> np.ndarray:
        """
        Compute hybrid cosine-euclidean similarity with feature weighting
        
        Combines directional alignment (cosine) and magnitude proximity (euclidean)
        into a single similarity metric: hybrid_sim = cosine_sim * euclidean_sim
        
        Args:
            query: Query vector (32D numpy array)
            vectors: Batch of vectors to compare against
            
        Returns:
            Array of hybrid similarities shape (N,)
        """
        # Convert PyTorch tensors to NumPy arrays if needed
        if isinstance(vectors, torch.Tensor):
            if vectors.is_cuda:
                vectors = vectors.cpu().numpy()
            else:
                vectors = vectors.numpy()
            
        # Handle single query vector by broadcasting to batch size
        if query.ndim == 1:
            query = np.tile(query, (vectors.shape[0], 1))
        
        # Call optimized Numba function
        return self._numba_masked_weighted_cosine_euclidean_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32),
            self.baseline_weights,
            self.genre_mask,
            self.availability_boost,
            self.genre_reduction
        )

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_masked_weighted_cosine_euclidean_similarity(query, vectors, baseline_weights, genre_mask, availability_boost, genre_reduction):
        """
        Optimized hybrid cosine-euclidean similarity computation
        Computes: hybrid_sim = cosine_sim * euclidean_sim
        """
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
                sq_diff = 0.0
                vector_has_genre = False
                
                # First pass: Check genre presence
                for j in range(dim):
                    if genre_mask[j] and vectors[i, j] != -1:
                        vector_has_genre = True
                        break
                
                # Unified adjustment factor (same as GPU)
                if query_has_genre and vector_has_genre:
                    adj_factor = 1.0
                
                # Second pass: Compute both metrics simultaneously
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
                    
                    # Cosine components
                    dot += w_q * w_v
                    norm_u += w_q * w_q
                    norm_v += w_v * w_v
                    
                    # Euclidean components
                    sq_diff += (w_q - w_v) ** 2
                
                # Compute cosine similarity
                norm_product = max(np.sqrt(norm_u) * np.sqrt(norm_v), 1e-9)
                cosine_sim = dot / norm_product
                
                # Compute Euclidean similarity
                euclidean_dist = np.sqrt(sq_diff)
                euclidean_sim = 1.0 / (1.0 + euclidean_dist)
                
                # Combine into hybrid similarity
                similarities[i] = cosine_sim * euclidean_sim
        
        return similarities

    def masked_euclidean_similarity(self, query: np.ndarray, vectors) -> np.ndarray:
        """Euclidean similarity with masking"""
        # Convert PyTorch tensors to NumPy arrays if needed
        if isinstance(vectors, torch.Tensor):
            if vectors.is_cuda:
                vectors = vectors.cpu().numpy()
            else:
                vectors = vectors.numpy()
            
        # Handle single query vector by broadcasting to batch size
        if query.ndim == 1:
            query = np.tile(query, (vectors.shape[0], 1))
        
        # Call optimized Numba function
        return self._numba_masked_euclidean_similarity(
            query.astype(np.float32),
            vectors.astype(np.float32)
        )

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_masked_euclidean_similarity(query, vectors):
        """Optimized Euclidean similarity computation"""
        n = vectors.shape[0]
        dim = vectors.shape[1]
        similarities = np.empty(n, dtype=np.float32)
        
        # Determine optimal block size
        block_size = 4096
        num_blocks = (n + block_size - 1) // block_size
        
        # Process in parallel blocks
        for block_idx in prange(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, n)
            
            # Process vectors in current block
            for i in range(start_idx, end_idx):
                sq_diff = 0.0
                valid_count = 0
                
                for j in range(dim):
                    q_val = query[i, j]
                    v_val = vectors[i, j]
                    
                    # Skip invalid dimensions
                    if q_val == -1 or v_val == -1:
                        continue
                    
                    sq_diff += (q_val - v_val) ** 2
                    valid_count += 1
                
                # Compute Euclidean distance
                distance = np.sqrt(sq_diff) if valid_count > 0 else 0
                
                # Convert to similarity
                similarities[i] = 1.0 / (1.0 + distance)
        
        return similarities
    
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
