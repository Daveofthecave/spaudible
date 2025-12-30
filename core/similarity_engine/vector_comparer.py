# core/similarity_engine/vector_comparer.py
"""
Chunked similarity search algorithms.
"""
import random
import time
from typing import List, Tuple, Optional, Callable
import numpy as np


class ChunkedSearch:
    """Search algorithms for finding similar vectors."""
    
    VECTOR_DIMENSIONS = 32
    
    def __init__(self, chunk_size: int = 100_000):
        """
        Initialize chunked search.
        
        Args:
            chunk_size: Number of vectors to process in one chunk
        """
        self.chunk_size = chunk_size
        # print(f"     Using Chunked Search (chunk_size = {chunk_size:,})")
    
    def sequential_scan(self,
                        query_vector: np.ndarray,
                        vector_source: Callable[[int, int], np.ndarray],
                        total_vectors: int,
                        vector_ops,
                        top_k: int = 10,
                        max_vectors: Optional[int] = None,
                        progress_callback = None,
                        **kwargs) -> Tuple[List[int], List[float]]:
        """
        Perform sequential scan search.
        
        Args:
            query_vector: Query vector (32D numpy array)
            vector_source: Function that returns vectors given (start_idx, num_vectors)
            total_vectors: Total number of vectors available
            vector_ops: Vector operations instance
            top_k: Number of top results to return
            max_vectors: Maximum vectors to scan (None = all)
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (indices, similarities)
        """
        if query_vector.shape != (self.VECTOR_DIMENSIONS,):
            raise ValueError(f"Query vector must be {self.VECTOR_DIMENSIONS}D")
        
        # Initialize results
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        vectors_to_scan = total_vectors if max_vectors is None else min(max_vectors, total_vectors)
        num_chunks = (vectors_to_scan + self.chunk_size - 1) // self.chunk_size
        
        print(f"🔍 Sequential scan of {vectors_to_scan:,} vectors in {num_chunks} chunks...")
        start_time = time.time()
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, vectors_to_scan)
            actual_chunk_size = chunk_end - chunk_start
            
            # Read vectors for this chunk
            vectors = vector_source(chunk_start, actual_chunk_size)
            
            # Compute similarities
            similarities = vector_ops.cosine_similarity_batch(query_vector, vectors)
            
            # Update top-k for this chunk
            if actual_chunk_size > 0:
                # Get indices of top similarities in this chunk
                chunk_top_k = min(top_k, actual_chunk_size)
                chunk_top_indices = np.argpartition(-similarities, chunk_top_k)[:chunk_top_k]
                
                # Combine with current top-k
                combined_similarities = np.concatenate([top_similarities, similarities[chunk_top_indices]])
                combined_indices = np.concatenate([top_indices, chunk_top_indices + chunk_start])
                
                # Get new top-k
                new_top_k = min(top_k, len(combined_similarities))
                top_indices_in_combined = np.argpartition(-combined_similarities, new_top_k)[:new_top_k]
                
                top_similarities = combined_similarities[top_indices_in_combined]
                top_indices = combined_indices[top_indices_in_combined]
            
            # Progress update
            if progress_callback:
                progress = ((chunk_idx + 1) / num_chunks) * 100
                progress_callback(progress)
            
            if chunk_idx % 10 == 0 or chunk_idx == num_chunks - 1:
                elapsed = time.time() - start_time
                rate = chunk_end / elapsed if elapsed > 0 else 0
                print(f"   Processed {chunk_end:,} vectors - {rate:,.0f} vec/sec")
        
        # Sort results
        sorted_indices = np.argsort(-top_similarities)
        top_similarities = top_similarities[sorted_indices]
        top_indices = top_indices[sorted_indices]
        
        search_time = time.time() - start_time
        avg_speed = vectors_to_scan / search_time
        print(f"\n✅ Sequential scan complete in {search_time:.3f} seconds")
        print(f"   Average speed: {avg_speed:,.0f} vectors/second")
        
        return top_indices.tolist(), top_similarities.tolist()
    
    def random_chunk_search(self,
                           query_vector: np.ndarray,
                           vector_source: Callable[[int, int], np.ndarray],
                           total_vectors: int,
                           vector_ops,
                           num_chunks: int = 100,
                           top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Search by sampling random chunks.
        
        Args:
            query_vector: Query vector (32D numpy array)
            vector_source: Function that returns vectors given (start_idx, num_vectors)
            total_vectors: Total number of vectors available
            vector_ops: Vector operations instance
            num_chunks: Number of random chunks to sample
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, similarities)
        """
        if query_vector.shape != (self.VECTOR_DIMENSIONS,):
            raise ValueError(f"Query vector must be {self.VECTOR_DIMENSIONS}D")
        
        # Initialize results
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        print(f"   Random chunk search ({num_chunks} chunks, {num_chunks * self.chunk_size:,} total vectors)...")
        start_time = time.time()
        
        for chunk_idx in range(num_chunks):
            # Pick a random chunk start
            max_start = total_vectors - self.chunk_size
            chunk_start = random.randint(0, max_start) if max_start > 0 else 0
            chunk_end = min(chunk_start + self.chunk_size, total_vectors)
            actual_chunk_size = chunk_end - chunk_start
            
            # Read vectors for this chunk
            vectors = vector_source(chunk_start, actual_chunk_size)
            
            # Compute similarities
            similarities = vector_ops.cosine_similarity_batch(query_vector, vectors)
            
            # Update top-k for this chunk
            if actual_chunk_size > 0:
                # Get indices of top similarities in this chunk
                chunk_top_k = min(top_k, actual_chunk_size)
                chunk_top_indices = np.argpartition(-similarities, chunk_top_k)[:chunk_top_k]
                
                # Combine with current top-k
                combined_similarities = np.concatenate([top_similarities, similarities[chunk_top_indices]])
                combined_indices = np.concatenate([top_indices, chunk_top_indices + chunk_start])
                
                # Get new top-k
                new_top_k = min(top_k, len(combined_similarities))
                top_indices_in_combined = np.argpartition(-combined_similarities, new_top_k)[:new_top_k]
                
                top_similarities = combined_similarities[top_indices_in_combined]
                top_indices = combined_indices[top_indices_in_combined]
            
            # Progress update
            if chunk_idx % 10 == 0 and chunk_idx > 0:
                elapsed = time.time() - start_time
                vectors_processed = (chunk_idx + 1) * self.chunk_size
                rate = vectors_processed / elapsed if elapsed > 0 else 0
                print(f"   Sampled {chunk_idx + 1}/{num_chunks} chunks - {rate:,.0f} vec/sec")
        
        # Sort results
        sorted_indices = np.argsort(-top_similarities)
        top_similarities = top_similarities[sorted_indices]
        top_indices = top_indices[sorted_indices]
        
        search_time = time.time() - start_time
        total_vectors_processed = num_chunks * self.chunk_size
        avg_speed = total_vectors_processed / search_time
        print(f"   Random chunk search completed in {search_time:.3f} seconds")
        print(f"   Average speed: {avg_speed:,.0f} vectors/second")
        
        return top_indices.tolist(), top_similarities.tolist()
    
    def progressive_search(self,
                          query_vector: np.ndarray,
                          vector_source: Callable[[int, int], np.ndarray],
                          total_vectors: int,
                          vector_ops,
                          min_chunks: int = 1,
                          max_chunks: int = 100,
                          quality_threshold: float = 0.95,
                          top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Progressive search until quality threshold reached.
        
        Args:
            query_vector: Query vector (32D numpy array)
            vector_source: Function that returns vectors given (start_idx, num_vectors)
            total_vectors: Total number of vectors available
            vector_ops: Vector operations instance
            min_chunks: Minimum chunks to sample
            max_chunks: Maximum chunks to sample
            quality_threshold: Stop when top similarity > threshold
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, similarities)
        """
        print(f"   Progressive search (target quality: {quality_threshold})")
        
        best_indices = []
        best_similarities = []
        current_chunks = min_chunks
        
        while current_chunks <= max_chunks:
            print(f"   Sampling {current_chunks} chunks...")
            
            indices, similarities = self.random_chunk_search(
                query_vector,
                vector_source,
                total_vectors,
                vector_ops,
                num_chunks=current_chunks,
                top_k=top_k
            )
            
            # Check if we have good enough results
            if similarities and similarities[0] >= quality_threshold:
                print(f"✅ Quality threshold reached: {similarities[0]:.4f} >= {quality_threshold}")
                return indices, similarities
            
            # Double chunk count for next iteration
            current_chunks = min(current_chunks * 2, max_chunks)
            best_indices, best_similarities = indices, similarities
        
        print(f"⚠️  Max chunks reached, returning best found")
        return best_indices, best_similarities
