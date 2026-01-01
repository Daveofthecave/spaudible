# core/similarity_engine/vector_comparer.py
"""
Chunked similarity search algorithms.
"""
import numpy as np
import random
import sys
import time
import torch
from typing import List, Tuple, Optional, Callable
from .vector_math import VectorOps
from .vector_math_gpu import VectorOpsGPU

class ChunkedSearch:
    """Search algorithms for finding similar vectors."""
    
    VECTOR_DIMENSIONS = 32
    
    def __init__(self, chunk_size: int = 100_000_000, use_gpu: bool = True):
        """
        Initialize chunked search.
        
        Args:
            chunk_size: Number of vectors to process in one chunk
        """
        self.chunk_size = chunk_size
        self.progress_bar_width = 50
        self.use_gpu = use_gpu
        self.gpu_ops = None
        self.performance_stats = {}

        # Initialize GPU operations if available
        if self.use_gpu and torch.cuda.is_available():
            try:
                # Check available VRAM
                free_vram = torch.cuda.mem_get_info()[0]
                if free_vram < 500_000_000:  # Less than 500MB
                    print("  ⚠️  Low VRAM available, disabling GPU acceleration")
                    self.use_gpu = False
                else:
                    self.gpu_ops = VectorOpsGPU()
                    print("  ✅ GPU acceleration enabled")
            except Exception as e:
                print(f"  ⚠️ GPU initialization failed: {e}")
                self.gpu_ops = None              
    
    def sequential_scan(self,
                        query_vector: np.ndarray,
                        vector_source: Callable[[int, int], np.ndarray],
                        total_vectors: int,
                        vector_ops: VectorOps,
                        top_k: int = 10,
                        max_vectors: Optional[int] = None,
                        progress_callback = None,
                        **kwargs) -> Tuple[List[int], List[float]]:
        """
        Perform a sequential scan of the vector cache.
        
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
        # If GPU is available and initialized, use GPU path
        if self.use_gpu and self.gpu_ops:
            return self._gpu_sequential_scan(
                query_vector,
                vector_source,
                total_vectors,
                vector_ops,
                top_k,
                max_vectors,
                progress_callback,
                **kwargs
            )
        else:
            return self._cpu_sequential_scan(
                query_vector,
                vector_source,
                total_vectors,
                vector_ops,
                top_k,
                max_vectors,
                progress_callback,
                **kwargs
            )

    def _gpu_sequential_scan(self,
                             query_vector: np.ndarray,
                             vector_source: Callable[[int, int], np.ndarray],
                             total_vectors: int,
                             vector_ops: VectorOps,
                             top_k: int = 10,
                             max_vectors: Optional[int] = None,
                             progress_callback = None,
                             **kwargs) -> Tuple[List[int], List[float]]:
        """
        GPU-optimized sequential scan implementation.
        """
        if query_vector.shape != (self.VECTOR_DIMENSIONS,):
            raise ValueError(f"Query vector must be {self.VECTOR_DIMENSIONS}D")
        
        # Initialize results
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        vectors_to_scan = total_vectors if max_vectors is None else min(max_vectors, total_vectors)
        num_chunks = (vectors_to_scan + self.chunk_size - 1) // self.chunk_size

        # Initialize progress bar
        start_time = self._init_progress_bar(
            vectors_to_scan,
            f"🔍 Sequentially scanning {vectors_to_scan:,} vectors in {num_chunks} chunks (GPU)...\n"
        )
        last_update = start_time

        # Performance monitoring
        total_transfer_time = 0.0
        total_compute_time = 0.0
        total_vectors_processed = 0
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, vectors_to_scan)
            actual_chunk_size = chunk_end - chunk_start

            # Time data transfer
            transfer_start = time.time()
            vectors = vector_source(chunk_start, actual_chunk_size)
            transfer_time = time.time() - transfer_start
            total_transfer_time += transfer_time

            # Compute similarities on GPU
            compute_start = time.time()
            similarities = self.gpu_ops.masked_cosine_similarity_batch(query_vector, vectors)
            compute_time = time.time() - compute_start
            total_compute_time += compute_time
            total_vectors_processed += actual_chunk_size
            
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
                
            # Update progress bar
            last_update = self._update_progress_bar(
                chunk_end, vectors_to_scan, start_time, last_update
            )
        
        # Finalize progress bar
        self._complete_progress_bar(vectors_to_scan, vectors_to_scan, start_time)
        print(f"\n✅ Sequential scan complete (GPU)")

        # Calculate performance metrics
        transfer_bytes = total_vectors_processed * 128  # 32 dimensions * 4 bytes
        transfer_bw = transfer_bytes / total_transfer_time / 1e9 if total_transfer_time > 0 else 0
        compute_throughput = total_vectors_processed / total_compute_time / 1e6 if total_compute_time > 0 else 0

        # Store performance stats
        self.performance_stats = {
            "transfer_time": total_transfer_time,
            "compute_time": total_compute_time,
            "transfer_bw": transfer_bw,
            "compute_throughput": compute_throughput,
            "total_vectors": total_vectors_processed
        }        

        return top_indices.tolist(), top_similarities.tolist()

    def _cpu_sequential_scan(self,
                             query_vector: np.ndarray,
                             vector_source: Callable[[int, int], np.ndarray],
                             total_vectors: int,
                             vector_ops: VectorOps,
                             top_k: int = 10,
                             max_vectors: Optional[int] = None,
                             progress_callback = None,
                             **kwargs) -> Tuple[List[int], List[float]]:
        """
        Pure CPU implementation of sequential scan (original version)
        """
        if query_vector.shape != (self.VECTOR_DIMENSIONS,):
            raise ValueError(f"Query vector must be {self.VECTOR_DIMENSIONS}D")
        
        # Initialize results
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        vectors_to_scan = total_vectors if max_vectors is None else min(max_vectors, total_vectors)
        num_chunks = (vectors_to_scan + self.chunk_size - 1) // self.chunk_size

        # Initialize progress bar
        start_time = self._init_progress_bar(
            vectors_to_scan,
            f"🔍 Sequentially scanning {vectors_to_scan:,} vectors in {num_chunks} chunks...\n"
        )
        last_update = start_time     
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, vectors_to_scan)
            actual_chunk_size = chunk_end - chunk_start
            
            # Read vectors for this chunk
            vectors = vector_source(chunk_start, actual_chunk_size)
            
            # Compute similarities
            similarities = vector_ops.masked_cosine_similarity_batch(query_vector, vectors)
            
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
                
            # Update progress bar
            last_update = self._update_progress_bar(
                chunk_end, vectors_to_scan, start_time, last_update
            )
        
        # Finalize progress bar
        self._complete_progress_bar(vectors_to_scan, vectors_to_scan, start_time)
        print(f"\n✅ Sequential scan complete")

        return top_indices.tolist(), top_similarities.tolist()
    
    def random_chunk_search(self,
                           query_vector: np.ndarray,
                           vector_source: Callable[[int, int], np.ndarray],
                           total_vectors: int,
                           vector_ops: VectorOps,
                           num_chunks: int = 100,
                           top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Scan the vector cache by sampling random chunks.
        
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
        # If no GPU is available, execute the CPU-based version of this search
        if not self.use_gpu:
            return self._cpu_random_chunk_search(
                query_vector,
                vector_source,
                total_vectors,
                vector_ops,
                num_chunks,
                top_k
            )

        total_to_process = num_chunks * self.chunk_size

        if query_vector.shape != (self.VECTOR_DIMENSIONS,):
            raise ValueError(f"Query vector must be {self.VECTOR_DIMENSIONS}D")
        
        # Initialize results
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        # Initialize progress bar
        start_time = self._init_progress_bar(
            total_to_process,
            f"Random chunk search ({num_chunks} chunks, {total_to_process:,} total vectors"
        )
        last_update = start_time
        
        # Performance monitoring
        total_transfer_time = 0.0
        total_compute_time = 0.0
        total_vectors_processed = 0
        
        for chunk_idx in range(num_chunks):
            # Pick a random chunk start
            max_start = total_vectors - self.chunk_size
            chunk_start = random.randint(0, max_start) if max_start > 0 else 0
            chunk_end = min(chunk_start + self.chunk_size, total_vectors)
            actual_chunk_size = chunk_end - chunk_start
            
            # Time data transfer
            transfer_start = time.time()
            vectors = vector_source(chunk_start, actual_chunk_size)
            transfer_time = time.time() - transfer_start
            total_transfer_time += transfer_time
            
            # Check if vectors are GPU tensors
            is_gpu_tensor = isinstance(vectors, torch.Tensor)

            # Time computation
            compute_start = time.time()
            # GPU acceleration for large batches
            if self.gpu_ops and actual_chunk_size > 50000 and is_gpu_tensor:
                similarities = self.gpu_ops.masked_cosine_similarity_batch(query_vector, vectors)
            else:
                # Convert GPU tensor to NumPy if needed
                if is_gpu_tensor:
                    vectors = vectors.cpu().numpy()
                similarities = vector_ops.masked_cosine_similarity_batch(query_vector, vectors)
            compute_time = time.time() - compute_start
            total_compute_time += compute_time
            
            total_vectors_processed += actual_chunk_size
            
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
            
            # Update progress bar
            processed = (chunk_idx + 1) * self.chunk_size
            last_update = self._update_progress_bar(
                processed, total_to_process, start_time, last_update
            )
        
        # Sort results
        sorted_indices = np.argsort(-top_similarities)
        top_similarities = top_similarities[sorted_indices]
        top_indices = top_indices[sorted_indices]
        
        # Finalize progress bar
        self._complete_progress_bar(total_to_process, total_to_process, start_time)
        print(f"\n✅ Random chunk search complete")

        # Calculate performance metrics
        transfer_bytes = total_vectors_processed * 128  # 32 dimensions * 4 bytes
        transfer_bw = transfer_bytes / total_transfer_time / 1e9 if total_transfer_time > 0 else 0
        compute_throughput = total_vectors_processed / total_compute_time / 1e6 if total_compute_time > 0 else 0

        # Store performance stats
        self.performance_stats = {
            "transfer_time": total_transfer_time,
            "compute_time": total_compute_time,
            "transfer_bw": transfer_bw,
            "compute_throughput": compute_throughput,
            "total_vectors": total_vectors_processed
        }

        return top_indices.tolist(), top_similarities.tolist()

    def _cpu_random_chunk_search(self,
                                query_vector: np.ndarray,
                                vector_source: Callable[[int, int], np.ndarray],
                                total_vectors: int,
                                vector_ops: VectorOps,
                                num_chunks: int = 100,
                                top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Pure CPU implementation of random chunk search (original version)
        """
        if query_vector.shape != (self.VECTOR_DIMENSIONS,):
            raise ValueError(f"Query vector must be {self.VECTOR_DIMENSIONS}D")
        
        # Initialize results
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        # Initialize progress bar
        start_time = self._init_progress_bar(
            total_to_process,
            f"Random chunk search ({num_chunks} chunks, {total_to_process:,} total vectors"
        )
        last_update = start_time
        
        for chunk_idx in range(num_chunks):
            # Pick a random chunk start
            max_start = total_vectors - self.chunk_size
            chunk_start = random.randint(0, max_start) if max_start > 0 else 0
            chunk_end = min(chunk_start + self.chunk_size, total_vectors)
            actual_chunk_size = chunk_end - chunk_start
            
            # Read vectors for this chunk
            vectors = vector_source(chunk_start, actual_chunk_size)
            
            # Compute similarities
            similarities = vector_ops.masked_cosine_similarity_batch(query_vector, vectors)
            
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
            
            # Update progress bar
            processed = (chunk_idx + 1) * self.chunk_size
            last_update = self._update_progress_bar(
                processed, total_to_process, start_time, last_update
            )
        
        # Sort results
        sorted_indices = np.argsort(-top_similarities)
        top_similarities = top_similarities[sorted_indices]
        top_indices = top_indices[sorted_indices]
        
        # Finalize progress bar
        self._complete_progress_bar(total_to_process, total_to_process, start_time)
        print(f"\n✅ Random chunk search complete")

        return top_indices.tolist(), top_similarities.tolist()
    
    def progressive_search(self,
                          query_vector: np.ndarray,
                          vector_source: Callable[[int, int], np.ndarray],
                          total_vectors: int,
                          vector_ops: VectorOps,
                          min_chunks: int = 1,
                          max_chunks: int = 100,
                          quality_threshold: float = 0.95,
                          top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Perform a progressive search on the vector cache 
        until the desired quality threshold is reached.
        
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
        
        # Performance monitoring
        total_transfer_time = 0.0
        total_compute_time = 0.0
        total_vectors_processed = 0
        
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
            
            # Accumulate performance stats
            if hasattr(self, 'performance_stats'):
                total_transfer_time += self.performance_stats.get('transfer_time', 0)
                total_compute_time += self.performance_stats.get('compute_time', 0)
                total_vectors_processed += self.performance_stats.get('total_vectors', 0)
            
            # Check if we have good enough results
            if similarities and similarities[0] >= quality_threshold:
                print(f"✅ Quality threshold reached: {similarities[0]:.4f} >= {quality_threshold}")
                break
            
            # Double chunk count for next iteration
            current_chunks = min(current_chunks * 2, max_chunks)
            best_indices, best_similarities = indices, similarities
        
        if current_chunks > max_chunks:
            print(f"⚠️  Max chunks reached, returning best found")
        
        # Store performance stats
        transfer_bytes = total_vectors_processed * 128
        transfer_bw = transfer_bytes / total_transfer_time / 1e9 if total_transfer_time > 0 else 0
        compute_throughput = total_vectors_processed / total_compute_time / 1e6 if total_compute_time > 0 else 0
        
        self.performance_stats = {
            "transfer_time": total_transfer_time,
            "compute_time": total_compute_time,
            "transfer_bw": transfer_bw,
            "compute_throughput": compute_throughput,
            "total_vectors": total_vectors_processed
        }

        return best_indices, best_similarities

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable units."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
    
    def _init_progress_bar(self, total_vectors: int, description: str):
        """Initialize the progress bar display."""
        print(f"{description}")
        print(f"  [{'░' * self.progress_bar_width}] 0.0%")
        print(f"   Speed: -- vectors/second | ETA: --")
        sys.stdout.flush()
        return time.time()
    
    def _update_progress_bar(self, processed: int, total: int, start_time: float, last_update: float):
        """
        Update the progress bar display.
        Returns the current time if updated, otherwise returns last_update.
        """
        current_time = time.time()
        if current_time - last_update < 0.5:
            return last_update
        
        elapsed = current_time - start_time
        percent = processed / total
        
        # Calculate speed and ETA
        speed = processed / elapsed if elapsed > 0 else 0
        remaining = total - processed
        eta = remaining / speed if speed > 0 else 0
        
        # Format speed and ETA
        speed_str = f"{speed/1e6:.2f}M" if speed > 1e6 else f"{speed/1e3:.1f}K"
        eta_str = self._format_time(eta)
        
        # Update progress bar
        filled = int(self.progress_bar_width * percent)
        bar = '█' * filled + '░' * (self.progress_bar_width - filled)
        
        # Move cursor up and rewrite lines
        sys.stdout.write("\033[2A")  # Move up two lines
        sys.stdout.write("\033[K")   # Clear line
        sys.stdout.write(f"  [{bar}] {percent:.1%}\n")
        sys.stdout.write(f"   Speed: {speed_str} vectors/second | ETA: {eta_str}\n")
        sys.stdout.flush()
        
        return current_time
    
    def _complete_progress_bar(self, processed: int, total: int, start_time: float):
        """Display final progress bar with summary statistics."""
        elapsed = time.time() - start_time
        avg_speed = total / elapsed
        
        # Move cursor up for final update
        sys.stdout.write("\033[2A")
        sys.stdout.write("\033[K")
        sys.stdout.write(f"  [{'█' * self.progress_bar_width}] 100.0%\n")
        
        # Format speed appropriately
        if avg_speed > 1e6:
            speed_str = f"{avg_speed/1e6:.2f}M"
        elif avg_speed > 1e3:
            speed_str = f"{avg_speed/1e3:.1f}K"
        else:
            speed_str = f"{avg_speed:.0f}"
            
        sys.stdout.write(f"   Average speed: {speed_str} vectors/second | Total time: {self._format_time(elapsed)}\n")
        sys.stdout.flush()
