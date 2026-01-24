# core/similarity_engine/vector_comparer.py
"""
Chunked similarity search algorithms for unified vector format.
Only sequential search is maintained for maximum performance.
"""

import numpy as np
import sys
import time
import torch
from typing import List, Tuple, Optional, Callable
from .vector_math import VectorOps
from .vector_math_gpu import VectorOpsGPU
from ui.cli.console_utils import format_elapsed_time
from core.utilities.config_manager import config_manager

class ChunkedSearch:
    """GPU-accelerated sequential similarity search."""
    
    VECTOR_DIMENSIONS = 32
    PROGRESS_BAR_WIDTH = 50
    
    def __init__(self, 
                 chunk_size: int = 100_000_000,
                 use_gpu: bool = True,
                 vector_ops: Optional[VectorOps] = None):
        """
        Initialize chunked search with unified format support.
        
        Args:
            chunk_size: Safe number of vectors to process per chunk (GPU limit or CPU chunk)
            use_gpu: Enable GPU acceleration
            vector_ops: Vector operations instance (contains algorithm)
        """
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu
        self.vector_ops = vector_ops
        
        # Extract algorithm from vector_ops or use default
        self.algorithm = getattr(vector_ops, 'algorithm', 'cosine-euclidean') if vector_ops else 'cosine-euclidean'
        
        # Device determination
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        # GPU ops initialization
        self.gpu_ops = None
        if self.use_gpu and self.device == "cuda":
            try:
                free_vram = torch.cuda.mem_get_info()[0]
                if free_vram < 500_000_000:  # 500MB minimum
                    print("⚠️ Low VRAM available, disabling GPU acceleration")
                    self.use_gpu = False
                    self.device = "cpu"
                else:
                    self.gpu_ops = VectorOpsGPU(device=self.device)
                    self.gpu_ops.set_user_weights(config_manager.get_weights())
            except Exception as e:
                print(f"⚠️  GPU initialization failed: {e}")
                self.gpu_ops = None
    
    def sequential_scan(self,
                        query_vector: np.ndarray,
                        vector_source: Callable[[int, int], torch.Tensor],
                        mask_source: Callable[[int, int], torch.Tensor],
                        region_source: Callable[[int, int], torch.Tensor],
                        total_vectors: int,
                        top_k: int = 10,
                        max_vectors: Optional[int] = None,
                        show_progress: bool = True,
                        query_region: int = -1,
                        region_strength: float = 1.0) -> Tuple[List[int], List[float]]:
        """
        Perform sequential scan - the ONLY search method for maximum performance.
        
        Args:
            query_vector: 32D query vector
            vector_source: Function to read vector chunks
            mask_source: Function to read mask chunks
            region_source: Function to read region chunks
            total_vectors: Total number of vectors to scan
            top_k: Number of top results to return
            max_vectors: Optional limit on vectors to scan
            show_progress: Show progress bar
            query_region: Region code for filtering (-1 = disabled)
            region_strength: Strength of region filtering (0.0-1.0)
        """
        if self.use_gpu and self.gpu_ops:
            return self._gpu_sequential_scan(
                query_vector, vector_source, mask_source, region_source,
                total_vectors, top_k, max_vectors, show_progress, query_region, region_strength
            )
        else:
            return self._cpu_sequential_scan(
                query_vector, vector_source, mask_source, region_source,
                total_vectors, top_k, max_vectors, show_progress, query_region, region_strength
            )
    
    def _gpu_sequential_scan(self,
                             query_vector: np.ndarray,
                             vector_source: Callable,
                             mask_source: Callable,
                             region_source: Callable,
                             total_vectors: int,
                             top_k: int,
                             max_vectors: Optional[int],
                             show_progress: bool,
                             query_region: int,
                             region_strength: float) -> Tuple[List[int], List[float]]:
        """GPU implementation using CUDA kernels."""
        # Move query to GPU
        query_t = torch.tensor(
            query_vector, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Use tuples for tensor dimensions (fixes shape error)
        top_similarities = torch.full(
            (top_k,),                     # Tuple for 1D tensor
            -1.0, 
            dtype=torch.float32, 
            device=self.device
        )
        top_indices = torch.full(
            (top_k,),                     # Tuple for 1D tensor
            -1, 
            dtype=torch.long, 
            device=self.device
        )
        
        # Calculate scan range
        vectors_to_scan = min(total_vectors, max_vectors or total_vectors)
        num_chunks = (vectors_to_scan + self.chunk_size - 1) // self.chunk_size
        
        # Progress tracking
        start_time = time.time()
        last_update = start_time
        
        if show_progress:
            self._init_progress_bar(vectors_to_scan, "🔍 GPU Sequential Scan")
        
        processed_vectors = 0
        
        for chunk_idx in range(num_chunks):
            # Calculate chunk bounds
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, vectors_to_scan)
            actual_chunk_size = chunk_end - chunk_start
            
            # Read vector, mask, and region data
            vectors_gpu = vector_source(chunk_start, actual_chunk_size)
            masks_gpu = mask_source(chunk_start, actual_chunk_size)
            regions_gpu = region_source(chunk_start, actual_chunk_size)
            
            # Compute similarities
            if query_region >= 0 and region_strength > 0.0:
                # Apply region-aware similarity
                similarities = self.gpu_ops.fused_similarity(
                    query_t, vectors_gpu, masks_gpu, regions_gpu,
                    query_region, region_strength, self.algorithm
                )
            else:
                # Standard similarity by algorithm
                if self.algorithm == 'cosine':
                    similarities = self.gpu_ops.masked_weighted_cosine_similarity(
                        query_t, vectors_gpu, masks_gpu
                    )
                elif self.algorithm == 'euclidean':
                    similarities = self.gpu_ops.masked_euclidean_similarity(
                        query_t, vectors_gpu, masks_gpu
                    )
                else:  # cosine-euclidean (default)
                    similarities = self.gpu_ops.masked_weighted_cosine_euclidean_similarity(
                        query_t, vectors_gpu, masks_gpu
                    )
            
            # Ensure similarities is a tensor (fixes type error)
            if not isinstance(similarities, torch.Tensor):
                similarities = torch.tensor(similarities, device=self.device, dtype=torch.float32)
            
            # Update global top-k
            self._update_topk(similarities, chunk_start, top_similarities, top_indices)
            
            # Update progress
            processed_vectors += actual_chunk_size
            if show_progress:
                last_update = self._update_progress_bar(
                    processed_vectors, vectors_to_scan, start_time, last_update
                )
        
        if show_progress:
            self._complete_progress_bar(vectors_to_scan, processed_vectors, start_time)
        
        # Return results on CPU
        return top_indices.cpu().numpy(), top_similarities.cpu().numpy()

    def _cpu_sequential_scan(self,
                             query_vector: np.ndarray,
                             vector_source: Callable,
                             mask_source: Callable,
                             region_source: Callable,
                             total_vectors: int,
                             top_k: int,
                             max_vectors: Optional[int],
                             show_progress: bool,
                             query_region: int,
                             region_strength: float) -> Tuple[List[int], List[float]]:
        """CPU implementation using NumPy."""
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        vectors_to_scan = min(total_vectors, max_vectors or total_vectors)
        num_chunks = (vectors_to_scan + self.chunk_size - 1) // self.chunk_size
        
        start_time = time.time()
        last_update = start_time
        
        if show_progress:
            self._init_progress_bar(vectors_to_scan, "🔍 CPU Sequential Scan")
        
        processed_vectors = 0
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, vectors_to_scan)
            actual_chunk_size = chunk_end - chunk_start
            
            # Read data
            vectors = vector_source(chunk_start, actual_chunk_size)
            masks = mask_source(chunk_start, actual_chunk_size)
            regions = region_source(chunk_start, actual_chunk_size)
            
            # Convert tensors to numpy if needed
            if isinstance(vectors, torch.Tensor):
                vectors = vectors.cpu().numpy()
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(regions, torch.Tensor):
                regions = regions.cpu().numpy()
            
            # Compute similarities
            if query_region >= 0 and region_strength > 0.0:
                similarities = self.vector_ops.fused_similarity(
                    query_vector, vectors, masks, regions,
                    query_region, region_strength, self.algorithm
                )
            else:
                similarities = self.vector_ops.compute_similarity(query_vector, vectors, masks)
            
            # Update top-k
            self._update_topk_cpu(similarities, chunk_start, top_similarities, top_indices)
            
            # Update progress
            processed_vectors += actual_chunk_size
            if show_progress:
                last_update = self._update_progress_bar(
                    processed_vectors, vectors_to_scan, start_time, last_update
                )
        
        if show_progress:
            self._complete_progress_bar(vectors_to_scan, processed_vectors, start_time)
        
        return top_indices.tolist(), top_similarities.tolist()

    def _update_topk(self, similarities: torch.Tensor, chunk_start: int,
                    top_sim: torch.Tensor, top_idx: torch.Tensor):
        """GPU: Update global top-k."""
        chunk_top_k = min(len(similarities), len(top_sim))
        chunk_vals, chunk_indices = torch.topk(similarities, chunk_top_k)
        
        combined_vals = torch.cat([top_sim, chunk_vals])
        combined_indices = torch.cat([top_idx, chunk_indices + chunk_start])
        
        new_top_k = min(len(combined_vals), len(top_sim))
        global_vals, global_pos = torch.topk(combined_vals, new_top_k)
        
        top_sim.copy_(global_vals)
        top_idx.copy_(combined_indices[global_pos])
    
    def _update_topk_cpu(self, similarities: np.ndarray, chunk_start: int,
                        top_sim: np.ndarray, top_idx: np.ndarray):
        """CPU: Update global top-k."""
        chunk_top_k = min(len(similarities), len(top_sim))
        chunk_indices = np.argpartition(-similarities, chunk_top_k)[:chunk_top_k]
        chunk_vals = similarities[chunk_indices]
        
        combined_vals = np.concatenate([top_sim, chunk_vals])
        combined_indices = np.concatenate([top_idx, chunk_indices + chunk_start])
        
        new_top_k = min(len(combined_vals), len(top_sim))
        top_pos = np.argpartition(-combined_vals, new_top_k)[:new_top_k]
        
        top_sim[:] = combined_vals[top_pos]
        top_idx[:] = combined_indices[top_pos]
    
    def _init_progress_bar(self, total: int, description: str):
        """Initialize progress bar display."""
        print(f"\n{description}")
        print(f"  [{'░' * self.PROGRESS_BAR_WIDTH}] 0.0%")
        print(f"   Speed: -- vectors/sec | ETA: --")
        return time.time()
    
    def _update_progress_bar(self, processed: int, total: int,
                           start_time: float, last_update: float) -> float:
        """Update progress with throttling."""
        current_time = time.time()
        if current_time - last_update < 0.5:
            return last_update
        
        elapsed = current_time - start_time
        percent = processed / total
        speed = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / speed if speed > 0 else 0
        
        speed_str = f"{speed/1e6:.2f}M" if speed > 1e6 else f"{speed/1e3:.1f}K"
        eta_str = format_elapsed_time(eta)
        
        filled = int(self.PROGRESS_BAR_WIDTH * percent)
        bar = '█' * filled + '░' * (self.PROGRESS_BAR_WIDTH - filled)
        
        sys.stdout.write("\033[2A\033[K")
        print(f"  [{bar}] {percent:.1%}")
        print(f"   Speed: {speed_str} vectors/sec | ETA: {eta_str}")
        
        return current_time
    
    def _complete_progress_bar(self, total: int, processed: int, start_time: float):
        """Finalize progress bar."""
        elapsed = time.time() - start_time
        avg_speed = processed / elapsed if elapsed > 0 else 0
        
        sys.stdout.write("\033[2A\033[K")
        bar = '█' * self.PROGRESS_BAR_WIDTH
        print(f"  [{bar}] 100.0%")
        
        speed_str = f"{avg_speed/1e6:.2f}M" if avg_speed > 1e6 else f"{avg_speed/1e3:.1f}K"
        print(f"   Average: {speed_str} vectors/sec | Total: {format_elapsed_time(elapsed)}\n")
