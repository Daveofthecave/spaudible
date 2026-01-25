# core/similarity_engine/vector_comparer.py
"""
Chunked similarity search algorithms for unified vector format.
Only sequential search is maintained for maximum performance.
"""
import numpy as np
import sys
import time
import torch
from collections import deque
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
        """GPU implementation using CUDA kernels via PyTorch."""
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
        """
        CPU implementation with intelligent adaptive chunk sizing 
        using hill climbing algorithm.
        
        This algorithm continuously searches for the optimal chunk size by:
        1. Exploring initial directions to find performance gradient
        2. Climbing toward peak performance with momentum
        3. Backtracking when performance degrades significantly
        4. Detecting and stabilizing oscillations
        5. Tracking the best-known configuration for fallback
        """
        if query_vector.shape != (self.VECTOR_DIMENSIONS,):
            raise ValueError(f"Query vector must be {self.VECTOR_DIMENSIONS}D")
        
        # Initialize global results arrays
        top_similarities = np.full(top_k, -1.0, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)
        
        vectors_to_scan = min(total_vectors, max_vectors or total_vectors)
        
        # Progress tracking
        start_time = time.time()
        last_update = start_time
        
        if show_progress:
            self._init_progress_bar(vectors_to_scan, "🔍 CPU Sequential Scan")
        
        # === Enhanced Adaptive Chunk Resizing State ===
        current_chunk_size = 200_000
        min_chunk_size = 2_000
        max_chunk_size = 100_000_000
        
        # State tracking with automatic length limits
        speed_history = deque(maxlen=15)  # Rolling window of speed measurements
        size_history = deque(maxlen=15)   # Corresponding chunk sizes
        
        # Optimization state variables
        best_speed = 0.0
        best_chunk_size = current_chunk_size
        direction = 0  # +1=increasing, -1=decreasing, 0=exploring
        step_size = 1.15  # Conservative initial step multiplier
        patience_counter = 0
        max_patience = 3  # Steps before forced backtrack
        
        # Performance stability tracking
        state = "exploring"  # exploring, climbing, tracking, oscillating
        recent_improvements = deque(maxlen=5)
        
        processed_count = 0
        samples_since_adjustment = 0
        
        while processed_count < vectors_to_scan:
            # Calculate chunk boundaries
            chunk_start = processed_count
            actual_chunk_size = min(current_chunk_size, vectors_to_scan - processed_count)
            
            # Wall-clock timing
            chunk_wall_start = time.time()
            
            # Read data from unified vector file
            vectors = vector_source(chunk_start, actual_chunk_size)
            masks = mask_source(chunk_start, actual_chunk_size)
            regions = region_source(chunk_start, actual_chunk_size)
            
            # Compute similarities with vectorized operations
            similarities = self.vector_ops.compute_similarity(query_vector, vectors, masks)
            
            # Apply region filtering if enabled
            if query_region >= 0 and region_strength > 0.0:
                region_match = (regions == query_region).astype(np.float32)
                region_penalty = np.where(region_match, 1.0, 1.0 - region_strength)
                similarities *= region_penalty
            
            chunk_wall_time = time.time() - chunk_wall_start
            
            # Update global top-k using efficient partial sort
            self._update_topk_cpu(similarities, chunk_start, top_similarities, top_indices)
            
            # === Adaptive Chunk Sizing Logic ===
            if chunk_wall_time > 0:
                speed = actual_chunk_size / chunk_wall_time
                speed_history.append(speed)
                size_history.append(current_chunk_size)
                
                # Track personal best for backtracking
                if speed > best_speed:
                    best_speed = speed
                    best_chunk_size = current_chunk_size
            
            samples_since_adjustment += 1
            
            # Run adaptation every 3 chunks with sufficient history
            if samples_since_adjustment >= 3 and len(speed_history) >= 5:
                recent_speeds = list(speed_history)[-5:]
                recent_sizes = list(size_history)[-5:]
                avg_speed = np.mean(recent_speeds)
                
                # Detect oscillation (frequent direction changes)
                if len(recent_sizes) >= 4:
                    size_changes = np.diff(recent_sizes)
                    direction_changes = np.sum(np.diff(np.sign(size_changes)) != 0)
                    if direction_changes >= 2:  # More than 1 reversal
                        state = "oscillating"
                        step_size = max(1.05, step_size * 0.6)
                
                # State-based adaptation
                if state == "exploring":
                    if direction == 0:
                        # Initial exploration: try increasing first
                        direction = +1
                        new_chunk_size = int(current_chunk_size * step_size)
                    else:
                        # Check if exploration is yielding improvement
                        baseline_speed = np.mean(list(speed_history)[0:5])
                        if avg_speed > baseline_speed * 1.05:
                            state = "climbing"
                        else:
                            # Try decreasing instead
                            direction = -1
                            state = "climbing"
                    
                    if show_progress and state == "climbing":
                        print(f"   Exploring: {current_chunk_size:,} → {new_chunk_size:,} "
                            f"(speed: {avg_speed/1e6:.2f}M vec/s)\n\n")
                
                elif state == "climbing":
                    # Measure improvement over pre-change baseline
                    prev_avg = np.mean(list(speed_history)[-6:-1])
                    improvement = (avg_speed - prev_avg) / (prev_avg + 1e-9)
                    
                    if improvement > 0.05:  # >5% speed gain
                        # Good direction - accelerate slightly
                        patience_counter = 0
                        step_size = min(2.0, step_size * 1.02)
                    elif improvement < -0.03:  # >3% speed loss
                        # Bad direction - reduce patience
                        patience_counter += 1
                        if (patience_counter >= max_patience or 
                            avg_speed < best_speed * 0.95):
                            # Backtrack to best-known configuration
                            state = "tracking"
                            direction = 0
                            new_chunk_size = best_chunk_size
                            step_size = max(1.05, step_size * 0.7)
                            if show_progress:
                                print(f"   ↩️  Backtracking to best: {best_chunk_size:,}\n\n")
                        else:
                            # Reverse direction
                            direction *= -1
                            step_size = max(1.05, step_size * 0.8)
                    else:
                        # Stable performance - coast with small steps
                        step_size = max(1.05, step_size * 0.98)
                    
                    # Continue in current direction unless backtracking
                    if state != "tracking":
                        new_chunk_size = int(current_chunk_size * (step_size ** direction))
                
                elif state == "tracking":
                    # Monitor if current conditions still support best config
                    if avg_speed < best_speed * 0.92:  # Degraded >8%
                        # Environment changed, re-explore
                        state = "exploring"
                        direction = 0
                        new_chunk_size = best_chunk_size
                    else:
                        # Stay near best configuration
                        new_chunk_size = best_chunk_size
                
                elif state == "oscillating":
                    # Stabilize at best-known size
                    new_chunk_size = best_chunk_size
                    step_size = max(1.05, step_size * 0.75)
                    # Exit oscillation if stable for 5+ samples
                    if np.std(recent_speeds) / (avg_speed + 1e-9) < 0.03:
                        state = "tracking"
                
                # Apply bounds and commit new size
                if state != "tracking":  # Don't modify if stabilizing/backtracking
                    current_chunk_size = max(min_chunk_size, 
                                        min(max_chunk_size, new_chunk_size))
                
                samples_since_adjustment = 0
            
            # Update progress display (throttled to max 2 updates/sec)
            processed_count += actual_chunk_size
            if show_progress:
                last_update = self._update_progress_bar(
                    processed_count, vectors_to_scan, start_time, last_update
                )
        
        if show_progress:
            self._complete_progress_bar(vectors_to_scan, processed_count, start_time)
        
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
        """CPU: Update global top-k using NumPy partitioning (no full sort)"""
        # Get top-k in this chunk
        chunk_top_k = min(len(similarities), len(top_sim))
        chunk_top_indices = np.argpartition(-similarities, chunk_top_k)[:chunk_top_k]
        chunk_top_values = similarities[chunk_top_indices]
        
        # Combine with global top-k
        combined_sim = np.concatenate([top_sim, chunk_top_values])
        combined_idx = np.concatenate([top_idx, chunk_top_indices + chunk_start])
        
        # Get new global top-k
        new_top_k = min(len(combined_sim), len(top_sim))
        global_indices = np.argpartition(-combined_sim, new_top_k)[:new_top_k]
        
        # Update in-place
        top_sim[:] = combined_sim[global_indices]
        top_idx[:] = combined_idx[global_indices]
    
    def _init_progress_bar(self, total: int, description: str):
        """Initialize progress bar display"""
        print(f"\n{description}")
        print(f"  [{'░' * self.PROGRESS_BAR_WIDTH}] 0.0%")
        print(f"   Speed: -- vectors/sec | ETA: --")
        sys.stdout.flush()
        return time.time()
    
    def _update_progress_bar(self, processed: int, total: int, 
                           start_time: float, last_update: float) -> float:
        """Update progress bar with throttling (max 2 updates/sec)"""
        current_time = time.time()
        if current_time - last_update < 0.5:  # Throttle updates
            return last_update
        
        elapsed = current_time - start_time
        percent = processed / total
        
        # Calculate speed and ETA
        speed = processed / elapsed if elapsed > 0 else 0
        remaining = total - processed
        eta = remaining / speed if speed > 0 else 0
        
        # Format display strings
        speed_str = f"{speed/1e6:.2f}M" if speed > 1e6 else f"{speed/1e3:.1f}K"
        eta_str = format_elapsed_time(eta).strip()
        
        # Update bar
        filled = int(self.PROGRESS_BAR_WIDTH * percent)
        bar = '█' * filled + '░' * (self.PROGRESS_BAR_WIDTH - filled)
        
        # Move cursor up 2 lines and overwrite
        sys.stdout.write("\033[2A\033[K")
        sys.stdout.write(f"  [{bar}] {percent:.1%}\n")
        sys.stdout.write(f"   Speed: {speed_str} vectors/sec | ETA: {eta_str}\n")
        sys.stdout.flush()
        
        return current_time
    
    def _complete_progress_bar(self, total: int, processed: int, start_time: float):
        """Finalize progress bar with summary"""
        elapsed = time.time() - start_time
        avg_speed = processed / elapsed if elapsed > 0 else 0
        
        # Move cursor up for final update
        sys.stdout.write("\033[2A\033[K")
        bar = '█' * self.PROGRESS_BAR_WIDTH
        print(f"  [{bar}] 100.0%")
        
        # Format final speed
        speed_str = f"{avg_speed/1e6:.2f}M" if avg_speed > 1e6 else f"{avg_speed/1e3:.1f}K"
        print(f"   Average: {speed_str} vectors/sec | Total: {format_elapsed_time(elapsed)}\n")
