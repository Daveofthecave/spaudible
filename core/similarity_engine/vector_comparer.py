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
from config import FRAME_WIDTH

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
                        vector_ops: VectorOps,
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
                total_vectors, vector_ops, top_k, max_vectors, show_progress,
                query_region=query_region, region_strength=region_strength
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
                             vector_source: Callable[[int, int], torch.Tensor],
                             mask_source: Callable[[int, int], torch.Tensor],
                             region_source: Callable[[int, int], torch.Tensor],
                             total_vectors: int,
                             vector_ops: VectorOps,
                             top_k: int = 10,
                             max_vectors: Optional[int] = None,
                             show_progress: bool = True,
                             **kwargs) -> Tuple[List[int], List[float]]:
        """
        CPU-based scan with adaptive chunk resizing using a hill climbing algorithm.
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
        
        # === Adaptive Chunk Resizer ===
        min_chunk_size = 2_000
        max_chunk_size = 100_000_000
        current_chunk_size = config_manager.get_optimal_chunk_size()
        if not (min_chunk_size <= current_chunk_size <= max_chunk_size):
            current_chunk_size = 200_000  # Fallback to default if corrupted
        
        # State tracking
        speed_history = deque(maxlen=15)
        size_history = deque(maxlen=15)
        
        # Optimization state variables for the adaptive chunk resizer
        best_speed = 0.0
        best_chunk_size = current_chunk_size
        direction = 0  # +1=increasing, -1=decreasing, 0=exploring
        step_size = 1.25  # Initial step multiplier
        last_speed = 0.0
        
        # Multi-phase optimization to handle cache warmup
        warmup_threshold = min(vectors_to_scan * 0.20, 30_000_000)  # 20% or 30M vectors
        exploration_phase = True
        stable_configurations = deque(maxlen=5)  # Track consistent performers
        
        # === Periodic forced exploration ===
        # Forces a downward probe every N vectors to escape bad basins
        last_exploration_reset = 0
        exploration_interval = 50_000_000  # Every 50M vectors, force exploration
        exploration_factor = 3  # Divide current chunk size by this amount
        
        processed_count = 0
        samples_since_adjustment = 0
        
        # Extract region parameters from kwargs
        query_region = kwargs.get('query_region', -1)
        region_strength = kwargs.get('region_strength', 1.0)
        
        # Store last adaptation message for display
        last_adaptation_msg = ""
        adaptation_display_time = 0
        
        # Force initial render to establish 3-line layout
        if show_progress:
            sys.stdout.flush()
        
        while processed_count < vectors_to_scan:
            # === Force exploration every interval ===
            # This prevents permanent entrapment in large chunk sizes
            if processed_count - last_exploration_reset >= exploration_interval:
                # Jump to a much smaller size to test if smaller is better
                new_chunk_size = max(min_chunk_size, current_chunk_size // exploration_factor)
                
                # Reset only if we're not already at a small size
                if new_chunk_size < current_chunk_size * 0.9:
                    # Reset momentum to allow upward climb from this new size
                    direction = -1  # Start decreasing from here
                    step_size = 1.25
                    last_exploration_reset = processed_count
                    
                    current_chunk_size = new_chunk_size
                    continue  # Skip normal adaptation this iteration
            
            # Calculate chunk boundaries
            chunk_start = processed_count
            actual_chunk_size = min(current_chunk_size, vectors_to_scan - processed_count)
            
            # Wall-clock timing
            chunk_wall_start = time.time()
            
            # Read data from unified vector file
            vectors = vector_source(chunk_start, actual_chunk_size)
            masks = mask_source(chunk_start, actual_chunk_size)
            
            # Read regions for filtering
            regions = region_source(chunk_start, actual_chunk_size)
            
            # Convert to numpy for Numba kernels
            vectors_np = vectors.numpy()
            masks_np = masks.numpy()
            regions_np = regions.numpy()
            
            # Compute similarities with vectorized operations
            similarities = vector_ops.compute_similarity(query_vector, vectors_np, masks_np)
            
            # Apply region filtering if enabled
            if query_region >= 0 and region_strength > 0.0:
                region_match = (regions_np == query_region).astype(np.float32)
                region_penalty = np.where(
                    region_match == 1.0,
                    1.0,
                    1.0 - region_strength
                )
                similarities *= region_penalty
            
            chunk_wall_time = time.time() - chunk_wall_start
            
            # Update global top-k efficiently
            if actual_chunk_size > 0:
                chunk_top_k = min(top_k, actual_chunk_size)
                chunk_top_indices = np.argpartition(-similarities, chunk_top_k)[:chunk_top_k]
                chunk_top_values = similarities[chunk_top_indices]
                
                combined_sim = np.concatenate([top_similarities, chunk_top_values])
                combined_idx = np.concatenate([top_indices, chunk_top_indices + chunk_start])
                
                new_top_k = min(len(combined_sim), len(top_similarities))
                top_indices_in_combined = np.argpartition(-combined_sim, new_top_k)[:new_top_k]
                
                top_similarities = combined_sim[top_indices_in_combined]
                top_indices = combined_idx[top_indices_in_combined]
            
            # === Adaptive Chunk Resizer ===
            new_chunk_size = current_chunk_size  # Default: no change
            
            if chunk_wall_time > 0:
                speed = actual_chunk_size / chunk_wall_time
                speed_history.append(speed)
                size_history.append(current_chunk_size)
                
                # Track best speed but ignore early cache-warmed results
                if speed > best_speed and processed_count > warmup_threshold:
                    best_speed = speed
                    best_chunk_size = current_chunk_size
                
                samples_since_adjustment += 1
                
                # Run adaptation every 3 chunks with sufficient history
                if samples_since_adjustment >= 3 and len(speed_history) >= 5:
                    recent_speeds = list(speed_history)[-5:]
                    recent_sizes = list(size_history)[-5:]
                    avg_speed = np.mean(recent_speeds)
                    
                    # Simple hill climbing: measure speed change
                    speed_change = 0
                    if last_speed > 0:
                        speed_change = (avg_speed - last_speed) / last_speed
                    
                    # Phase transition: exit exploration after warmup
                    if exploration_phase and processed_count > warmup_threshold:
                        exploration_phase = False
                        # Reset best to ignore cache-warmed values
                        best_speed = 0.0
                    
                    # Adjust direction and step size based on performance
                    if speed_change > 0.01:  # Speed improved
                        direction = 1 if direction >= 0 else -1  # Continue current direction
                        step_size = min(1.5, step_size * 1.02)  # Slightly more aggressive
                        
                        # Track stable configurations (only after warmup)
                        if not exploration_phase:
                            stable_configurations.append((avg_speed, current_chunk_size))
                    elif speed_change < -0.05:  # Speed dropped significantly
                        # Aggressive backoff with direction reversal
                        direction = -direction if direction != 0 else -1
                        step_size = max(1.05, step_size * 0.6)  # Very aggressive backoff
                        
                        # If we regressed significantly, reset to best known configuration
                        if avg_speed < best_speed * 0.85 and best_speed > 0:
                            new_chunk_size = best_chunk_size
                    else:  # Stable or small change
                        # Reduce momentum gradually
                        step_size = max(1.05, step_size * 0.95)
                        # Decay direction gradually
                        if abs(direction) > 0.1:
                            direction *= 0.9
                        
                        # Track stable configurations (only after warmup)
                        if not exploration_phase:
                            stable_configurations.append((avg_speed, current_chunk_size))
                    
                    # Calculate new chunk size if we haven't set it via reset
                    if new_chunk_size == current_chunk_size and abs(direction) > 0.1:
                        potential_new_size = int(current_chunk_size * (step_size ** direction))
                        new_chunk_size = max(min_chunk_size, min(max_chunk_size, potential_new_size))
                    
                    # Store adaptation message
                    if show_progress and new_chunk_size != current_chunk_size:
                        last_adaptation_msg = (
                            f"   Chunk size: {new_chunk_size:,} "
                            f"({speed_change:+.1%} speed)                "
                        )
                        adaptation_display_time = time.time()
                    
                    # Update tracking variables
                    last_speed = avg_speed
                    samples_since_adjustment = 0
            
            # Apply the new chunk size
            current_chunk_size = new_chunk_size
            
            # Update progress display
            processed_count += actual_chunk_size
            if show_progress:
                # Check if adaptation message should be cleared (after 2 seconds)
                clear_adaptation = (time.time() - adaptation_display_time > 2.0)
                
                last_update = self._update_progress_bar(
                    processed_count, vectors_to_scan, start_time, last_update,
                    last_adaptation_msg if not clear_adaptation else ""
                )
        
        if show_progress:
            self._complete_progress_bar(vectors_to_scan, processed_count, start_time)
        
        # Store the best chunk size discovered during this scan for the next run
        # Prefer sustainable performance over early cache-spiked performance
        if len(stable_configurations) > 0:
            # Use median of stable configurations for robustness
            stable_sizes = [size for _, size in stable_configurations]
            final_chunk_size = int(np.median(stable_sizes))
        elif best_speed > 0:  # Should only trigger if scan was very short
            final_chunk_size = best_chunk_size
        else:  # Ultimate fallback
            final_chunk_size = current_chunk_size
        
        config_manager.set_optimal_chunk_size(final_chunk_size)
        
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
        """Initialize progress bar display with 3 reserved lines"""
        print(f"\n{description}")
        print(f"  [{'░' * self.PROGRESS_BAR_WIDTH}] 0.0%")
        print(f"   Speed: -- vectors/sec | ETA: --")
        print(" " * FRAME_WIDTH)  # Reserve third line for adaptation
        sys.stdout.flush()
        return time.time()

    def _update_progress_bar(self, processed: int, total: int, 
                        start_time: float, last_update: float,
                        adaptation_msg: str = "") -> float:
        """Update progress display with consistent 3-line positioning"""
        current_time = time.time()
        if current_time - last_update < 0.5:
            return last_update
        
        elapsed = current_time - start_time
        percent = processed / total
        speed = processed / elapsed if elapsed > 0 else 0
        remaining = total - processed
        eta = remaining / speed if speed > 0 else 0
        
        filled = int(self.PROGRESS_BAR_WIDTH * percent)
        speed_str = f"{speed/1e6:.2f}M" if speed > 1e6 else f"{speed/1e3:.1f}K"
        eta_str = format_elapsed_time(eta)
        
        # Always move up 3 lines and redraw all three
        sys.stdout.write("\033[3A\033[K")
        print(f"  [{'█' * filled}{'░' * (self.PROGRESS_BAR_WIDTH - filled)}] {percent:.1%}")
        print(f"   Speed: {speed_str} vectors/sec | ETA: {eta_str}")
        print(adaptation_msg if adaptation_msg else " " * FRAME_WIDTH)  # Clear third line if no message
        
        sys.stdout.flush()
        return current_time

    def _complete_progress_bar(self, total: int, processed: int, start_time: float):
        """Finalize progress bar with proper 3-line cleanup"""
        elapsed = time.time() - start_time
        avg_speed = processed / elapsed if elapsed > 0 else 0
        
        # Move up 3 lines and clear each line individually
        sys.stdout.write("\033[3A")
        
        # Line 1: Progress bar
        sys.stdout.write("\033[K")  # Clear line
        bar = '█' * self.PROGRESS_BAR_WIDTH
        print(f"  [{bar}] 100.0%")
        
        # Line 2: Final stats
        sys.stdout.write("\033[K")  # Clear line
        speed_str = f"{avg_speed/1e6:.2f}M" if avg_speed > 1e6 else f"{avg_speed/1e3:.1f}K"
        print(f"   Average: {speed_str} vectors/sec | Total: {format_elapsed_time(elapsed)}")
        
        # Line 3: Clear and move cursor to next line
        sys.stdout.write("\033[K")  # Clear line
        # print()  # Newline to move cursor to clean position
        sys.stdout.flush()
