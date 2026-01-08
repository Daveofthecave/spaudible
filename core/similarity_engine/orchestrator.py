# core/similarity_engine/orchestrator.py
import numpy as np
import re
import time
import torch
from config import PathConfig, VRAM_SAFETY_FACTOR, VRAM_SCALING_FACTOR_MB
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable
from .chunk_size_optimizer import ChunkSizeOptimizer
from .index_manager import IndexManager
from .metadata_service import MetadataManager
from .vector_comparer import ChunkedSearch
from .vector_io import VectorReader
from .vector_io_gpu import VectorReaderGPU
from .vector_math import VectorOps
from .vector_math_gpu import VectorOpsGPU
from core.utilities.gpu_utils import get_gpu_info, recommend_max_batch_size
from core.utilities.config_manager import config_manager
from core.vectorization.canonical_track_resolver import build_canonical_vector

class SearchOrchestrator:
    """High-level coordinator for similarity search operations."""
    
    # Class-level cache for benchmark results
    _benchmark_results = None
    
    def __init__(self, 
                vectors_path: Optional[str] = None, 
                index_path: Optional[str] = None,
                masks_path: Optional[str] = None,
                metadata_db: Optional[str] = None,
                chunk_size: int = 100_000_000,
                use_gpu: bool = True,
                vram_scaling_factor_mb: int = VRAM_SCALING_FACTOR_MB,
                skip_cpu_benchmark: bool = False,
                skip_gpu_benchmark: bool = False,
                skip_benchmark: bool = False,
                force_cpu: bool = False,
                force_gpu: bool = False,
                **kwargs):
        """
        Initialize the search orchestrator.
        
        Args:
            vectors_path: Path to track_vectors.bin (default: from PathConfig)
            masks_path: Path to track_masks.bin (default: from PathConfig)
            index_path: Path to track_index.bin (default: from PathConfig)
            metadata_db: Path to metadata database
            chunk_size: Chunk size for vector processing
            use_gpu: Whether to use GPU acceleration
            vram_scaling_factor_mb: Factor that scales how much VRAM to allocate
            skip_cpu_benchmark: Skip CPU benchmarking
            skip_gpu_benchmark: Skip GPU benchmarking
            skip_benchmark: Deprecated - skip both CPU and GPU benchmarking
            force_cpu: Whether CPU mode is forced
            force_gpu: Whether GPU mode is forced
            **kwargs: Additional keyword arguments
        """
        self.skip_cpu_benchmark = skip_cpu_benchmark or skip_benchmark
        self.skip_gpu_benchmark = skip_gpu_benchmark or skip_benchmark
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu
        self.force_cpu = config_manager.get_force_cpu()
        self.force_gpu = config_manager.get_force_gpu()
        self.vram_scaling_factor_mb = vram_scaling_factor_mb
        self._is_gpu_available(verbose=False)

        # Set default paths if not provided
        vectors_path = vectors_path or str(PathConfig.get_vector_file())
        masks_path = masks_path or str(PathConfig.get_mask_file())
        index_path = index_path or str(PathConfig.get_index_file())

        # Initialize components
        self.vector_reader = self._init_vector_reader(vectors_path, masks_path)
        self.algorithm = config_manager.get_algorithm()
        self.vector_ops = VectorOps(algorithm=self.algorithm)  # Initialize first

        # Apply user-defined weights AFTER initialization
        weights = config_manager.get_weights()
        self.vector_ops.set_user_weights(weights)

        self.index_manager = IndexManager(index_path)
        self.metadata_manager = MetadataManager(metadata_db)
        
        # Get max batch size from vector reader
        max_batch_size = None
        if hasattr(self.vector_reader, 'get_max_batch_size'):
            max_batch_size = self.vector_reader.get_max_batch_size()

        # Apply force settings
        if self.force_cpu:
            self.use_gpu = False
            print("ℹ️  CPU mode forced by configuration")
        elif self.force_gpu:
            self.use_gpu = True
            print("ℹ️  GPU mode forced by configuration")

        # Clear cache if settings changed
        if SearchOrchestrator._benchmark_results is not None:
            current_force_cpu = config_manager.get_force_cpu()
            current_force_gpu = config_manager.get_force_gpu()
            
            if (current_force_cpu != self.force_cpu) or (current_force_gpu != self.force_gpu):
                SearchOrchestrator._benchmark_results = None
        
        # Always run benchmark when neither mode is forced
        if not self.force_cpu and not self.force_gpu:
            # Run auto-benchmark if not done yet
            if SearchOrchestrator._benchmark_results is None:
                if not (self.skip_cpu_benchmark and self.skip_gpu_benchmark):
                    SearchOrchestrator._benchmark_results = self.run_auto_benchmark()
            
            # Apply benchmark results if available
            if SearchOrchestrator._benchmark_results:
                self.use_gpu = (SearchOrchestrator._benchmark_results['recommended_device'] == 'gpu')
                self.chunk_size = SearchOrchestrator._benchmark_results['optimal_chunk_size']
                # print(f"   Using {self.use_gpu and 'GPU' or 'CPU'} with vector chunk size {self.chunk_size:,}")
        
        # Handle forced CPU mode - always run benchmark
        if self.force_cpu:
            # Initialize CPU optimizer
            self.chunk_size_optimizer = ChunkSizeOptimizer(self.vector_reader)
            self.chunk_size = self.chunk_size_optimizer.optimize()
            # print(f"   Using CPU with vector chunk size {self.chunk_size:,}")
        
        # Handle forced GPU mode
        elif self.force_gpu:
            if self.use_gpu and hasattr(self.vector_reader, 'get_max_batch_size'):
                self.chunk_size = self.vector_reader.get_max_batch_size()
                # print(f"   Using GPU with vector chunk size {self.chunk_size:,}")
            else:
                # Fallback to CPU if GPU not available
                self.chunk_size_optimizer = ChunkSizeOptimizer(self.vector_reader)
                self.chunk_size = self.chunk_size_optimizer.optimize()
                # print(f"   Using CPU (GPU fallback) with vector chunk size {self.chunk_size:,}")
        
        # Handle non-forced modes where benchmark wasn't run
        elif not hasattr(self, 'chunk_size'):
            if self.use_gpu and hasattr(self.vector_reader, 'get_max_batch_size'):
                self.chunk_size = self.vector_reader.get_max_batch_size()
                # print(f"   Using GPU with vector chunk size {self.chunk_size:,}")
            else:
                # Initialize CPU optimizer
                self.chunk_size_optimizer = ChunkSizeOptimizer(self.vector_reader)
                self.chunk_size = self.chunk_size_optimizer.optimize()
                # print(f"   Using CPU with vector chunk size {self.chunk_size:,}")
        
        # Pass self.use_gpu and max_batch_size to ChunkedSearch
        self.chunked_search = ChunkedSearch(
            self.chunk_size, 
            use_gpu=self.use_gpu,
            max_batch_size=max_batch_size,
            vector_ops=self.vector_ops
        )
        
        self.total_vectors = self.vector_reader.get_total_vectors()

        # Validate implementations (only when not forced)
        if not self.force_cpu and not self.force_gpu:
            try:
                if not self.skip_cpu_benchmark and not self.skip_gpu_benchmark:
                    self._validate_implementation_parity()
                    # print("  ✅ CPU/GPU implementations validated")
            except Exception as e:
                print(f"  ⚠️  Implementation validation failed: {e}")

    def run_auto_benchmark(self):
        """Run auto-benchmark with optimized test sizes"""
        # print("   Running auto-benchmark...")
        results = {
            'cpu_speed': 0,
            'gpu_speed': 0,
            'recommended_device': 'cpu',
            'optimal_chunk_size': 100_000
        }

        # Skip if GPU is forced
        if self.force_gpu:
            print("  ⏭️ Skipping benchmark due to GPU mode")
            return {
                'recommended_device': 'gpu',
                'optimal_chunk_size': self.vector_reader.get_max_batch_size()
            }
        
        # Skip GPU benchmarking if CPU is forced
        skip_gpu_benchmark = self.force_cpu or self.skip_gpu_benchmark
        
        # Create test vector
        test_vector = np.random.rand(32).astype(np.float32)
        
        # Benchmark CPU with optimized chunk sizes
        if not self.skip_cpu_benchmark:
            # print("   Benchmarking CPU with optimized chunk sizes...")
            cpu_orchestrator = SearchOrchestrator(
                skip_cpu_benchmark=True,
                skip_gpu_benchmark=True,
                use_gpu=False,
                force_cpu=self.force_cpu  # Pass force_cpu setting
            )
            
            # Use CPU optimizer directly
            optimizer = ChunkSizeOptimizer(cpu_orchestrator.vector_reader)
            best_chunk = optimizer.optimize()
            cpu_orchestrator.chunk_size = best_chunk
            
            # Create new ChunkedSearch with vector_ops
            cpu_orchestrator.chunked_search = ChunkedSearch(
                best_chunk,
                use_gpu=False,
                vector_ops=cpu_orchestrator.vector_ops
            )
            
            # Run test with optimal chunk size
            result = cpu_orchestrator.run_performance_test(test_vector, 1_000_000, show_progress=False)
            results['cpu_speed'] = result['speed']
            results['optimal_cpu_chunk_size'] = best_chunk
            # print(f"   Optimal CPU chunk: {best_chunk:,} ({result['speed']/1e6:.2f}M vec/sec)")
            
            cpu_orchestrator.close()
        
        # Benchmark GPU only if available and not skipped
        if torch.cuda.is_available() and not skip_gpu_benchmark:
            print("   Benchmarking GPU with max batch size...")
            gpu_orchestrator = SearchOrchestrator(
                skip_cpu_benchmark=True,
                skip_gpu_benchmark=True,
                use_gpu=True
            )
            
            # Get max batch size from vector reader
            max_batch = gpu_orchestrator.vector_reader.get_max_batch_size()
            # print(f"      Max GPU batch: {max_batch:,} vectors")
            
            # Run test with max batch size - show progress bars
            gpu_result = gpu_orchestrator.run_performance_test(test_vector, max_batch, show_progress=True)
            results['gpu_speed'] = gpu_result['speed']
            # print(f"   GPU speed: {gpu_result['speed']/1e6:.2f}M vec/sec")
            
            gpu_orchestrator.close()
        
        # Determine optimal configuration - respect force_cpu
        if self.force_cpu:
            results['recommended_device'] = 'cpu'
            results['optimal_chunk_size'] = results.get('optimal_cpu_chunk_size', 100_000)
        elif results.get('gpu_speed', 0) > results.get('cpu_speed', 0) * 1.1:
            results['recommended_device'] = 'gpu'
            results['optimal_chunk_size'] = max_batch
        else:
            results['recommended_device'] = 'cpu'
            results['optimal_chunk_size'] = results.get('optimal_cpu_chunk_size', 100_000)
        
        # print(f"✅ Auto-benchmark complete: Using {results['recommended_device'].upper()} "
            # f"with chunk size {results['optimal_chunk_size']:,}")
        return results
        
    def _is_gpu_available(self, verbose: bool = False):
        """Validate GPU availability considering force settings"""
        if self.force_cpu:
            if verbose: 
                print("ℹ️  Using CPU (forced by configuration)")
            self.use_gpu = False
            return False
            
        if self.force_gpu:
            if not torch.cuda.is_available():
                print("⚠️ GPU not available but forced by configuration")
                return False
            if verbose:
                print("ℹ️  Using GPU (forced by configuration)")
            self.use_gpu = True
            return True
            
        if self.use_gpu and torch.cuda.is_available():
            if verbose: 
                print("ℹ️  Using GPU")
            return True
        else:
            if verbose: 
                print("ℹ️  Using CPU...")
            self.use_gpu = False
            return False

    def _init_vector_reader(self, vectors_path: str, masks_path: str):
        """Initialize vector reader with GPU support if available"""
        # Force GPU if requested
        if self.force_gpu and torch.cuda.is_available():
            try:
                return VectorReaderGPU(vectors_path, masks_path, vram_scaling_factor_mb=self.vram_scaling_factor_mb)
            except ImportError:
                print("  ⚠️  GPU acceleration not available")
                return VectorReader(vectors_path, masks_path)
        
        # Normal GPU detection
        if self._is_gpu_available():
            try:
                return VectorReaderGPU(vectors_path, masks_path, vram_scaling_factor_mb=self.vram_scaling_factor_mb)
            except ImportError:
                return VectorReader(vectors_path, masks_path)
        else:
            return VectorReader(vectors_path, masks_path)  
    
    def _vector_source(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Wrapper for vector reader."""
        return self.vector_reader.read_chunk(start_idx, num_vectors)
    
    def _mask_source(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Wrapper for mask reader."""
        return self.vector_reader.read_masks(start_idx, num_vectors)
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        search_mode: str = "sequential",
        top_k: int = 10,
        with_metadata: bool = True,
        show_progress: bool = True,
        deduplicate: Optional[bool] = None,  # None means use config setting
        **kwargs
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Find similar tracks to a query vector.
        """
        # Convert query to numpy array
        if isinstance(query_vector, list):
            query_np = self.vector_ops.to_numpy_array(query_vector)
        else:
            query_np = query_vector
        
        # Execute chosen search algorithm
        if search_mode == "sequential":
            indices, similarities = self.chunked_search.sequential_scan(
                query_np,
                self._vector_source,
                self._mask_source,
                self.total_vectors,
                top_k=top_k * 3,  # Get extra candidates for deduplication
                show_progress=show_progress,
                **kwargs
            )
        elif search_mode == "random":
            indices, similarities = self.chunked_search.random_chunk_search(
                query_np,
                self._vector_source,
                self._mask_source,
                self.total_vectors,
                num_chunks=100,
                top_k=top_k * 3,  # Get extra candidates
                show_progress=show_progress,
                **kwargs
            )
        elif search_mode == "progressive":
            indices, similarities = self.chunked_search.progressive_search(
                query_np,
                self._vector_source,
                self._mask_source,
                self.total_vectors,
                min_chunks=1,
                max_chunks=100,
                quality_threshold=0.95,
                top_k=top_k * 3,  # Get extra candidates
                show_progress=show_progress,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
        
        # Apply deduplication if requested
        if deduplicate is None:
            deduplicate = config_manager.get_deduplicate()
            
        if deduplicate:
            dedupe_threshold = config_manager.get_dedupe_threshold()
            indices, similarities = self._advanced_deduplication(
                indices, 
                similarities, 
                top_k,
                dedupe_threshold
            )
        
        # Validate results completeness
        if len(indices) < top_k and not deduplicate:
            print(f"\n  ⚠️  Warning: Only {len(indices)} results found (requested {top_k})")
            print("  This may indicate incomplete processing due to low VRAM settings.")
            print("  Consider increasing VRAM_SCALING_FACTOR_MB in config.py")

        # Convert indices to track IDs
        track_ids = self.index_manager.get_track_ids_batch(indices)
        
        # Get metadata if requested
        if with_metadata:
            metadata_list = self.metadata_manager.get_track_metadata_batch(track_ids)
            results = list(zip(track_ids, similarities, metadata_list))
        else:
            results = list(zip(track_ids, similarities))
        
        # Add secondary sort by popularity to break ties
        results = self._apply_secondary_sort(results, with_metadata)
        
        return results[:top_k]  # Ensure exactly top_k results

    def _advanced_deduplication(self, indices: List[int], similarities: List[float], 
                                top_k: int, dedupe_threshold: float) -> Tuple[List[int], List[float]]:
        """
        Optimized deduplication using ISRC + core metadata signature.
        Removes duplicates while preserving version diversity and guaranteeing top_k results.
        """
        # Get ISRCs and track IDs
        isrcs = self.index_manager.get_isrcs_batch(indices)
        track_ids = self.index_manager.get_track_ids_batch(indices)
        
        # Fetch metadata in bulk
        metadata_list = self.metadata_manager.get_track_metadata_batch(track_ids)
        
        # Create track info list with signatures
        track_info = []
        for i, idx in enumerate(indices):
            metadata = metadata_list[i]
            artist = metadata.get('artist_name', 'Unknown').lower()
            title = metadata.get('track_name', 'Unknown').lower()
            
            # Extract core title (before any modifiers)
            core_title = title.split('(')[0].split('-')[0].split('[')[0].strip()
            
            # Extract year (first 4 characters if available)
            release_year = self._get_release_year(metadata)
            
            # Create unique signature
            signature = f"{artist}|{core_title}|{release_year}"
            
            track_info.append({
                'index': idx,
                'similarity': similarities[i],
                'isrc': isrcs[i],
                'signature': signature
            })
        
        # Sort by similarity descending (prioritize higher quality matches)
        track_info.sort(key=lambda x: x['similarity'], reverse=True)
        
        seen_signatures = set()
        seen_isrcs = set()
        deduped_indices = []
        deduped_similarities = []
        
        # First pass: Strict deduplication
        for track in track_info:
            if len(deduped_indices) >= top_k:
                break
                
            # Skip duplicate ISRCs (exact recording match)
            if track['isrc'] and track['isrc'] in seen_isrcs:
                continue
                
            # Skip duplicate signatures (same song version)
            if track['signature'] in seen_signatures:
                continue
                
            deduped_indices.append(track['index'])
            deduped_similarities.append(track['similarity'])
            seen_signatures.add(track['signature'])
            seen_isrcs.add(track['isrc'])
        
        # Second pass: Fill remaining slots (only check ISRC duplicates)
        if len(deduped_indices) < top_k:
            for track in track_info:
                if len(deduped_indices) >= top_k:
                    break
                    
                # Skip if already added
                if track['index'] in deduped_indices:
                    continue
                    
                # Skip duplicate ISRCs
                if track['isrc'] and track['isrc'] in seen_isrcs:
                    continue
                    
                deduped_indices.append(track['index'])
                deduped_similarities.append(track['similarity'])
                seen_isrcs.add(track['isrc'])
        
        return deduped_indices[:top_k], deduped_similarities[:top_k]

    def _get_release_year(self, metadata: Dict) -> str:
        """Safely extract first 4 characters of release year."""
        year = metadata.get('album_release_year', '')
        return year[:4] if year and len(year) >= 4 else ''

    def _apply_secondary_sort(self, results, with_metadata):
        """Apply secondary sort by popularity to break similarity ties"""
        def get_popularity(item):
            if with_metadata:
                _, similarity, metadata = item
                return metadata.get('popularity', 0) if metadata else 0
            else:
                return 0  # No metadata available
        
        # First sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Group by similarity and sort each group by popularity
        grouped = {}
        for item in results:
            similarity = item[1]
            if similarity not in grouped:
                grouped[similarity] = []
            grouped[similarity].append(item)
        
        # Sort each group by popularity (descending)
        sorted_results = []
        for similarity, group in grouped.items():
            group.sort(key=get_popularity, reverse=True)
            sorted_results.extend(group)
        
        return sorted_results

    def _validate_implementation_parity(self):
        """Validate CPU and GPU implementations produce identical results"""
        test_vector = np.random.rand(32).astype(np.float32)
        test_vectors = np.random.rand(1000, 32).astype(np.float32)
        test_masks = np.random.randint(0, 2**32, size=1000, dtype=np.uint32)
        
        # CPU results
        cpu_ops = VectorOps(algorithm=self.algorithm)
        cpu_results = cpu_ops.compute_similarity(test_vector, test_vectors, test_masks)
        
        # GPU results
        if torch.cuda.is_available():
            gpu_ops = VectorOpsGPU()
            gpu_tensor = torch.tensor(test_vectors, device='cuda')
            gpu_masks = torch.tensor(test_masks, device='cuda')
            
            # Call the correct GPU method based on current algorithm
            if self.algorithm == 'cosine':
                gpu_results = gpu_ops.masked_weighted_cosine_similarity(test_vector, gpu_tensor, gpu_masks)
            elif self.algorithm == 'cosine-euclidean':
                gpu_results = gpu_ops.masked_weighted_cosine_euclidean_similarity(test_vector, gpu_tensor, gpu_masks)
            elif self.algorithm == 'euclidean':
                gpu_results = gpu_ops.masked_euclidean_similarity(test_vector, gpu_tensor, gpu_masks)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # Compare results
            if not np.allclose(cpu_results, gpu_results, atol=1e-5):
                max_diff = np.max(np.abs(cpu_results - gpu_results))
                raise ValueError(
                    f"CPU/GPU implementation mismatch! Max diff: {max_diff:.6f}"
                )
        
        return True
    
    def find_similar_to_track(
        self,
        track_id: str,
        top_k: int = 10,
        search_mode: str = "sequential",
        with_metadata: bool = True,
        deduplicate: Optional[bool] = None,
        **kwargs
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Find tracks similar to a given track ID.
        
        Args:
            track_id: Spotify track ID
            top_k: Number of results to return
            search_mode: Search algorithm to use
            with_metadata: Whether to include metadata
            deduplicate: Whether to deduplicate results (None = use config)
            **kwargs: Additional search parameters
            
        Returns:
            List of (track_id, similarity, metadata) tuples
        """
        # Build vector for the track
        vector, _ = build_canonical_vector(track_id)
        
        if vector is None:
            raise ValueError(f"Could not build vector for track: {track_id}")
        
        return self.search(
            vector,
            search_mode=search_mode,
            top_k=top_k,
            with_metadata=with_metadata,
            deduplicate=deduplicate,
            **kwargs
        )
    
    def close(self):
        """Clean up resources."""
        self.metadata_manager.close()

    def run_performance_test(self, query_vector, num_vectors, start_idx=0, show_progress=True):
        """Run a performance test with CUDA-synchronized timing"""
        # Warm-up run (suppress progress)
        self.search(
            query_vector,
            search_mode="sequential",
            top_k=10,
            with_metadata=False,
            start_idx=start_idx,
            max_vectors=min(100_000, num_vectors),
            show_progress=False
        )
        
        # Create CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Synchronize before starting
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Start timing
        start_time = time.time()
        if torch.cuda.is_available():
            start_event.record()
        
        # Run search with progress control
        results = self.search(
            query_vector,
            search_mode="sequential",
            top_k=10,
            with_metadata=False,
            start_idx=start_idx,
            max_vectors=num_vectors,
            show_progress=show_progress
        )
        
        # End timing
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed = time.time() - start_time
        
        speed = num_vectors / elapsed if elapsed > 0 else 0
        
        return {
            'vectors': num_vectors,
            'time': elapsed,
            'speed': speed
        }

    @classmethod
    def clear_benchmark_cache(cls):
        """Clear the benchmark results cache."""
        cls._benchmark_results = None

def find_similar_tracks(
    track_id: str,
    top_k: int = 10,
    search_mode: str = "sequential",
    deduplicate: Optional[bool] = None,
    **kwargs
) -> List:
    """
    Simplified API for finding similar tracks.
    
    Args:
        track_id: Spotify track ID
        top_k: Number of results
        search_mode: Search algorithm
        deduplicate: Whether to deduplicate results (None = use config)
        **kwargs: Additional parameters
        
    Returns:
        List of results
    """
    orchestrator = SearchOrchestrator()
    results = orchestrator.find_similar_to_track(
        track_id,
        top_k=top_k,
        search_mode=search_mode,
        deduplicate=deduplicate,
        **kwargs
    )
    orchestrator.close()
    return results
