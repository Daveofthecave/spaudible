# core/similarity_engine/orchestrator.py
import numpy as np
import time
import torch
from config import PathConfig, VRAM_SAFETY_FACTOR
from core.vectorization.canonical_track_resolver import build_canonical_vector
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable
from .chunk_size_optimizer import ChunkSizeOptimizer
from .index_manager import IndexManager
from .metadata_service import MetadataManager
from .vector_comparer import ChunkedSearch
from .vector_io import VectorReader
from .vector_io_gpu import VectorReaderGPU
from .vector_math import VectorOps
from core.utilities.gpu_utils import get_gpu_info, recommend_max_batch_size

class SearchOrchestrator:
    """High-level coordinator for similarity search operations."""
    
    # Class-level cache for benchmark results
    _benchmark_results = None
    
    def __init__(self,
                vectors_path: Optional[str] = None,
                index_path: Optional[str] = None,
                metadata_db: Optional[str] = None,
                chunk_size: int = 100_000_000,
                use_gpu: bool = True,
                max_gpu_mb: int = 2048,  # originally 2048
                skip_cpu_benchmark: bool = False,
                skip_gpu_benchmark: bool = False,
                skip_benchmark: bool = False,  # Deprecated but kept for backward compatibility
                **kwargs):
        """
        Initialize the search orchestrator.
        
        Args:
            vectors_path: Path to track_vectors.bin (default: from PathConfig)
            index_path: Path to track_index.bin (default: from PathConfig)
            metadata_db: Path to metadata database
            chunk_size: Chunk size for vector processing
            use_gpu: Whether to use GPU acceleration
            max_gpu_mb: Maximum GPU memory to use (MB)
            skip_cpu_benchmark: Skip CPU benchmarking
            skip_gpu_benchmark: Skip GPU benchmarking
            skip_benchmark: Deprecated - skip both CPU and GPU benchmarking
            **kwargs: Additional keyword arguments
        """
        # Set benchmark skip flags
        self.skip_cpu_benchmark = skip_cpu_benchmark or skip_benchmark
        self.skip_gpu_benchmark = skip_gpu_benchmark or skip_benchmark
        
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu
        self.max_gpu_mb = max_gpu_mb

        # Validate GPU availability
        self._is_gpu_available(verbose=True)

        # Set default paths if not provided
        vectors_path = vectors_path or str(PathConfig.get_vector_file())
        index_path = index_path or str(PathConfig.get_index_file())

        # Initialize components
        self.vector_reader = self._init_vector_reader(vectors_path)
        self.vector_ops = VectorOps()
        self.index_manager = IndexManager(index_path)
        self.metadata_manager = MetadataManager(metadata_db)
        
        # Get max batch size from vector reader
        max_batch_size = None
        if hasattr(self.vector_reader, 'get_max_batch_size'):
            max_batch_size = self.vector_reader.get_max_batch_size()

        # Run auto-benchmark if not skipped and not done yet
        if SearchOrchestrator._benchmark_results is None:
            if not (self.skip_cpu_benchmark and self.skip_gpu_benchmark):
                SearchOrchestrator._benchmark_results = self.run_auto_benchmark()
        
        # Apply benchmark results if available
        if SearchOrchestrator._benchmark_results:
            self.use_gpu = (SearchOrchestrator._benchmark_results['recommended_device'] == 'gpu')
            self.chunk_size = SearchOrchestrator._benchmark_results['optimal_chunk_size']
            print(f"  🚀 Using {self.use_gpu and 'GPU' or 'CPU'} acceleration with chunk size {self.chunk_size:,}")
        
        # For CPU mode: Initialize optimizer if not set by benchmark
        if not self.use_gpu and not hasattr(self, 'chunk_size_optimizer'):
            self.chunk_size_optimizer = ChunkSizeOptimizer(self.vector_reader)
            self.chunk_size = self.chunk_size_optimizer.optimize()
        
        # Pass self.use_gpu and max_batch_size to ChunkedSearch
        self.chunked_search = ChunkedSearch(
            self.chunk_size, 
            use_gpu=self.use_gpu,
            max_batch_size=max_batch_size
        )
        
        self.total_vectors = self.vector_reader.get_total_vectors()

    def run_auto_benchmark(self):
        """Run auto-benchmark with optimized test sizes"""
        print("   Running auto-benchmark...")
        results = {
            'cpu_speed': 0,
            'gpu_speed': 0,
            'recommended_device': 'cpu',
            'optimal_chunk_size': 100_000
        }
        
        # Create test vector
        test_vector = np.random.rand(32).astype(np.float32)
        
        # Benchmark CPU with specific chunk sizes
        if not self.skip_cpu_benchmark:
            print("   Benchmarking CPU with optimized chunk sizes...")
            cpu_orchestrator = SearchOrchestrator(
                skip_cpu_benchmark=True,
                skip_gpu_benchmark=True,
                use_gpu=False
            )
            
            # Test specific chunk sizes on first 500K vectors
            chunk_sizes = [5_000, 10_000, 15_000, 20_000, 30_000, 50_000, 100_000, 200_000, 500_000]
            cpu_speeds = []
            
            for chunk_size in chunk_sizes:
                cpu_orchestrator.chunk_size = chunk_size
                cpu_orchestrator.chunked_search = ChunkedSearch(chunk_size, use_gpu=False)
                
                # Run test on first 500K vectors - suppress progress bars
                result = cpu_orchestrator.run_performance_test(test_vector, 500_000, show_progress=False)
                speed = result['speed']
                cpu_speeds.append((chunk_size, speed))
                print(f"      Chunk {chunk_size:6,}: {speed/1e6:.2f}M vec/sec")
            
            # Find fastest chunk size
            best_chunk, best_speed = max(cpu_speeds, key=lambda x: x[1])
            results['cpu_speed'] = best_speed
            results['optimal_cpu_chunk_size'] = best_chunk
            print(f"      Optimal CPU chunk: {best_chunk:,} ({best_speed/1e6:.2f}M vec/sec)")
            
            cpu_orchestrator.close()
        
        # Benchmark GPU with max batch size
        if torch.cuda.is_available() and not self.skip_gpu_benchmark:
            print("   Benchmarking GPU with max batch size...")
            gpu_orchestrator = SearchOrchestrator(
                skip_cpu_benchmark=True,
                skip_gpu_benchmark=True,
                use_gpu=True
            )
            
            # Get max batch size from vector reader
            max_batch = gpu_orchestrator.vector_reader.get_max_batch_size()
            print(f"      Max GPU batch: {max_batch:,} vectors")
            
            # Run test with max batch size - show progress bars
            gpu_result = gpu_orchestrator.run_performance_test(test_vector, max_batch, show_progress=True)
            results['gpu_speed'] = gpu_result['speed']
            print(f"      GPU speed: {gpu_result['speed']/1e6:.2f}M vec/sec")
            
            gpu_orchestrator.close()
        
        # Determine optimal configuration
        if results.get('gpu_speed', 0) > results.get('cpu_speed', 0) * 1.1:
            results['recommended_device'] = 'gpu'
            results['optimal_chunk_size'] = max_batch  # Use max batch as chunk size
        else:
            results['recommended_device'] = 'cpu'
            results['optimal_chunk_size'] = results.get('optimal_cpu_chunk_size', 100_000)
        
        print(f"✅ Auto-benchmark complete: Using {results['recommended_device'].upper()} "
            f"with chunk size {results['optimal_chunk_size']:,}")
        return results

    def _is_gpu_available(self, verbose: bool = False):
        """Validate GPU availability and adjust chunk size"""
        if self.use_gpu and torch.cuda.is_available():
            if verbose: 
                print("✅ Using GPU")
            return True
        else:
            if verbose: 
                print("ℹ️  Using CPU...")
            self.use_gpu = False
            # Reduce chunk size for CPU mode
            self.chunk_size = min(self.chunk_size, 200_000)
            return False

    def _init_vector_reader(self, vectors_path: str):
        """Initialize vector reader with GPU support if available"""
        if self._is_gpu_available():
            try:
                return VectorReaderGPU(vectors_path, max_gpu_mb=self.max_gpu_mb)
            except ImportError:
                return VectorReader(vectors_path)
        else:
            return VectorReader(vectors_path)        
    
    def _vector_source(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Wrapper for vector reader."""
        return self.vector_reader.read_chunk(start_idx, num_vectors)
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        search_mode: str = "sequential",
        top_k: int = 10,
        with_metadata: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Find similar tracks to a query vector.
        
        Args:
            query_vector: 32D vector to search for
            search_mode: One of "sequential", "random", "progressive"
            top_k: Number of results to return
            with_metadata: Whether to include metadata
            **kwargs: Additional search parameters
            
        Returns:
            List of (track_id, similarity, metadata) tuples
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
                self.total_vectors,
                self.vector_ops,
                top_k=top_k,
                show_progress=show_progress,
                **kwargs
            )
        elif search_mode == "random":
            indices, similarities = self.chunked_search.random_chunk_search(
                query_np,
                self._vector_source,
                self.total_vectors,
                self.vector_ops,
                num_chunks=100,
                top_k=top_k,
                show_progress=show_progress,
                **kwargs
            )
        elif search_mode == "progressive":
            indices, similarities = self.chunked_search.progressive_search(
                query_np,
                self._vector_source,
                self.total_vectors,
                self.vector_ops,
                min_chunks=1,
                max_chunks=100,
                quality_threshold=0.95,
                top_k=top_k,
                show_progress=show_progress,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
        
        # Convert indices to track IDs
        track_ids = self.index_manager.get_track_ids_batch(indices)
        
        # Get metadata if requested
        if with_metadata:
            metadata_list = self.metadata_manager.get_track_metadata_batch(track_ids)
            results = list(zip(track_ids, similarities, metadata_list))
        else:
            results = list(zip(track_ids, similarities))
        
        return results
    
    def find_similar_to_track(
        self,
        track_id: str,
        top_k: int = 10,
        search_mode: str = "sequential",
        with_metadata: bool = True,
        **kwargs
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Find tracks similar to a given track ID.
        
        Args:
            track_id: Spotify track ID
            top_k: Number of results to return
            search_mode: Search algorithm to use
            with_metadata: Whether to include metadata
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

def find_similar_tracks(
    track_id: str,
    top_k: int = 10,
    search_mode: str = "sequential",
    **kwargs
) -> List:
    """
    Simplified API for finding similar tracks.
    
    Args:
        track_id: Spotify track ID
        top_k: Number of results
        search_mode: Search algorithm
        **kwargs: Additional parameters
        
    Returns:
        List of results
    """
    orchestrator = SearchOrchestrator()
    results = orchestrator.find_similar_to_track(
        track_id,
        top_k=top_k,
        search_mode=search_mode,
        **kwargs
    )
    orchestrator.close()
    return results
    