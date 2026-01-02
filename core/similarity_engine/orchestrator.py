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
                 max_gpu_mb: int = 2048, # originally 2048
                 skip_benchmark: bool = False,
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
            skip_benchmark: Skip auto-benchmarking (for internal use)
        """    
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu
        self.max_gpu_mb = max_gpu_mb
        self.skip_benchmark = skip_benchmark

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
        if not self.skip_benchmark and SearchOrchestrator._benchmark_results is None:
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
            max_batch_size=max_batch_size  # Pass max batch size here
        )
        
        self.total_vectors = self.vector_reader.get_total_vectors()

    def run_auto_benchmark(self, test_size=10_000_000):
        """Run a quick benchmark to determine the optimal configuration."""
        print("  🔧 Running auto-benchmark...")
        results = {
            'cpu_speed': 0,
            'gpu_speed': 0,
            'recommended_device': 'cpu',
            'optimal_chunk_size': 100_000
        }
        
        # Create test vector
        test_vector = np.random.rand(32).astype(np.float32)
        
        # Benchmark CPU
        print("    Benchmarking CPU...")
        cpu_orchestrator = SearchOrchestrator(
            skip_benchmark=True,
            use_gpu=False
        )
        cpu_result = cpu_orchestrator.run_performance_test(test_vector, test_size)
        cpu_orchestrator.close()
        results['cpu_speed'] = cpu_result['speed']
        print(f"      CPU speed: {cpu_result['speed']/1e6:.2f}M vec/sec")
        
        # Benchmark GPU if available
        if torch.cuda.is_available():
            print("    Benchmarking GPU...")
            # Use larger chunk size for GPU benchmark
            gpu_orchestrator = SearchOrchestrator(
                skip_benchmark=True,
                use_gpu=True,
                chunk_size=100_000_000
            )
            gpu_result = gpu_orchestrator.run_performance_test(test_vector, test_size)
            gpu_orchestrator.close()
            results['gpu_speed'] = gpu_result['speed']
            print(f"      GPU speed: {gpu_result['speed']/1e6:.2f}M vec/sec")
        
        # Determine optimal configuration
        if results['gpu_speed'] > results['cpu_speed'] * 1.1:  # GPU must be at least 10% faster
            results['recommended_device'] = 'gpu'
            # Use the maximum chunk size that fits in GPU memory
            gpu_info = get_gpu_info()
            if gpu_info:
                free_vram = gpu_info[0]['free_vram']
                # We want a chunk size that uses about X% of free VRAM
                bytes_per_vector = 32 * 4  # 32 floats * 4 bytes
                vectors_per_chunk = int((free_vram * VRAM_SAFETY_FACTOR) // bytes_per_vector)
                results['optimal_chunk_size'] = vectors_per_chunk
            else:
                results['optimal_chunk_size'] = 30_000_000
        else:
            # Optimize CPU chunk size
            optimizer = ChunkSizeOptimizer(self.vector_reader)
            results['optimal_chunk_size'] = optimizer.optimize()
        
        print(f"  ✅ Auto-benchmark complete: Using {results['recommended_device'].upper()} "
              f"with chunk size {results['optimal_chunk_size']:,}")
        return results

    def _is_gpu_available(self, verbose: bool = False):
        """Validate GPU availability and adjust chunk size"""
        if self.use_gpu and torch.cuda.is_available():
            if verbose: 
                print("  ✅ GPU acceleration available")
            return True
        else:
            if verbose: 
                print("⚠️  GPU acceleration unavailable; falling back on CPU...")
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

    def run_performance_test(self, query_vector, num_vectors):
        """Run a performance test with CUDA-synchronized timing"""
        # Warm-up run
        self.search(
            query_vector,
            search_mode="sequential",
            top_k=10,
            with_metadata=False,
            max_vectors=min(100_000, num_vectors)
        )
        
        # Create CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Synchronize before starting
        torch.cuda.synchronize()
        
        # Start timing
        start_event.record()
        
        # Run search
        results = self.search(
            query_vector,
            search_mode="sequential",
            top_k=10,
            with_metadata=False,
            max_vectors=num_vectors
        )
        
        # End timing
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate elapsed time in seconds
        elapsed = start_event.elapsed_time(end_event) / 1000.0
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
    