# core/similarity_engine/orchestrator.py
"""
SearchOrchestrator coordinates similarity search operations using the unified vector format.
Supports only sequential search for maximum performance.
"""
import numpy as np
import time
import torch
import mmap
import gc
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable
from config import PathConfig, VRAM_SAFETY_FACTOR
from .index_manager import IndexManager
from .metadata_service import MetadataManager
from .vector_comparer import ChunkedSearch
from .vector_io import VectorReader
from .vector_io_gpu import VectorReaderGPU
from .vector_math import VectorOps
from .vector_math_gpu import VectorOpsGPU
from core.utilities.gpu_utils import get_gpu_info
from core.utilities.config_manager import config_manager
from core.vectorization.canonical_track_resolver import build_canonical_vector

M = 6 # Multiplier for top_k to pull k * M song results

class SearchOrchestrator:
    """High-performance similarity search coordinator for unified vector format."""
    
    _benchmark_results = None
    
    def __init__(self, 
                 vectors_path: Optional[str] = None, 
                 index_path: Optional[str] = None,
                 metadata_db: Optional[str] = None,
                 chunk_size: int = 100_000_000,
                 use_gpu: bool = True,
                 force_cpu: bool = False,
                 force_gpu: bool = False,
                 skip_benchmark: bool = False,
                 skip_validation: bool = False,
                 **kwargs):
        """
        Initialize orchestrator for unified vector format.
        
        Args:
            vectors_path: Path to unified track_vectors.bin
            index_path: Path to sorted track_index.bin
            metadata_db: Path to Spotify metadata database
            chunk_size: Base chunk size for processing
            use_gpu: Whether to prefer GPU acceleration
            force_cpu: Override to force CPU mode
            force_gpu: Override to force GPU mode
            skip_benchmark: Skip auto-benchmark for lightweight operations
            skip_validation: Skip validation during benchmark
        """
        # Start with provided settings
        self.use_gpu = use_gpu
        
        # Initialize chunk_size to safe default before any conditional logic
        self.chunk_size = 200_000
        
        # Get and apply force settings from config
        self.force_cpu = force_cpu or config_manager.get_force_cpu()
        self.force_gpu = force_gpu or config_manager.get_force_gpu()
        
        # Apply force settings before initializing any components
        if self.force_cpu:
            self.use_gpu = False
            # print("ℹ️  CPU mode forced by configuration")
        elif self.force_gpu:
            self.use_gpu = True
            # print("ℹ️  GPU mode forced by configuration")
        
        # Determine device early
        gpu_available = torch.cuda.is_available()
        if self.use_gpu and gpu_available:
            self.device = "cuda"
        else:
            self.device = "cpu"
            self.use_gpu = False  # Override if GPU not available
        
        # Path resolution
        vectors_path = vectors_path or str(PathConfig.get_vector_file())
        index_path = index_path or str(PathConfig.get_index_file())
        
        # Initialize components with correct GPU/CPU setting
        self.vector_reader = self._init_vector_reader(vectors_path)
        self.index_manager = IndexManager(index_path)
        self.index_manager._vector_reader = self.vector_reader
        self.metadata_manager = MetadataManager(metadata_db)
        
        # Initialize vector operations
        self.algorithm = config_manager.get_algorithm()
        self.vector_ops = VectorOps(algorithm=self.algorithm)
        self.vector_ops.set_user_weights(config_manager.get_weights())
        
        # Region filtering from config
        self.region_strength = config_manager.get_region_strength()
        
        # Determine optimal chunk size and device
        if not self.force_cpu and not self.force_gpu and not skip_benchmark:
            # Clear memory cache when entering auto mode
            SearchOrchestrator._benchmark_results = None
            
            # Check for cached benchmark result (from config file)
            config_result = config_manager.get_benchmark_result()
            if config_result is not None:
                SearchOrchestrator._benchmark_results = config_result
            else:
                # Run benchmark and save to both memory and config
                SearchOrchestrator._benchmark_results = self.run_auto_benchmark()
                config_manager.set_benchmark_result(SearchOrchestrator._benchmark_results)

            if SearchOrchestrator._benchmark_results:
                self.use_gpu = (
                    SearchOrchestrator._benchmark_results['recommended_device'] == 'gpu'
                )
                
                # Re-determine device after benchmark
                if self.use_gpu and gpu_available:
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                    self.use_gpu = False
        else:
            if self.force_cpu:
                # Skip optimizer; use adaptive resizing during search
                self.chunk_size = 200_000
            elif self.force_gpu:
                self.chunk_size = self.vector_reader.get_max_batch_size()
        
        # Initialize GPU operations if needed
        if self.use_gpu and self.device == "cuda":
            self._init_gpu_ops()
        
        # Initialize chunked search
        self.chunked_search = ChunkedSearch(
            chunk_size=self.chunk_size,
            use_gpu=self.use_gpu,
            vector_ops=self.vector_ops
        )
        
        self.total_vectors = self.vector_reader.get_total_vectors()
        
        # Skip validation during benchmark to avoid dtype errors
        if not skip_validation and not self.force_cpu and not self.force_gpu and hasattr(self, 'gpu_ops') and self.gpu_ops is not None:
            try:
                self._validate_implementation_parity()
            except Exception as e:
                print(f"  ⚠️  Implementation validation failed: {e}")

    def _init_vector_reader(self, vectors_path: str):
        """Initialize appropriate vector reader (CPU or GPU)."""
        # Use dedicated GPU reader for GPU mode
        if self.use_gpu and torch.cuda.is_available():
            try:
                from .vector_io_gpu import VectorReaderGPU
                return VectorReaderGPU(vectors_path)
            except ImportError:
                print("  ⚠️  VectorReaderGPU not available, falling back to CPU")
                self.use_gpu = False
                self.device = "cpu"
        
        # Use new unified VectorReader for CPU mode (fast PyTorch implementation)
        device = self.device
        return VectorReader(vectors_path, device=device)

    def _init_gpu_ops(self):
        """Initialize GPU operations."""
        try:
            free_vram = torch.cuda.mem_get_info()[0]
            if free_vram < 500_000_000:  # 500MB minimum
                print("⚠️ Low VRAM available, disabling GPU acceleration")
                self.use_gpu = False
                self.device = "cpu"
                return False
            else:
                self.gpu_ops = VectorOpsGPU(device=self.device)
                self.gpu_ops.set_user_weights(config_manager.get_weights())
                return True
        except Exception as e:
            print(f"⚠️  GPU initialization failed: {e}")
            self.gpu_ops = None
            return False

    def _optimize_cpu_chunk(self) -> int:
        """Optimize chunk size for CPU processing."""
        optimizer = ChunkSizeOptimizer(self.vector_reader)
        return optimizer.optimize()
    
    def run_auto_benchmark(self) -> dict:
        """
        Run CPU vs GPU benchmark and return best configuration.
        OPTIMIZED: Uses small test vectors and lets CPU adaptive resizer do its job.
        """
        print("   Running auto-benchmark...")
        results = {
            'cpu_speed': 0,
            'gpu_speed': 0,
            'recommended_device': 'cpu',
            'optimal_chunk_size': 200_000  # Default for CPU
        }
        
        # Skip GPU benchmark if forced CPU
        if self.force_cpu:
            results['cpu_speed'] = self._benchmark_device('cpu')
            results['optimal_chunk_size'] = 200_000
            print(f"   ✅ Using CPU ({results['cpu_speed']/1e6:.1f}M vec/sec)")
            return results
        
        # Skip GPU benchmark if forced GPU or no GPU available
        if not self.force_gpu and not torch.cuda.is_available():
            results['cpu_speed'] = self._benchmark_device('cpu')
            results['optimal_chunk_size'] = 200_000
            print(f"   ✅ Using CPU ({results['cpu_speed']/1e6:.1f}M vec/sec)")
            return results
        
        # Benchmark both CPU and GPU at the same time
        if torch.cuda.is_available():
            # Use a small test size for quick benchmark
            test_vectors_limit = 1_000_000  # Small subset for speed
            
            # CPU benchmark - let adaptive resizer handle chunk sizing automatically
            print("     Testing CPU performance...")
            cpu_speed = self._benchmark_device('cpu', test_vectors_limit)
            results['cpu_speed'] = cpu_speed
            print(f"       CPU: {results['cpu_speed']/1e6:.1f}M vec/sec")
            
            # GPU benchmark - use its max batch size
            print("     Testing GPU performance...")
            gpu_chunk = self.vector_reader.get_max_batch_size()
            gpu_speed = self._benchmark_device('gpu', test_vectors_limit)
            results['gpu_speed'] = gpu_speed
            print(f"       GPU: {results['gpu_speed']/1e6:.1f}M vec/sec")
            
            # Choose faster device (with 10% GPU preference)
            if gpu_speed > cpu_speed * 1.1:
                results['recommended_device'] = 'gpu'
                results['optimal_chunk_size'] = gpu_chunk
                print(f"   Using GPU...")
            else:
                results['recommended_device'] = 'cpu'
                # For CPU, we don't need to tune chunk size - adaptive resizer will handle it
                results['optimal_chunk_size'] = 200_000  # Reasonable default
                print(f"   Using CPU...")
        
        return results

    def _benchmark_device(self, device: str, test_vectors: int = 500_000) -> float:
        """
        Benchmark a single device with small test size for speed.
        Uses skip_benchmark=True to avoid recursion.
        """
        if device == 'cpu':
            orchestrator = SearchOrchestrator(
                use_gpu=False, 
                force_cpu=True, 
                skip_benchmark=True,
                skip_validation=True
            )
        else:
            orchestrator = SearchOrchestrator(
                use_gpu=True, 
                force_gpu=True, 
                skip_benchmark=True,
                skip_validation=True
            )
        
        test_vector = np.random.rand(32).astype(np.float32)
        
        # Warm-up with small subset
        orchestrator.search(
            test_vector, 
            top_k=10, 
            show_progress=False,
            max_vectors=test_vectors // 10
        )
        
        # Timed run with small subset
        start = time.time()
        orchestrator.search(
            test_vector, 
            top_k=10, 
            show_progress=False,
            max_vectors=test_vectors
        )
        elapsed = time.time() - start
        
        orchestrator.close()
        return test_vectors / elapsed if elapsed > 0 else 0

    def _vector_source(self, start_idx: int, num_vectors: int):
        """Read vector chunk from track_vectors.bin."""
        return self.vector_reader.read_chunk(start_idx, num_vectors)

    def _mask_source(self, start_idx: int, num_vectors: int):
        """Read mask chunk from track_vectors.bin."""
        return self.vector_reader.read_masks(start_idx, num_vectors)

    def _region_source(self, start_idx: int, num_vectors: int):
        """Read region chunk from track_vectors.bin."""
        return self.vector_reader.read_regions(start_idx, num_vectors)
    
    def search(self,
               query_vector: Union[List[float], np.ndarray],
               top_k: int = 10,
               with_metadata: bool = True,
               show_progress: bool = True,
               deduplicate: Optional[bool] = None,
               query_track_id: Optional[str] = None,
               max_vectors: Optional[int] = None,
               **kwargs) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Perform sequential similarity search.
        
        Args:
            query_vector: 32D query vector
            top_k: Number of results to return
            with_metadata: Include track metadata
            show_progress: Show progress bar
            deduplicate: Remove duplicate tracks
            query_track_id: Track ID for region filtering
            max_vectors: Optional limit on vectors to scan (for benchmarks)
            
        Returns:
            List of (track_id, similarity, metadata) tuples
        """
        # Convert query to numpy array
        if isinstance(query_vector, list):
            query_np = self.vector_ops.to_numpy_array(query_vector)
        else:
            query_np = query_vector
        
        # Determine query region if track ID provided
        query_region = -1
        if query_track_id and self.region_strength > 0.0:
            query_region = self._get_query_region(query_track_id)
        
        # Execute sequential scan
        indices, similarities = self.chunked_search.sequential_scan(
            query_np,
            self._vector_source,
            self._mask_source,
            self._region_source,
            self.total_vectors,
            self.vector_ops,
            top_k=top_k * M,
            max_vectors=max_vectors,
            show_progress=show_progress,
            query_region=query_region,
            region_strength=self.region_strength
        )
        
        # Apply deduplication
        if deduplicate is None:
            deduplicate = config_manager.get_deduplicate()
        
        if deduplicate:
            indices, similarities = self._advanced_deduplication(
                indices, similarities, top_k
            )
        
        # Convert vector indices to track IDs
        track_ids = self.index_manager.get_track_ids_from_vector_indices(indices)
        
        # Fetch metadata if requested
        if with_metadata:
            metadata_list = self.metadata_manager.get_track_metadata_batch(track_ids)
            results = list(zip(track_ids, similarities, metadata_list))
            # Apply secondary sort to break ties
            results = self._apply_secondary_sort(results, with_metadata=True)
        else:
            results = list(zip(track_ids, similarities))
        
        return results[:top_k]
    
    def _get_query_region(self, track_id: str) -> int:
        """Get region code for query track ID."""
        try:
            vector_index = self.index_manager.get_index_from_track_id(track_id)
            if vector_index is None:
                return -1
            
            # Read region directly from vector record using context manager
            with open(PathConfig.get_vector_file(), 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Region is at byte 69 of each 104-byte record, after 16-byte header
                    offset = 16 + vector_index * 104 + 69
                    region_byte = mm[offset]
                    return region_byte
            
        except Exception as e:
            print(f"  ⚠️  Error reading region for {track_id}: {e}")
            return -1
    
    def _apply_secondary_sort(self, results: List, with_metadata: bool) -> List:
        """Apply secondary sort by popularity to break similarity ties"""
        def get_popularity(item):
            if with_metadata:
                _, similarity, metadata = item
                return metadata.get('popularity', 0) if metadata else 0
            else:
                return 0
        
        # First sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Group by similarity and sort each group by popularity
        grouped = {}
        for item in results:
            similarity = round(item[1], 6)
            if similarity not in grouped:
                grouped[similarity] = []
            grouped[similarity].append(item)
        
        # Reconstruct with popularity sort within each similarity group
        sorted_results = []
        for similarity in sorted(grouped.keys(), reverse=True):
            group = grouped[similarity]
            group.sort(key=get_popularity, reverse=True)
            sorted_results.extend(group)
        
        return sorted_results
    
    def _advanced_deduplication(self, indices: List[int], similarities: List[float], 
                            top_k: int) -> Tuple[List[int], List[float]]:
        """
        Single-pass strict deduplication: Same signature = same song.
        """
        # Get ISRCs and track IDs
        isrcs = self.vector_reader.get_isrcs_batch(indices[:top_k * M])
        track_ids = self.vector_reader.get_track_ids_batch(indices[:top_k * M])
        
        # Fetch metadata batch
        metadata_list = self.metadata_manager.get_track_metadata_batch(track_ids)
        
        # Build track info with normalized signatures
        track_info = []
        for i, idx in enumerate(indices[:top_k * M]):
            metadata = metadata_list[i]
            artist = metadata.get('artist_name', 'Unknown').lower()
            title = metadata.get('track_name', 'Unknown').lower()
            
            # Extract and normalize core title
            core_title = title.split('(')[0].split('-')[0].split('[')[0].strip()
            
            # Aggressive whitespace normalization
            import re
            core_title = re.sub(r'\s+', ' ', core_title).strip()
            artist = re.sub(r'\s+', ' ', artist).strip()
            
            signature = f"{artist}|{core_title}"
            
            track_info.append({
                'index': idx,
                'similarity': similarities[i],
                'isrc': isrcs[i],
                'signature': signature
            })
        
        # Sort by similarity
        track_info.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Single pass: Strict on both ISRC and signature
        seen_signatures = set()
        seen_isrcs = set()
        deduped_indices = []
        deduped_similarities = []
        
        for track in track_info:
            if len(deduped_indices) >= top_k:
                break
            
            # Skip if signature matches (same song)
            if track['signature'] in seen_signatures:
                continue
            
            # Skip if ISRC matches (same recording)
            if track['isrc'] and track['isrc'] in seen_isrcs:
                continue
            
            deduped_indices.append(track['index'])
            deduped_similarities.append(track['similarity'])
            seen_signatures.add(track['signature'])
            seen_isrcs.add(track['isrc'])
        
        return deduped_indices[:top_k], deduped_similarities[:top_k]

    def _validate_implementation_parity(self):
        """Ensure CPU and GPU produce identical results."""
        test_vector = np.random.rand(32).astype(np.float32)
        test_vectors = np.random.rand(1000, 32).astype(np.float32)
        # Use uint32 for masks since they're stored as unsigned in the file
        test_masks = np.random.randint(0, 2**32, size=1000, dtype=np.uint32)
        
        # CPU
        cpu_results = self.vector_ops.compute_similarity(test_vector, test_vectors, test_masks)
        
        # GPU
        if torch.cuda.is_available() and hasattr(self, 'gpu_ops') and self.gpu_ops is not None:
            gpu_tensor = torch.tensor(test_vectors, device='cuda')
            gpu_masks = torch.tensor(test_masks.astype(np.int32), device='cuda')
            
            if self.algorithm == 'cosine':
                gpu_results = self.gpu_ops.masked_weighted_cosine_similarity(test_vector, gpu_tensor, gpu_masks)
            elif self.algorithm == 'cosine-euclidean':
                gpu_results = self.gpu_ops.masked_weighted_cosine_euclidean_similarity(test_vector, gpu_tensor, gpu_masks)
            else:  # cosine-euclidean (default)
                gpu_results = self.gpu_ops.masked_euclidean_similarity(test_vector, gpu_tensor, gpu_masks)
            
            if not np.allclose(cpu_results, gpu_results.cpu().numpy(), atol=1e-5):
                raise ValueError("CPU/GPU implementation mismatch!")

    def close(self):
        """Clean up resources."""
        self.metadata_manager.close()
        
        # Clean up vector reader if it exists
        if hasattr(self, 'vector_reader'):
            # Force garbage collection to clear any cached array views
            gc.collect()
            try:
                self.vector_reader.close()
            except BufferError:
                # Ignore "cannot close exported pointers exist" error; it's harmless.
                # This happens when NumPy arrays from mmap are still referenced
                # The OS will clean up the mmap when the process exits anyway
                pass
    
    @classmethod
    def clear_benchmark_cache(cls):
        """Clear benchmark results cache."""
        cls._benchmark_results = None
        config_manager.clear_benchmark_result()


def find_similar_tracks(
    track_id: str,
    top_k: int = 10,
    deduplicate: Optional[bool] = None,
    **kwargs
) -> List:
    """Simplified API for finding similar tracks."""
    orchestrator = SearchOrchestrator()
    results = orchestrator.find_similar_to_track(
        track_id,
        top_k=top_k,
        deduplicate=deduplicate,
        **kwargs
    )
    orchestrator.close()
    return results
