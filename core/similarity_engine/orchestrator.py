# core/similarity_engine/orchestrator.py
"""
SearchOrchestrator coordinates similarity search operations using the unified vector format.
Supports only sequential search for maximum performance.
"""
import numpy as np
import time
import torch
import struct
import mmap
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from config import PathConfig, VRAM_SAFETY_FACTOR
from .chunk_size_optimizer import ChunkSizeOptimizer
from .index_manager import IndexManager
from .metadata_service import MetadataManager
from .vector_comparer import ChunkedSearch
from .vector_io_gpu import VectorReaderGPU, RegionReaderGPU
from .vector_math import VectorOps
from core.utilities.config_manager import config_manager
from core.vectorization.canonical_track_resolver import build_canonical_vector

# Constants for unified vector format (must match vector_io_gpu.py)
VECTOR_RECORD_SIZE = 104    # Total bytes per vector record
VECTOR_HEADER_SIZE = 16     # Header size at start of file
MASK_OFFSET_IN_RECORD = 65  # 4-byte mask starts at byte 65
REGION_OFFSET_IN_RECORD = 69  # 1-byte region code at byte 69
ISRC_OFFSET_IN_RECORD = 70  # 12-byte ISRC at bytes 70-81
TRACK_ID_OFFSET_IN_RECORD = 82  # 22-byte track ID at bytes 82-103


class SearchOrchestrator:
    """High-performance similarity search coordinator for unified vector format."""
    
    # Class-level cache for benchmark results
    _benchmark_results = None
    
    def __init__(self, 
                 vectors_path: Optional[str] = None, 
                 index_path: Optional[str] = None,
                 metadata_db: Optional[str] = None,
                 chunk_size: int = 100_000_000,
                 use_gpu: bool = True,
                 force_cpu: bool = None,
                 force_gpu: bool = None,
                 **kwargs):
        """
        Initialize orchestrator for unified vector format.
        
        Args:
            vectors_path: Path to unified track_vectors.bin
            index_path: Path to sorted track_index.bin
            metadata_db: Path to Spotify metadata database
            chunk_size: Chunk size for processing (auto-optimized if not forced)
            use_gpu: Whether to prefer GPU acceleration
            force_cpu: Override to force CPU mode
            force_gpu: Override to force GPU mode
        """
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        
        # Force settings from config
        if force_cpu is None:
            force_cpu = config_manager.get_force_cpu()
        if force_gpu is None:
            force_gpu = config_manager.get_force_gpu()
        
        self.force_cpu = force_cpu
        self.force_gpu = force_gpu
        
        # Path resolution
        vectors_path = vectors_path or str(PathConfig.get_vector_file())
        index_path = index_path or str(PathConfig.get_index_file())
        
        # Initialize components
        self._init_vector_reader(vectors_path)
        self.index_manager = IndexManager(index_path)
        self.index_manager._vector_reader = self.vector_reader
        self.metadata_manager = MetadataManager(metadata_db)
        
        # Algorithm and weights
        self.algorithm = config_manager.get_algorithm()
        self.vector_ops = VectorOps(algorithm=self.algorithm)
        self.vector_ops.set_user_weights(config_manager.get_weights())
        
        # Region filtering from config
        self.region_strength = config_manager.get_region_strength()
        
        # Clear benchmark cache if settings changed
        if SearchOrchestrator._benchmark_results:
            current_cpu = config_manager.get_force_cpu()
            current_gpu = config_manager.get_force_gpu()
            
            if (current_cpu != self.force_cpu) or (current_gpu != self.force_gpu):
                SearchOrchestrator._benchmark_results = None
        
        # Auto-benchmark if not forced
        if not self.force_cpu and not self.force_gpu:
            if SearchOrchestrator._benchmark_results is None:
                SearchOrchestrator._benchmark_results = self.run_auto_benchmark()
            
            if SearchOrchestrator._benchmark_results:
                self.use_gpu = (
                    SearchOrchestrator._benchmark_results['recommended_device'] == 'gpu'
                )
                self.chunk_size = SearchOrchestrator._benchmark_results['optimal_chunk_size']
        
        # Force CPU/GPU settings
        if self.force_cpu:
            self.use_gpu = False
            self.chunk_size = self._optimize_cpu_chunk()
        elif self.force_gpu:
            self.use_gpu = True
            self.chunk_size = self.vector_reader.get_max_batch_size()
        
        # Initialize chunked search
        max_batch = self.vector_reader.get_max_batch_size() if self.use_gpu else None
        self.chunked_search = ChunkedSearch(
            self.chunk_size,
            use_gpu=self.use_gpu,
            max_batch_size=max_batch,
            vector_ops=self.vector_ops
        )
        
        self.total_vectors = self.vector_reader.get_total_vectors()
        
        # Validate CPU/GPU parity
        if not self.force_cpu and not self.force_gpu:
            try:
                self._validate_implementation_parity()
            except Exception as e:
                print(f"  ⚠️  Implementation validation failed: {e}")
    
    def _init_vector_reader(self, vectors_path: str):
        """Initialize appropriate vector reader (GPU or CPU)."""
        if self.use_gpu and torch.cuda.is_available():
            self.vector_reader = VectorReaderGPU(
                vectors_path,
                device="cuda",
                vram_scaling_factor_mb=2**8  # 256MB scaling
            )
        else:
            # Unified CPU reader to be implemented
            from .vector_io import UnifiedVectorReaderCPU
            self.vector_reader = UnifiedVectorReaderCPU(vectors_path)
    
    def _optimize_cpu_chunk(self) -> int:
        """Optimize chunk size for CPU processing."""
        optimizer = ChunkSizeOptimizer(self.vector_reader)
        return optimizer.optimize()
    
    def run_auto_benchmark(self) -> dict:
        """Run CPU vs GPU benchmark and return best configuration."""
        print("   Running auto-benchmark...")
        results = {
            'cpu_speed': 0,
            'gpu_speed': 0,
            'recommended_device': 'cpu',
            'optimal_chunk_size': 100_000,
            'cpu_chunk_size': 100_000
        }
        
        test_vector = np.random.rand(32).astype(np.float32)
        
        # Benchmark CPU
        if not self.force_gpu:
            cpu_chunk = self._optimize_cpu_chunk()
            cpu_speed = self._benchmark_device(test_vector, 'cpu', cpu_chunk)
            results['cpu_speed'] = cpu_speed
            results['cpu_chunk_size'] = cpu_chunk
            results['optimal_chunk_size'] = cpu_chunk
        
        # Benchmark GPU
        if torch.cuda.is_available() and not self.force_cpu:
            gpu_chunk = self.vector_reader.get_max_batch_size()
            gpu_speed = self._benchmark_device(test_vector, 'gpu', gpu_chunk)
            results['gpu_speed'] = gpu_speed
            
            # Choose faster device (with 10% GPU preference)
            if gpu_speed > results['cpu_speed'] * 1.1:
                results['recommended_device'] = 'gpu'
                results['optimal_chunk_size'] = gpu_chunk
            else:
                results['recommended_device'] = 'cpu'
                results['optimal_chunk_size'] = results['cpu_chunk_size']
        
        device_name = results['recommended_device'].upper()
        speed = results['gpu_speed'] or results['cpu_speed']
        print(f"   ✅ Using {device_name} ({speed/1e6:.1f}M vec/sec)")
        
        return results
    
    def _benchmark_device(self, test_vector: np.ndarray, device: str, chunk_size: int) -> float:
        """Benchmark a single device."""
        if device == 'cpu':
            orchestrator = SearchOrchestrator(use_gpu=False, skip_benchmark=True)
        else:
            orchestrator = SearchOrchestrator(use_gpu=True, skip_benchmark=True)
        
        start = time.time()
        orchestrator.search(test_vector, top_k=10, show_progress=False)
        elapsed = time.time() - start
        
        orchestrator.close()
        return 1_000_000 / elapsed if elapsed > 0 else 0
    
    def search(self,
               query_vector: Union[List[float], np.ndarray],
               top_k: int = 10,
               with_metadata: bool = True,
               show_progress: bool = True,
               deduplicate: Optional[bool] = None,
               query_track_id: Optional[str] = None,
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
            top_k=top_k * 3,  # Get extra candidates for deduplication
            show_progress=show_progress,
            query_region=query_region,
            region_strength=self.region_strength
        )
        
        # Apply deduplication
        if deduplicate is None:
            deduplicate = config_manager.get_deduplicate()
        
        if deduplicate:
            indices, similarities = self._advanced_deduplication(
                indices, 
                similarities, 
                top_k
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
    
    def _vector_source(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read vector chunk from track_vectors.bin."""
        return self.vector_reader.read_chunk(start_idx, num_vectors)
    
    def _mask_source(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read mask chunk from track_vectors.bin."""
        return self.vector_reader.read_masks(start_idx, num_vectors)

    def _region_source(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read region chunk from track_vectors.bin."""
        return self.vector_reader.read_regions(start_idx, num_vectors)
    
    def _get_query_region(self, track_id: str) -> int:
        """Get region code for query track ID."""
        try:
            vector_index = self.index_manager.get_index_from_track_id(track_id)
            if vector_index is None:
                return -1
            
            # Read region directly from vector record
            with open(PathConfig.get_vector_file(), 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                offset = VECTOR_HEADER_SIZE + vector_index * VECTOR_RECORD_SIZE + REGION_OFFSET_IN_RECORD
                region_byte = mm[offset]
                mm.close()
            
            return region_byte
        
        except Exception as e:
            print(f"  ⚠️  Error reading region for {track_id}: {e}")
            return -1
    
    def _get_release_year(self, metadata: Dict) -> str:
        """Safely extract first 4 characters of release year."""
        year = metadata.get('album_release_year', '')
        return year[:4] if year and len(year) >= 4 else ''

    def _apply_secondary_sort(self, results: List, with_metadata: bool) -> List:
        """Apply secondary sort by popularity to break similarity ties"""
        def get_popularity(item):
            if with_metadata:
                _, similarity, metadata = item
                return metadata.get('popularity', 0) if metadata else 0
            else:
                return 0
        
        # Sort by similarity descending first
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Then sort groups with same similarity by popularity
        grouped = {}
        for item in results:
            similarity = round(item[1], 6)  # Group by rounded similarity
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
        ISRC-based deduplication with two-pass filtering.
        Pass 1: Strict deduplication (ISRC + metadata signature)
        Pass 2: Fill remaining slots (ISRC only)
        """
        # Get raw data in bulk from vector file
        isrcs = self.vector_reader.get_isrcs_batch(indices[:top_k * 2])
        track_ids = self.vector_reader.get_track_ids_batch(indices[:top_k * 2])
        
        # Fetch metadata in single batch
        metadata_list = self.metadata_manager.get_track_metadata_batch(track_ids)
        
        # Build track info with signatures
        track_info = []
        for i, idx in enumerate(indices[:top_k * 2]):
            metadata = metadata_list[i]
            artist = metadata.get('artist_name', 'Unknown').lower()
            title = metadata.get('track_name', 'Unknown').lower()
            
            # Extract core title (remove featured artists, remixes, live versions)
            core_title = title.split('(')[0].split('-')[0].split('[')[0].strip()
            
            # Create robust signature
            signature = f"{artist}|{core_title}"
            
            track_info.append({
                'index': idx,
                'similarity': similarities[i],
                'isrc': isrcs[i],
                'signature': signature
            })
        
        # Sort by similarity descending for deterministic processing
        track_info.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Pass 1: Strict deduplication
        seen_signatures = set()
        seen_isrcs = set()
        deduped_indices = []
        deduped_similarities = []
        
        for track in track_info:
            if len(deduped_indices) >= top_k:
                break
            
            # Skip exact ISRC matches (same recording)
            if track['isrc'] and track['isrc'] in seen_isrcs:
                continue
            
            # Skip core song duplicates (different recordings, same song)
            if track['signature'] in seen_signatures:
                continue
            
            deduped_indices.append(track['index'])
            deduped_similarities.append(track['similarity'])
            seen_signatures.add(track['signature'])
            seen_isrcs.add(track['isrc'])
        
        # Pass 2: Fill remaining slots if needed (only filter by ISRC)
        if len(deduped_indices) < top_k:
            for track in track_info:
                if len(deduped_indices) >= top_k:
                    break
                
                # Skip if already added
                if track['index'] in deduped_indices:
                    continue
                
                # Skip only ISRC duplicates (allow different song versions)
                if track['isrc'] and track['isrc'] in seen_isrcs:
                    continue
                
                deduped_indices.append(track['index'])
                deduped_similarities.append(track['similarity'])
                seen_isrcs.add(track['isrc'])
        
        return deduped_indices[:top_k], deduped_similarities[:top_k]
    
    def _validate_implementation_parity(self):
        """Ensure CPU and GPU produce identical results."""
        test_vector = np.random.rand(32).astype(np.float32)
        test_vectors = np.random.rand(1000, 32).astype(np.float32)
        test_masks = np.random.randint(0, 2**32, size=1000, dtype=np.uint32)
        
        # CPU
        cpu_ops = VectorOps(algorithm=self.algorithm)
        cpu_results = cpu_ops.compute_similarity(test_vector, test_vectors, test_masks)
        
        # GPU
        if torch.cuda.is_available():
            gpu_ops = self.gpu_ops
            gpu_tensor = torch.tensor(test_vectors, device='cuda')
            gpu_masks = torch.tensor(test_masks, device='cuda')
            
            if self.algorithm == 'cosine':
                gpu_results = gpu_ops.masked_weighted_cosine_similarity(test_vector, gpu_tensor, gpu_masks)
            elif self.algorithm == 'cosine-euclidean':
                gpu_results = gpu_ops.masked_weighted_cosine_euclidean_similarity(test_vector, gpu_tensor, gpu_masks)
            else:
                gpu_results = gpu_ops.masked_euclidean_similarity(test_vector, gpu_tensor, gpu_masks)
            
            if not np.allclose(cpu_results, gpu_results.cpu().numpy(), atol=1e-5):
                raise ValueError("CPU/GPU implementation mismatch!")
    
    def close(self):
        """Clean up resources."""
        self.metadata_manager.close()
        if hasattr(self.vector_reader, '__del__'):
            self.vector_reader.__del__()
    
    @classmethod
    def clear_benchmark_cache(cls):
        """Clear benchmark results cache."""
        cls._benchmark_results = None


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
