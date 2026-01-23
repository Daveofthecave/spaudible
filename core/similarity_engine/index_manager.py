# core/similarity_engine/index_manager.py
"""
Manage bidirectional mapping between track IDs and vector indices
from the sorted track_index.bin file.
"""
import struct
import mmap
import os
import threading
from pathlib import Path
from typing import List, Optional, Dict, Union
from collections import OrderedDict
from config import PathConfig

class IndexManager:
    """
    Manages the bidirectional mapping between Spotify track IDs and their positions
    in the vector cache (track_vectors.bin) with ISRC support.
    
    The index file (track_index.bin) stores fixed-size records containing:
    - Track ID (22 bytes, ASCII, null-padded)
    - Vector Index (4 bytes, uint32 little-endian)
    
    Each record is 26 bytes total (22 + 4).
    The file is sorted by track ID for O(log(n)) lookups.
    """
    
    TRACK_ID_SIZE = 22
    INDEX_SIZE = 4
    INDEX_ENTRY_SIZE = TRACK_ID_SIZE + INDEX_SIZE
    
    def __init__(self, index_path: Optional[str] = None, cache_size: int = 10000):
        """
        Initialize index manager with memory mapping and LRU cache.
        
        Args:
            index_path: Optional custom path to track_index.bin file.
                        Defaults to PathConfig.get_index_file()
            cache_size: Maximum number of track IDs to cache (default: 10,000)
        """
        self.index_path = Path(index_path) if index_path else PathConfig.get_index_file()
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector index file not found: {self.index_path}")
        
        # Open file and create memory map
        self._file = open(self.index_path, 'rb')
        self._file_size = self.index_path.stat().st_size
        self._mmap = mmap.mmap(self._file.fileno(), self._file_size, access=mmap.ACCESS_READ)
        
        # Calculate number of entries
        self.total_entries = self._file_size // self.INDEX_ENTRY_SIZE
        
        # Thread-safe LRU cache for track ID → vector index lookups
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._vector_index_cache = OrderedDict()
        self._max_cache_size = cache_size
        
        # Pre-compute memory offsets for quick access
        self._header_offset = 0  # No header in index file
        
        # Validate file size
        if self._file_size % self.INDEX_ENTRY_SIZE != 0:
            raise ValueError(f"Index file size {self._file_size} is not a multiple of {self.INDEX_ENTRY_SIZE}")
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()
    
    def _read_record(self, entry_idx: int) -> tuple:
        """
        Read a single record at the given entry index.
        
        Args:
            entry_idx: Index of the record (0-based)
            
        Returns:
            Tuple of (track_id_bytes, vector_index)
        """
        offset = entry_idx * self.INDEX_ENTRY_SIZE
        if offset >= self._file_size:
            raise IndexError(f"Entry index {entry_idx} out of bounds")
        
        # Read track ID (22 bytes)
        track_id_bytes = self._mmap[offset:offset + self.TRACK_ID_SIZE]
        
        # Read vector index (4 bytes, little-endian)
        vector_index_bytes = self._mmap[offset + self.TRACK_ID_SIZE:
                                        offset + self.INDEX_ENTRY_SIZE]
        vector_index = struct.unpack("<I", vector_index_bytes)[0]
        
        return track_id_bytes, vector_index
    
    def get_track_id(self, index: int) -> str:
        """
        Get track ID for a given vector index.
        
        Args:
            index: Position in vector cache (0-based index)
            
        Returns:
            Spotify track ID string
        """
        track_id_bytes, _ = self._read_record(index)
        return track_id_bytes.decode('ascii', errors='ignore').rstrip('\x00')
    
    def _binary_search(self, target_track_id: str) -> Optional[int]:
        """
        Perform binary search for a track ID using memory-mapped data.
        
        Args:
            target_track_id: Spotify track ID to search for
            
        Returns:
            Vector index if found, None otherwise
        """
        # Convert target to bytes for direct comparison
        target_bytes = target_track_id.ljust(self.TRACK_ID_SIZE, '\0').encode('ascii')
        
        left = 0
        right = self.total_entries - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Read track ID at midpoint
            offset = mid * self.INDEX_ENTRY_SIZE
            mid_track_id_bytes = self._mmap[offset:offset + self.TRACK_ID_SIZE]
            
            # Compare track IDs as byte strings
            if mid_track_id_bytes == target_bytes:
                # Found! Extract vector index
                vector_index_bytes = self._mmap[offset + self.TRACK_ID_SIZE:
                                               offset + self.INDEX_ENTRY_SIZE]
                return struct.unpack("<I", vector_index_bytes)[0]
            elif mid_track_id_bytes < target_bytes:
                left = mid + 1
            else:
                right = mid - 1
        
        return None
    
    def get_index_from_track_id(self, track_id: str, use_cache: bool = True) -> Optional[int]:
        """
        Find vector index for a given track ID (fast O(log n) binary search).
        
        Args:
            track_id: Spotify track ID
            use_cache: Whether to use the LRU cache (default: True)
            
        Returns:
            Vector index or None if not found
        """
        # Check cache first
        if use_cache:
            with self._cache_lock:
                if track_id in self._cache:
                    # Move to end (most recently used)
                    self._cache.move_to_end(track_id)
                    return self._cache[track_id]
        
        # Perform binary search
        vector_index = self._binary_search(track_id)
        
        # Cache the result
        if vector_index is not None and use_cache:
            with self._cache_lock:
                self._cache[track_id] = vector_index
                # Evict oldest if cache is full
                if len(self._cache) > self._max_cache_size:
                    self._cache.popitem(last=False)
        
        return vector_index
    
    def get_track_ids_batch(self, indices: List[int]) -> List[str]:
        """
        Get multiple track IDs at once (optimized sequential read).
        
        Args:
            indices: List of vector cache positions
            
        Returns:
            List of Spotify track IDs in same order as input indices
        """
        track_ids = []
        
        # Sort indices to enable sequential reads
        sorted_indices = sorted(enumerate(indices), key=lambda x: x[1])
        
        # Read in sorted order
        last_offset = -1
        for original_pos, idx in sorted_indices:
            # Calculate offset
            offset = idx * self.INDEX_ENTRY_SIZE
            
            # Check if we can reuse previous read
            if offset == last_offset:
                # Same record as last read
                track_id = track_ids[-1] if track_ids else ""
            else:
                # New read
                track_id_bytes = self._mmap[offset:offset + self.TRACK_ID_SIZE]
                track_id = track_id_bytes.decode('ascii', errors='ignore').rstrip('\0')
            
            track_ids.append((original_pos, track_id))
            last_offset = offset
        
        # Re-order to match original input order
        track_ids.sort(key=lambda x: x[0])
        
        return [tid for _, tid in track_ids]
    
    def get_isrcs_batch(self, indices: List[int]) -> List[str]:
        """
        Get multiple ISRCs at once.
        
        Note: ISRCs are not stored in the index file.
        Use vector file to retrieve ISRCs.
        
        Args:
            indices: List of vector cache positions
            
        Returns:
            List of empty strings (ISRCs not available in index)
        """
        # ISRCs are stored in vector file, not index
        return ["" for _ in indices]

    def get_track_ids_from_vector_indices(self, vector_indices: List[int]) -> List[str]:
        """
        High-level wrapper to get track IDs from vector indices.
        Uses VectorReader and caches results for performance.
        """
        # Cache setup (add to __init__ if not present)
        if not hasattr(self, '_vector_index_cache'):
            self._vector_index_cache = OrderedDict()
            self._max_cache_size = 10000 
        
        if not vector_indices:
            return []
        
        # Check cache first
        results = [None] * len(vector_indices)
        uncached_positions = []
        uncached_indices = []
        
        for i, idx in enumerate(vector_indices):
            if idx in self._vector_index_cache:
                results[i] = self._vector_index_cache[idx]
            else:
                uncached_positions.append(i)
                uncached_indices.append(idx)
        
        # Batch fetch uncached IDs
        if uncached_indices:
            # Get vector reader reference (set by orchestrator)
            vector_reader = getattr(self, '_vector_reader', None)
            
            if vector_reader is None:
                # Fallback: create temporary reader
                from .vector_io_gpu import VectorReaderGPU
                from config import PathConfig
                
                vector_reader = VectorReaderGPU(str(PathConfig.get_vector_file()))
                uncached_ids = vector_reader.get_track_ids_batch(uncached_indices)
                # Don't cache since this is temporary
                
                # Clean up temporary reader
                vector_reader.close()
            else:
                uncached_ids = vector_reader.get_track_ids_batch(uncached_indices)
                
                # Cache the new results
                for idx, track_id in zip(uncached_indices, uncached_ids):
                    self._vector_index_cache[idx] = track_id
                    
                    # Maintain cache size limit
                    if len(self._vector_index_cache) > self._max_cache_size:
                        self._vector_index_cache.popitem(last=False)
            
            # Place fetched IDs in result array
            for pos, track_id in zip(uncached_positions, uncached_ids):
                results[pos] = track_id
        
        return results
    
    def validate_sorted_order(self, sample_size: int = 1000) -> bool:
        """
        Validate that the index file is properly sorted by track ID.
        
        Args:
            sample_size: Number of random samples to check
            
        Returns:
            True if sorted, False otherwise
        """
        import random
        
        # Sample random entries and verify they're in order
        prev_track_id = None
        
        for _ in range(sample_size):
            idx = random.randint(0, self.total_entries - 1)
            track_id = self.get_track_id(idx)
            
            if prev_track_id is not None and track_id < prev_track_id:
                print(f"Sorting violation: {track_id} < {prev_track_id}")
                return False
            
            prev_track_id = track_id
        
        return True
    
    def build_cache_dict(self, track_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Build a complete dictionary mapping of track ID → vector index.
        Warning: This consumes significant memory (~8 GB for all 256M tracks).
        
        Args:
            track_ids: Optional list of specific track IDs to cache.
                      If None, caches all entries (not recommended).
            
        Returns:
            Dictionary mapping track IDs to vector indices
        """
        cache_dict = {}
        
        if track_ids is None:
            # Cache all entries - memory intensive!
            print("Warning: Caching all entries will use ~8 GB RAM...")
            
            for idx in range(self.total_entries):
                track_id_bytes, vector_index = self._read_record(idx)
                track_id = track_id_bytes.decode('ascii', errors='ignore').rstrip('\0')
                
                if track_id:  # Skip empty track IDs
                    cache_dict[track_id] = vector_index
        else:
            # Cache only requested track IDs
            for track_id in track_ids:
                vector_index = self.get_index_from_track_id(track_id, use_cache=False)
                if vector_index is not None:
                    cache_dict[track_id] = vector_index
        
        return cache_dict
    
    def get_total_entries(self) -> int:
        """Get total number of entries in the index."""
        return self.total_entries
    
    def get_file_size(self) -> int:
        """Get the size of the index file in bytes."""
        return self._file_size
