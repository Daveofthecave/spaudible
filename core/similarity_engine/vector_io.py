# core/similarity_engine/vector_io.py
import mmap
import numpy as np
from pathlib import Path
from typing import Union

class VectorReader:
    """CPU-based reader for track_vectors.bin employing structured dtypes
    to interpret the new 104-byte unified vector format"""
    
    VECTOR_RECORD_SIZE = 104
    VECTOR_HEADER_SIZE = 16
    
    def __init__(self, vectors_path: Union[str, Path]):
        self.vectors_path = Path(vectors_path)
        
        # Define structured dtype matching the binary record layout in track_vectors.bin
        self.record_dtype = np.dtype([
            ('binary', np.uint8),             # byte 0: packed binary dims
            ('scaled', np.uint16, (22,)),     # bytes 1-44: 22 uint16 values
            ('fp32', np.float32, (5,)),       # bytes 45-64: 5 float32 values
            ('mask', np.uint32),              # bytes 65-68: validity bitmask
            ('region', np.uint8),             # byte 69: region code
            ('isrc', 'S12'),                  # bytes 70-81: ISRC
            ('track_id', 'S22'),              # bytes 82-103: track ID
        ])
        
        # Open and memory-map the unified file
        self._file = open(self.vectors_path, 'rb')
        self._file_size = self.vectors_path.stat().st_size
        self.total_vectors = (self._file_size - self.VECTOR_HEADER_SIZE) // self.VECTOR_RECORD_SIZE
        
        # Create memory map
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Map as structured records (avoids alignment issues)
        self._records = np.frombuffer(self._mmap, dtype=self.record_dtype,
                                      offset=self.VECTOR_HEADER_SIZE,
                                      count=self.total_vectors)
        
    def get_total_vectors(self) -> int:
        return self.total_vectors
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Read and unpack vectors to float32[32] from the unified format"""
        if start_idx >= self.total_vectors:
            return np.empty((0, 32), dtype=np.float32)
        
        if start_idx + num_vectors > self.total_vectors:
            num_vectors = self.total_vectors - start_idx
        
        # Access records slice (this is a view, very fast)
        records = self._records[start_idx:start_idx + num_vectors]
        
        # Initialize output array
        vectors = np.full((num_vectors, 32), -1.0, dtype=np.float32)
        
        # === Unpack binary-packed dimensions (byte 0) ===
        binary_byte = records['binary']
        vectors[:, 9]  = (binary_byte & 0b00000001).astype(np.float32)  # mode (dim 10)
        vectors[:, 11] = ((binary_byte >> 1) & 1).astype(np.float32)   # ts_4_4 (dim 12)
        vectors[:, 12] = ((binary_byte >> 2) & 1).astype(np.float32)   # ts_3_4 (dim 13)
        vectors[:, 13] = ((binary_byte >> 3) & 1).astype(np.float32)   # ts_5_4 (dim 14)
        vectors[:, 14] = ((binary_byte >> 4) & 1).astype(np.float32)   # ts_other (dim 15)
        
        # === Unpack scaled uint16 dimensions (bytes 1-44) ===
        scaled = records['scaled'].astype(np.float32) * 0.0001
        vectors[:, 0:7] = scaled[:, 0:7]       # dims 1-7: acousticness through liveness
        vectors[:, 8]   = scaled[:, 7]         # dim 9: key
        vectors[:, 16]  = scaled[:, 8]         # dim 17: release_date
        vectors[:, 19:32] = scaled[:, 9:22]    # dims 20-32: meta-genres
        
        # === Unpack fp32 dimensions (bytes 45-64) ===
        fp32 = records['fp32']
        vectors[:, 7]  = fp32[:, 0]  # dim 8: loudness
        vectors[:, 10] = fp32[:, 1]  # dim 11: tempo
        vectors[:, 15] = fp32[:, 2]  # dim 16: duration
        vectors[:, 17] = fp32[:, 3]  # dim 18: popularity
        vectors[:, 18] = fp32[:, 4]  # dim 19: artist followers
        
        return vectors
    
    def read_masks(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Read 4-byte validity masks from byte 65-68 of each record"""
        if start_idx >= self.total_vectors:
            return np.empty(0, dtype=np.uint32)
        
        return self._records[start_idx:start_idx + num_vectors]['mask']
    
    def read_regions(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Read 1-byte region codes from byte 69 of each record"""
        if start_idx >= self.total_vectors:
            return np.empty(0, dtype=np.uint8)
        
        return self._records[start_idx:start_idx + num_vectors]['region'].astype(np.uint8)
    
    def get_isrcs_batch(self, indices: np.ndarray) -> list:
        """Extract ISRCs from bytes 70-81 of records for given indices"""
        return [r['isrc'].decode('ascii', errors='ignore').rstrip('\0') 
                for r in self._records[indices]]
    
    def get_track_ids_batch(self, indices: np.ndarray) -> list:
        """Extract track IDs from bytes 82-103 of records for given indices"""
        return [r['track_id'].decode('ascii', errors='ignore').rstrip('\0') 
                for r in self._records[indices]]
    
    def close(self):
        """Clean up memory-mapped resources"""
        if hasattr(self, '_records'):
            # Delete numpy reference first
            del self._records
        
        if hasattr(self, '_mmap'):
            self._mmap.close()
        
        if hasattr(self, '_file'):
            self._file.close()
