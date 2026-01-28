# core/preprocessing/unified_vector_reader.py
import struct
import mmap
import numpy as np
from pathlib import Path
from config import PathConfig

class UnifiedVectorReader:
    """Reader for extracting metadata during index rebuild."""
    
    HEADER_SIZE = 16
    RECORD_SIZE = 104
    TRACK_ID_OFFSET = 82  # Track ID starts at byte 82 in record
    
    def __init__(self, vectors_path: Path):
        self.path = Path(vectors_path)
        self.file = open(self.path, "rb")
        self.file_size = self.path.stat().st_size
        self.total_vectors = (self.file_size - self.HEADER_SIZE) // self.RECORD_SIZE
        
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def get_total_vectors(self) -> int:
        return self.total_vectors
    
    def extract_metadata_batch(self, start_idx: int, num_vectors: int) -> list:
        """Extract (track_id, vector_index) pairs from vectors file."""
        metadata = []
        base_offset = self.HEADER_SIZE + start_idx * self.RECORD_SIZE
        
        for i in range(num_vectors):
            record_offset = base_offset + i * self.RECORD_SIZE
            track_id_bytes = self.mmap[record_offset + self.TRACK_ID_OFFSET:
                                       record_offset + self.RECORD_SIZE]
            track_id = track_id_bytes.decode('ascii', 'ignore').rstrip('\0')
            metadata.append((track_id, start_idx + i))
        
        return metadata
    
    def __del__(self):
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
