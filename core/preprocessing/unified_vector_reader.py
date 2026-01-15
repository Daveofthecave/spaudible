# core/preprocessing/unified_vector_reader.py
import struct
from pathlib import Path

class UnifiedVectorReader:
    """Efficient reader for unified vector format."""
    
    HEADER_SIZE = 16
    RECORD_SIZE = 104
    
    def __init__(self, file_path: Path):
        self.path = Path(file_path)
        self.file = open(self.path, "rb")
        
        # Read header
        self.file.seek(0)
        header = self.file.read(self.HEADER_SIZE)
        self.magic = header[:4]
        self.version = struct.unpack("<I", header[4:8])[0]
        
        # Calculate number of vectors
        file_size = self.path.stat().st_size
        self.total_vectors = (file_size - self.HEADER_SIZE) // self.RECORD_SIZE
    
    def get_total_vectors(self) -> int:
        return self.total_vectors
    
    def get_vector_metadata_batch(self, start_idx: int, num_vectors: int):
        """Efficiently get metadata for a batch of vectors."""
        # Calculate byte range
        start_byte = self.HEADER_SIZE + start_idx * self.RECORD_SIZE
        bytes_to_read = num_vectors * self.RECORD_SIZE
        
        # Read entire chunk at once
        self.file.seek(start_byte)
        chunk_data = self.file.read(bytes_to_read)
        
        if len(chunk_data) != bytes_to_read:
            raise IOError(f"Expected {bytes_to_read} bytes, got {len(chunk_data)}")
        
        # Parse metadata
        metadata = []
        for i in range(num_vectors):
            # Track ID is at offset 82 in each record
            offset = i * self.RECORD_SIZE + 82
            track_id_bytes = chunk_data[offset:offset+22]
            track_id = track_id_bytes.decode('ascii', 'ignore').rstrip('\0')
            metadata.append((track_id, start_idx + i))
        
        return metadata
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()
