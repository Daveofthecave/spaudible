# core/preprocessing/vector_exporter.py
import struct
import json
import time
import numpy as np
from pathlib import Path
import os
import mmap

class VectorWriter:
    """High-performance vector writer with optimized buffering and indexing."""
    
    # Constants for index format
    TRACK_ID_SIZE = 22
    ISRC_SIZE = 12
    OFFSET_SIZE = 8  # uint64
    INDEX_ENTRY_SIZE = TRACK_ID_SIZE + ISRC_SIZE + OFFSET_SIZE
    WRITE_BUFFER_SIZE = 1_000_000  # Buffer 1 million vectors before flush
    
    def __init__(self, output_dir="data/vectors"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # File paths
        self.vectors_path = self.output_dir / "track_vectors.bin"
        self.index_path = self.output_dir / "track_index.bin"
        self.metadata_path = self.output_dir / "metadata.json"
        
        # File handles
        self.vectors_file = None
        self.index_file = None
        
        # Buffers
        self.vector_buffer = bytearray()
        self.index_buffer = bytearray()
        self.buffer_count = 0
        
        # Metadata tracking
        self.total_tracks = 0
        self.valid_isrcs = 0
        self.start_time = time.time()
    
    def __enter__(self):
        """Open files for writing in binary mode."""
        self.vectors_file = open(self.vectors_path, 'wb')
        self.index_file = open(self.index_path, 'wb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close files and write metadata."""
        # Flush any remaining buffered data
        self._flush_buffers()
        
        if self.vectors_file:
            self.vectors_file.close()
        if self.index_file:
            self.index_file.close()
        
        # Write metadata if no error occurred
        if exc_type is None:
            self.write_metadata()
    
    def write_vector(self, track_id: str, vector: np.ndarray, isrc: str = ""):
        """
        Write a vector to the binary file with optimized buffering.
        
        Args:
            track_id: Spotify track ID (22 characters)
            vector: 32-dimensional float vector
            isrc: International Standard Recording Code (optional)
        """
        # Validate and pack vector
        vector_bytes = struct.pack('32f', *vector)
        
        # Get current offset
        offset = self.vectors_file.tell() + len(self.vector_buffer)
        
        # Add to vector buffer
        self.vector_buffer.extend(vector_bytes)
        
        # Prepare index entry
        padded_id = track_id.ljust(self.TRACK_ID_SIZE, '\0')[:self.TRACK_ID_SIZE].encode('utf-8')
        
        # Handle None or empty ISRC
        if isrc is None:
            isrc = ""
        padded_isrc = isrc.ljust(self.ISRC_SIZE, '\0')[:self.ISRC_SIZE].encode('utf-8')
        
        offset_bytes = struct.pack('Q', offset)
        
        # Add to index buffer
        self.index_buffer.extend(padded_id)
        self.index_buffer.extend(padded_isrc)
        self.index_buffer.extend(offset_bytes)
        
        # Update metadata
        self.total_tracks += 1
        if isrc.strip():
            self.valid_isrcs += 1
        
        self.buffer_count += 1
        
        # Flush buffers if full
        if self.buffer_count >= self.WRITE_BUFFER_SIZE:
            self._flush_buffers()
    
    def _flush_buffers(self):
        """Write buffered data to disk."""
        if self.buffer_count > 0:
            # Write vectors
            self.vectors_file.write(self.vector_buffer)
            
            # Write index
            self.index_file.write(self.index_buffer)
            
            # Clear buffers
            self.vector_buffer = bytearray()
            self.index_buffer = bytearray()
            self.buffer_count = 0
            
            # Flush OS buffers
            self.vectors_file.flush()
            self.index_file.flush()
    
    def write_metadata(self):
        """Write processing metadata to JSON file."""
        isrc_coverage = self.valid_isrcs / self.total_tracks if self.total_tracks > 0 else 0
        
        metadata = {
            "total_tracks": self.total_tracks,
            "vector_dimensions": 32,
            "bytes_per_vector": 128,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": round(time.time() - self.start_time),
            "database_source": "Spotify clean databases",
            "vector_format": "32 × float32 binary",
            "index_format": f"track_id({self.TRACK_ID_SIZE}B) + isrc({self.ISRC_SIZE}B) + offset({self.OFFSET_SIZE}B)",
            "isrc_coverage": f"{isrc_coverage:.4f}",
            "isrc_valid_count": self.valid_isrcs,
            "description": "Vector cache: track embeddings; Vector index: track ID + ISRC to offset mapping",
            "files": {
                "vectors": self.vectors_path.name,
                "index": self.index_path.name
            }
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
