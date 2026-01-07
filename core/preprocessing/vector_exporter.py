# core/preprocessing/vector_exporter.py
import struct
import json
import time
import numpy as np
from pathlib import Path

class VectorWriter:
    """High-performance vector writer with buffered I/O."""
    
    # Constants for new index format
    ISRC_SIZE = 12  # Max ISRC length (12 characters)
    TRACK_ID_SIZE = 22
    OFFSET_SIZE = 8  # uint64
    INDEX_ENTRY_SIZE = ISRC_SIZE + TRACK_ID_SIZE + OFFSET_SIZE
    WRITE_BUFFER_SIZE = 100000  # Buffer 100,000 vectors before flush
    
    def __init__(self, output_dir="data/vectors"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # File paths
        self.vectors_path = self.output_dir / "track_vectors.bin"
        self.index_path = self.output_dir / "track_index.bin"
        self.masks_path = self.output_dir / "track_masks.bin"
        self.metadata_path = self.output_dir / "metadata.json"
        
        # File handles
        self.vectors_file = None
        self.index_file = None
        self.masks_file = None
        
        # Track metadata
        self.track_ids = []
        self.track_isrcs = []
        self.vector_offsets = []
        
        # Buffers
        self.vector_buffer = bytearray()
        self.index_buffer = bytearray()
        self.mask_buffer = bytearray()
        self.buffer_count = 0
    
    def __enter__(self):
        """Open files for writing."""
        self.vectors_file = open(self.vectors_path, 'wb')
        self.index_file = open(self.index_path, 'wb')
        self.masks_file = open(self.masks_path, 'wb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close files and write metadata."""
        # Flush any remaining buffered data
        self._flush_buffers()
        
        if self.vectors_file:
            self.vectors_file.close()
        if self.index_file:
            self.index_file.close()
        if self.masks_file:
            self.masks_file.close()
        
        # Write metadata if no error occurred
        if exc_type is None:
            self.write_metadata()
    
    def write_vector(self, track_id: str, vector, isrc: str = ""):
        """
        Write a single vector to the binary file with buffered I/O.
        """
        # Validate vector before packing
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                try:
                    vector[i] = float(val)
                except (ValueError, TypeError):
                    vector[i] = -1.0

        try:
            vector_bytes = struct.pack('32f', *vector)
        except struct.error as e:
            print(f"\n❌ Failed to pack vector for track {track_id}: {e}")
            vector = [-1.0] * 32
            vector_bytes = struct.pack('32f', *vector)
        
        # Compute validity bitmask (1 bit per dimension)
        bitmask = 0
        for i, val in enumerate(vector):
            if val != -1.0:  # Valid dimension
                bitmask |= (1 << i)
        mask_bytes = struct.pack('<I', bitmask)  # Little-endian uint32
        
        # Get current offset
        offset = self.vectors_file.tell() + len(self.vector_buffer)
        
        # Add to buffers
        self.vector_buffer.extend(vector_bytes)
        self.mask_buffer.extend(mask_bytes)
        
        # Prepare index entry components
        # Handle None ISRC values
        safe_isrc = isrc if isrc is not None else ""
        padded_isrc = safe_isrc.encode('utf-8').ljust(self.ISRC_SIZE, b'\0')[:self.ISRC_SIZE]
        
        padded_id = track_id.ljust(self.TRACK_ID_SIZE, '\0').encode('utf-8')
        offset_bytes = struct.pack('Q', offset)
        
        self.index_buffer.extend(padded_isrc)
        self.index_buffer.extend(padded_id)
        self.index_buffer.extend(offset_bytes)
        
        # Store metadata
        self.track_ids.append(track_id)
        self.track_isrcs.append(safe_isrc)
        self.vector_offsets.append(offset)
        
        self.buffer_count += 1
        
        # Flush buffers if full
        if self.buffer_count >= self.WRITE_BUFFER_SIZE:
            self._flush_buffers()
    
    def _flush_buffers(self):
        """Write buffered data to disk."""
        if self.buffer_count > 0:
            self.vectors_file.write(self.vector_buffer)
            self.index_file.write(self.index_buffer)
            self.masks_file.write(self.mask_buffer)
            
            # Clear buffers
            self.vector_buffer = bytearray()
            self.index_buffer = bytearray()
            self.mask_buffer = bytearray()
            self.buffer_count = 0
            
            # Flush OS buffers
            self.vectors_file.flush()
            self.index_file.flush()
            self.masks_file.flush()
    
    def write_metadata(self):
        """Write processing metadata to JSON file."""
        # Count valid ISRCs
        valid_isrcs = sum(1 for isrc in self.track_isrcs if isrc)
        total_tracks = len(self.track_ids)
        isrc_coverage = valid_isrcs / total_tracks if total_tracks > 0 else 0
        
        metadata = {
            "total_tracks": total_tracks,
            "vector_dimensions": 32,
            "bytes_per_vector": 128,
            "mask_bytes_per_vector": 4,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "database_source": "Spotify clean databases",
            "vector_format": "32 × float32 binary",
            "mask_format": "32-bit bitmask (1 bit per dimension)",
            "index_format": "isrc(12B) + track_id(22B) + offset(8B)",
            "isrc_coverage": f"{isrc_coverage:.4f}",
            "isrc_valid_count": valid_isrcs,
            "description": "Vector cache: track embeddings; Mask: validity bitmask; Vector index: ISRC + track ID to offset mapping",
            "files": {
                "vectors": str(self.vectors_path.name),
                "masks": str(self.masks_path.name),
                "index": str(self.index_path.name)
            }
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
