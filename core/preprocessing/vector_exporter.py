# core/preprocessing/vector_exporter.py
import struct
import json
import time
import numpy as np
from pathlib import Path

class VectorWriter:
    """Manages binary storage of vectors, masks, and index."""
    
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
        self.vector_offsets = []
        
    def __enter__(self):
        """Open files for writing."""
        self.vectors_file = open(self.vectors_path, 'wb')
        self.index_file = open(self.index_path, 'wb')
        self.masks_file = open(self.masks_path, 'wb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close files and write metadata."""
        if self.vectors_file:
            self.vectors_file.close()
        if self.index_file:
            self.index_file.close()
        if self.masks_file:
            self.masks_file.close()
        
        # Write metadata if no error occurred
        if exc_type is None:
            self.write_metadata()
    
    def write_vector(self, track_id: str, vector):
        """
        Write a single vector to the binary file and its validity mask.
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
        offset = self.vectors_file.tell()
        
        # Write vector
        self.vectors_file.write(vector_bytes)
        
        # Write mask
        self.masks_file.write(mask_bytes)
        
        # Store index information
        self.track_ids.append(track_id)
        self.vector_offsets.append(offset)
        
        # Write to index file
        padded_id = track_id.ljust(22, '\0').encode('utf-8')
        offset_bytes = struct.pack('Q', offset)
        self.index_file.write(padded_id + offset_bytes)
        
        # Flush periodically
        if len(self.track_ids) % 10000 == 0:
            self.vectors_file.flush()
            self.index_file.flush()
            self.masks_file.flush()
    
    def write_metadata(self):
        """Write processing metadata to JSON file."""
        metadata = {
            "total_tracks": len(self.track_ids),
            "vector_dimensions": 32,
            "bytes_per_vector": 128,
            "mask_bytes_per_vector": 4,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "database_source": "Spotify clean databases",
            "vector_format": "32 × float32 binary",
            "mask_format": "32-bit bitmask (1 bit per dimension)",
            "index_format": "track_id (22B) + offset (8B)",
            "description": "Vector cache: track embeddings; Mask: validity bitmask; Vector index: track ID to offset mapping",
            "files": {
                "vectors": str(self.vectors_path.name),
                "masks": str(self.masks_path.name),
                "index": str(self.index_path.name)
            }
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
