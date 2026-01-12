# core/preprocessing/unified_vector_writer.py
import struct
import numpy as np
from pathlib import Path
from typing import Tuple, List
import zlib
import os

class UnifiedVectorWriter:
    """Highly optimized writer with vectorized packing and batched I/O."""
    
    # File header format (16 bytes)
    HEADER_FORMAT = "<4sI8s"  # Magic, version, checksum placeholder
    HEADER_SIZE = 16
    
    # Record format (104 bytes)
    RECORD_SIZE = 104
    
    # Magic number for file identification
    MAGIC = b"SPAU"  # Spaudible Packed Vector
    
    # Batch size for vector packing
    WRITE_BATCH_SIZE = 500_000
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.vectors_path = self.output_dir / "track_vectors.bin"
        self.index_path = self.output_dir / "track_index.bin"
        
        # File handles
        self.vector_file = None
        self.index_file = None
        
        # Buffers
        self.track_ids = []
        self.vectors = []
        self.isrcs = []
        self.regions = []
        self.batch_count = 0
        self.total_records = 0

    def __enter__(self):
        """Open files for writing and write header"""
        self.vector_file = open(self.vectors_path, "wb")
        self.index_file = open(self.index_path, "wb")
        self._write_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize files and write index"""
        if exc_type is None:
            # Flush any remaining records
            self._flush_buffers()
            self._write_index()
            self._write_checksum()
        
        # Close files
        if self.vector_file:
            self.vector_file.close()
        if self.index_file:
            self.index_file.close()
        
        return False

    def _write_header(self):
        """Write file header with magic, version, and checksum placeholder"""
        header = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            1,  # Version 1
            b"\0" * 8  # Checksum placeholder
        )
        self.vector_file.write(header)
    
    def write_record(
        self,
        track_id: str,
        vector: np.ndarray,
        isrc: str = "",
        region: int = 7  # Default to "Other" region
    ):
        # Add to batch
        self.track_ids.append(track_id)
        self.vectors.append(vector)
        self.isrcs.append(isrc)
        self.regions.append(region)
        self.batch_count += 1
        self.total_records += 1
        
        # Flush if batch is full
        if self.batch_count >= self.WRITE_BATCH_SIZE:
            self._flush_buffers()

    def _flush_buffers(self):
        """Process and write a full batch of vectors"""
        if self.batch_count == 0:
            return
            
        # Convert to NumPy arrays for vectorized operations
        vectors = np.array(self.vectors)
        regions = np.array(self.regions)
        
        # Pack binary dimensions
        binary_bytes = self._pack_binary_dims_batch(vectors)
        
        # Pack scaled dimensions
        scaled_dims = self._pack_scaled_dims_batch(vectors)
        
        # Pack FP32 dimensions
        fp32_dims = self._pack_fp32_dims_batch(vectors)
        
        # Generate validity masks
        validity_masks = self._get_validity_masks_batch(vectors)
        
        # Clean ISRCs - handle non-ASCII characters
        clean_isrcs = []
        for isrc in self.isrcs:
            # Remove non-ASCII characters
            clean = ''.join(c for c in isrc if ord(c) < 128)
            # Truncate to 12 characters and pad with nulls
            clean = clean[:12].ljust(12, '\0')
            clean_isrcs.append(clean)
            
        # Format track IDs - ensure ASCII-only
        clean_track_ids = []
        for tid in self.track_ids:
            # Remove non-ASCII characters
            clean = ''.join(c for c in tid if ord(c) < 128)
            # Truncate to 22 characters and pad with nulls
            clean = clean[:22].ljust(22, '\0')
            clean_track_ids.append(clean)
        
        # Write records
        for i in range(self.batch_count):
            record = struct.pack(
                "<B22H5fIB12s22s",
                binary_bytes[i],
                *scaled_dims[i],
                *fp32_dims[i],
                validity_masks[i],
                regions[i],
                clean_isrcs[i].encode("ascii"),
                clean_track_ids[i].encode("ascii")
            )
            self.vector_file.write(record)
        
        # Clear buffers
        self.track_ids = []
        self.vectors = []
        self.isrcs = []
        self.regions = []
        self.batch_count = 0
        
        # Flush to disk
        self.vector_file.flush()
    
    def _pack_binary_dims_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Pack binary dimensions for a batch of vectors."""
        # Replace NaNs with -1.0
        vectors = np.nan_to_num(vectors, nan=-1.0)
        
        # Initialize with zeros
        binary_bytes = np.zeros(vectors.shape[0], dtype=np.uint8)
        
        # Mode (dimension 9)
        mode_vals = vectors[:, 9]
        valid_mask = (mode_vals != -1.0)
        mode_bits = np.clip(mode_vals[valid_mask], 0, 1).astype(np.uint8)
        binary_bytes[valid_mask] |= mode_bits
        
        # Time signatures (dimensions 11-14)
        time_sig_dims = [11, 12, 13, 14]
        for i, dim_idx in enumerate(time_sig_dims, start=1):
            vals = vectors[:, dim_idx]
            valid_mask = (vals != -1.0)
            sig_bits = np.clip(vals[valid_mask], 0, 1).astype(np.uint8) << i
            binary_bytes[valid_mask] |= sig_bits
        
        return binary_bytes
        
    def _pack_scaled_dims_batch(self, vectors: np.ndarray) -> List[List[int]]:
        """Pack scaled dimensions for a batch of vectors."""
        # Replace NaNs with -1.0
        vectors = np.nan_to_num(vectors, nan=-1.0)
        
        # Initialize output
        scaled_dims = []
        
        # Dimensions 1-7, 9, 17
        scaled_indices = [0, 1, 2, 3, 4, 5, 6, 8, 16]
        for i in scaled_indices:
            dim_vals = vectors[:, i]
            # Handle missing values
            valid_mask = dim_vals != -1.0
            scaled = np.zeros_like(dim_vals, dtype=np.uint16)
            scaled[valid_mask] = (np.clip(dim_vals[valid_mask], 0, 1) * 10000).astype(np.uint16)
            scaled_dims.append(scaled)
        
        # Meta-genres (dimensions 20-32)
        for i in range(19, 32):
            dim_vals = vectors[:, i]
            valid_mask = dim_vals != -1.0
            scaled = np.zeros_like(dim_vals, dtype=np.uint16)
            scaled[valid_mask] = (np.clip(dim_vals[valid_mask], 0, 1) * 10000).astype(np.uint16)
            scaled_dims.append(scaled)
        
        # Transpose to match record format
        return np.array(scaled_dims).T.tolist()
    
    def _pack_fp32_dims_batch(self, vectors: np.ndarray) -> List[List[float]]:
        """Pack FP32 dimensions for a batch of vectors."""
        fp32_dims = []
        fp32_indices = [7, 10, 15, 17, 18]  # Loudness, tempo, duration, popularity, followers
        
        # Copy valid values, replace NaNs/invalids with 0.0
        for i in fp32_indices:
            dim_vals = vectors[:, i].copy()
            invalid_mask = (dim_vals == -1.0) | np.isnan(dim_vals)
            dim_vals[invalid_mask] = 0.0
            fp32_dims.append(dim_vals)
        
        # Transpose to match record format
        return np.array(fp32_dims).T.tolist()
    
    def _get_validity_masks_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Generate validity bitmasks for a batch of vectors."""
        validity_masks = np.zeros(vectors.shape[0], dtype=np.uint32)
        
        # Create mask of invalid values
        invalid_mask = (vectors == -1.0) | np.isnan(vectors)
        
        # Set bits for invalid dimensions
        for i in range(32):
            validity_masks[invalid_mask[:, i]] |= (1 << i)
        
        return validity_masks

    def _write_index(self):
        """Write sorted index file."""
        # Sort index entries by track ID
        sorted_indices = sorted(range(len(self.track_ids)), key=lambda i: self.track_ids[i])
        
        # Write each entry (22B track ID + 4B index)
        for idx in sorted_indices:
            self.index_file.write(self.track_ids[idx].encode("ascii").ljust(22, b'\0'))
            self.index_file.write(struct.pack("<I", idx))
        
        self.index_file.flush()
    
    def _write_checksum(self):
        """Calculate CRC32 checksum and update file header."""
        # Close the vector file first to ensure all data is flushed
        self.vector_file.close()
        
        # Read entire file except header
        with open(self.vectors_path, "rb") as f:
            f.seek(self.HEADER_SIZE)
            data = f.read()
        
        # Calculate checksum
        checksum = zlib.crc32(data)
        
        # Update header with checksum
        with open(self.vectors_path, "r+b") as f:
            f.seek(8)  # Position after magic and version
            f.write(struct.pack("<Q", checksum))
    
    def finalize(self):
        """Finalize processing."""
        self._flush_buffers()
