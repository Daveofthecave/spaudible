# core/preprocessing/unified_vector_writer.py
import struct
import numpy as np
from pathlib import Path
from typing import Tuple, List
import zlib
import os
import tempfile
import shutil
from numba import njit, prange
import csv
from config import PathConfig
from core.utilities.region_utils import REGION_MAPPING

# Precomputed genre mappings
GENRE_MAPPING = {}
GENRE_ID_MAP = {}
GENRE_INTENSITY_MAP = {}

def _preload_genre_data():
    """Preload genre mapping data once at module import"""
    csv_path = PathConfig.get_genre_mapping()
    if not csv_path.exists():
        return
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 4:
                continue
            try:
                meta_genre = int(row[1])
                genre = row[2].strip().lower()
                intensity = float(row[3])
                
                # Store mappings
                GENRE_MAPPING[genre] = (meta_genre, intensity)
                GENRE_ID_MAP[genre] = len(GENRE_ID_MAP)
                GENRE_INTENSITY_MAP[genre] = intensity
            except (ValueError, IndexError):
                continue

# Load genre data on import
_preload_genre_data()

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
        
        # Precomputed mappings
        self.region_map = self._preload_region_map()
        
        # Temporary index storage
        self.temp_index_dir = Path(tempfile.mkdtemp())
        self.temp_index_file = None

    def _preload_region_map(self):
        """Precompute region mappings for faster lookup"""
        region_map = {}
        for region_id, countries in REGION_MAPPING.items():
            for country in countries:
                region_map[country] = region_id
        return region_map

    def __enter__(self):
        """Open files for writing and write header"""
        self.vector_file = open(self.vectors_path, "wb")
        self.index_file = open(self.index_path, "wb")
        self._write_header()
        
        # Open temporary index file
        self.temp_index_file = open(self.temp_index_dir / "temp_index.bin", "wb")
        
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
        if self.temp_index_file:
            self.temp_index_file.close()
        
        # Clean up temporary directory
        shutil.rmtree(self.temp_index_dir)
        
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
        
        # Write temporary index entry
        tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
        self.temp_index_file.write(tid_bytes)
        self.temp_index_file.write(struct.pack("<I", self.total_records))
        
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
        
        # Precompute clean ISRCs and track IDs
        clean_isrcs = [self._clean_isrc(isrc) for isrc in self.isrcs]
        clean_track_ids = [self._clean_track_id(tid) for tid in self.track_ids]
        
        # Write records in bulk
        records = []
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
            records.append(record)
        
        # Write all records at once
        self.vector_file.write(b"".join(records))
        
        # Clear buffers
        self.track_ids = []
        self.vectors = []
        self.isrcs = []
        self.regions = []
        self.batch_count = 0
        
        # Flush to disk
        self.vector_file.flush()
        self.temp_index_file.flush()
    
    def _clean_isrc(self, isrc: str) -> str:
        """Clean and format ISRC string"""
        if not isrc:
            return ""
        # Remove non-ASCII characters
        clean = ''.join(c for c in isrc if ord(c) < 128)
        # Truncate to 12 characters and pad with nulls
        return clean[:12].ljust(12, '\0')
    
    def _clean_track_id(self, track_id: str) -> str:
        """Clean and format track ID string"""
        if not track_id:
            return ""
        # Remove non-ASCII characters
        clean = ''.join(c for c in track_id if ord(c) < 128)
        # Truncate to 22 characters and pad with nulls
        return clean[:22].ljust(22, '\0')

    def _pack_binary_dims_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Pure Python implementation for packing binary dimensions with NaN handling"""
        n = vectors.shape[0]
        binary_bytes = np.zeros(n, dtype=np.uint8)
        
        for i in range(n):
            # Mode (dimension 9)
            mode_val = vectors[i, 9]
            if not np.isnan(mode_val) and mode_val != -1.0:
                binary_bytes[i] |= int(mode_val) & 1
            
            # Time signatures (dimensions 11-14)
            for j, dim_idx in enumerate([11, 12, 13, 14], start=1):
                val = vectors[i, dim_idx]
                if not np.isnan(val) and val != -1.0:
                    binary_bytes[i] |= (int(val) & 1) << j
        
        return binary_bytes

    def _pack_scaled_dims_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Pure Python implementation for packing scaled dimensions"""
        n = vectors.shape[0]
        scaled_dims = np.zeros((n, 22), dtype=np.uint16)
        
        # Dimensions 1-7, 9, 17
        scaled_indices = [0, 1, 2, 3, 4, 5, 6, 8, 16]  # indices 0,1,2,3,4,5,6,7,8
        for idx, dim in enumerate(scaled_indices):
            for i in range(n):
                val = vectors[i, dim]
                if np.isnan(val) or val == -1.0:
                    scaled_dims[i, idx] = 0
                else:
                    # Manual clamping instead of np.clip
                    clamped_val = max(0.0, min(val, 1.0))
                    scaled_dims[i, idx] = int(clamped_val * 10000)
        
        # Meta-genres (dimensions 19-31)
        for dim in range(19, 32):
            idx = dim - 19 + 9  # Continue from index 9
            for i in range(n):
                val = vectors[i, dim]
                if np.isnan(val) or val == -1.0:
                    scaled_dims[i, idx] = 0
                else:
                    # Manual clamping instead of np.clip
                    clamped_val = max(0.0, min(val, 1.0))
                    scaled_dims[i, idx] = int(clamped_val * 10000)
        
        return scaled_dims
    
    @staticmethod
    @njit(parallel=True)
    def _pack_fp32_dims_batch(vectors: np.ndarray) -> np.ndarray:
        """Vectorized packing of FP32 dimensions"""
        n = vectors.shape[0]
        fp32_dims = np.zeros((n, 5), dtype=np.float32)
        
        # Loudness, tempo, duration, popularity, followers
        fp32_indices = [7, 10, 15, 17, 18]
        for idx, dim in enumerate(fp32_indices):
            for i in prange(n):
                val = vectors[i, dim]
                if val == -1.0 or np.isnan(val):
                    fp32_dims[i, idx] = 0.0
                else:
                    fp32_dims[i, idx] = val
        
        return fp32_dims
    
    @staticmethod
    @njit(parallel=True)
    def _get_validity_masks_batch(vectors: np.ndarray) -> np.ndarray:
        """Vectorized validity mask generation"""
        n = vectors.shape[0]
        masks = np.zeros(n, dtype=np.uint32)
        
        for i in prange(n):
            for j in range(32):
                val = vectors[i, j]
                if val == -1.0 or np.isnan(val):
                    masks[i] |= (1 << j)
        
        return masks

    def _write_index(self):
        """Write sorted index file from temporary storage"""
        # Close temporary file and reopen for reading
        self.temp_index_file.close()
        temp_index_path = self.temp_index_dir / "temp_index.bin"
        
        # Read all entries
        entries = []
        with open(temp_index_path, "rb") as f:
            while True:
                tid_bytes = f.read(22)
                if not tid_bytes:
                    break
                index_bytes = f.read(4)
                track_id = tid_bytes.decode('ascii', 'ignore').rstrip('\0')
                vector_index = struct.unpack("<I", index_bytes)[0]
                entries.append((track_id, vector_index))
        
        # Sort by track ID
        entries.sort(key=lambda x: x[0])
        
        # Write sorted index
        for track_id, vector_index in entries:
            tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
            self.index_file.write(tid_bytes)
            self.index_file.write(struct.pack("<I", vector_index))
        
        # Validate index size
        expected_size = len(entries) * 26  # 22B + 4B
        actual_size = self.index_file.tell()
        if actual_size != expected_size:
            print(f"⚠️ Index file size mismatch: expected {expected_size} bytes, got {actual_size}")

    def _write_checksum(self):
        """Calculate CRC32 checksum and update file header"""
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
        """Finalize processing"""
        self._flush_buffers()
