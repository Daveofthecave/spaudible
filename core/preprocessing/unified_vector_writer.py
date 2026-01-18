# core/preprocessing/unified_vector_writer.py
import struct
import numpy as np
from pathlib import Path
from typing import List
import os
import gc
import shutil
import time
from config import PathConfig, EXPECTED_VECTORS
import hashlib

class UnifiedVectorWriter:
    """Optimized vector writer with new 104-byte record format."""
    
    HEADER_FORMAT = "<4sI8s"  # Magic (4B) + Version (4B) + Checksum placeholder (8B)
    HEADER_SIZE = 16
    RECORD_SIZE = 104
    MAGIC = b"SPAU"
    WRITE_BATCH_SIZE = 500000
    
    def __init__(self, output_dir: Path, resume_from=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.vectors_path = self.output_dir / "track_vectors.bin"
        self.index_path = self.output_dir / "track_index.bin"
        self.resume_from = resume_from
        self.total_records = resume_from
        
        # Buffers
        self.track_ids = []
        self.vectors_buffer = []
        self.isrcs = []
        self.regions = []
        self.batch_count = 0
        
        # Temp index directory - will NOT be deleted automatically
        self.temp_index_dir = self.output_dir / "temp_index"
        self.temp_index_dir.mkdir(exist_ok=True)
        
        # For checksum calculation
        self.checksum = hashlib.blake2b(digest_size=8)
    
    def __enter__(self):
        """Open files for writing."""
        gc.disable()
        
        if self.resume_from > 0 and self.vectors_path.exists():
            self.vector_file = open(self.vectors_path, "ab")
        else:
            self.vector_file = open(self.vectors_path, "wb")
            self._write_header()
        
        # Create initial temp index file
        temp_index_path = self.temp_index_dir / f"temp_index_{self.resume_from}.bin"
        self.temp_index_file = open(temp_index_path, "ab" if self.resume_from > 0 else "wb")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize files - temp index dir is NOT deleted."""
        try:
            if exc_type is None:
                self._flush_buffers()
                # DO NOT write final index here - will be sorted separately
                # DO NOT delete temp dir - needed for sorting step
        finally:
            if self.vector_file:
                self.vector_file.close()
            if self.temp_index_file:
                self.temp_index_file.close()
            gc.enable()
        return False
    
    def _write_header(self):
        """Write header with placeholder checksum."""
        header = struct.pack(self.HEADER_FORMAT, self.MAGIC, 1, b"\0" * 8)
        self.vector_file.write(header)
    
    def write_record(self, track_id: str, vector: np.ndarray, isrc: str = "", region: int = 7):
        """Write a record - format: 65B vector + 4B mask + 1B region + 12B ISRC + 22B track_id."""
        self.track_ids.append(track_id)
        self.vectors_buffer.append(vector)
        self.isrcs.append(isrc)
        self.regions.append(region)
        
        # Write unsorted temp index entry (26 bytes: 22B track_id + 4B vector_index)
        tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
        self.temp_index_file.write(tid_bytes)
        self.temp_index_file.write(struct.pack("<I", self.total_records))
        
        self.total_records += 1
        self.batch_count += 1
        
        if self.batch_count >= self.WRITE_BATCH_SIZE:
            self._flush_buffers()
    
    def _flush_buffers(self):
        """Pack and write batch to disk."""
        if not self.batch_count:
            return
        
        # Stack vectors
        vectors_array = np.stack(self.vectors_buffer)
        
        # Create structured record
        dtype = np.dtype([
            ('binary', np.uint8),
            ('scaled', np.uint16, (22,)),
            ('fp32', np.float32, (5,)),
            ('mask', np.uint32),
            ('region', np.uint8),
            ('isrc', 'S12'),
            ('track_id', 'S22')
        ])
        
        records = np.zeros(self.batch_count, dtype=dtype)
        
        # Pack binary dimensions (mode + time signatures)
        binary_vals = np.zeros(self.batch_count, dtype=np.uint8)
        binary_vals |= (vectors_array[:, 9] >= 0.5).astype(np.uint8) << 0   # mode
        binary_vals |= (vectors_array[:, 11] >= 0.5).astype(np.uint8) << 1  # ts_4_4
        binary_vals |= (vectors_array[:, 12] >= 0.5).astype(np.uint8) << 2  # ts_3_4
        binary_vals |= (vectors_array[:, 13] >= 0.5).astype(np.uint8) << 3  # ts_5_4
        binary_vals |= (vectors_array[:, 14] >= 0.5).astype(np.uint8) << 4  # ts_other
        records['binary'] = binary_vals
        
        # Pack scaled dimensions (uint16 with 0.0001 precision)
        scaled_indices = [0,1,2,3,4,5,6,8,16] + list(range(19,32))
        for i, idx in enumerate(scaled_indices):
            vals = np.clip(vectors_array[:, idx], -1.0, 1.0)
            scaled = np.where(vals == -1.0, 0, (vals * 10000).astype(np.uint16))
            records['scaled'][:, i] = np.clip(scaled, 0, 65535)
        
        # Pack FP32 dimensions
        fp32_indices = [7, 10, 15, 17, 18]
        for i, idx in enumerate(fp32_indices):
            records['fp32'][:, i] = vectors_array[:, idx].astype(np.float32)
        
        # Validity mask
        valid_mask = (vectors_array != -1.0) & np.isfinite(vectors_array)
        for j in range(32):
            records['mask'] |= (valid_mask[:, j].astype(np.uint32) << j)
        
        # Region
        records['region'] = np.array(self.regions, dtype=np.uint8)
        
        # Strings
        records['track_id'] = self._clean_strings_batch(self.track_ids, 22)
        records['isrc'] = self._clean_strings_batch(self.isrcs, 12)
        
        # Write
        data_bytes = records.tobytes()
        self.vector_file.write(data_bytes)
        
        # Update checksum
        self.checksum.update(data_bytes)
        
        # Clear buffers
        self._clear_buffers()
    
    def _clear_buffers(self):
        self.batch_count = 0
        self.track_ids.clear()
        self.vectors_buffer.clear()
        self.isrcs.clear()
        self.regions.clear()
    
    def _clean_strings_batch(self, strings: List[str], max_len: int) -> np.ndarray:
        """Fast ASCII string cleaning."""
        if not strings:
            return np.array([], dtype=f'S{max_len}')
        
        n = len(strings)
        buffer = np.zeros((n, max_len), dtype=np.uint8)
        
        for i, s in enumerate(strings):
            if s:
                try:
                    data = s.encode('ascii', 'ignore')[:max_len]
                    buffer[i, :len(data)] = np.frombuffer(data, dtype=np.uint8)
                except:
                    for j, c in enumerate(s[:max_len]):
                        code = ord(c)
                        if 32 <= code < 127:
                            buffer[i, j] = code
        
        return np.frombuffer(buffer.tobytes(), dtype=f'S{max_len}')
    
    def finalize(self):
        """Finalize vector file."""
        self._flush_buffers()
    
    def get_temp_index_dir(self):
        """Return temp index directory for sorting step."""
        return self.temp_index_dir
    
    def update_header_checksum(self):
        """Compute checksum of ENTIRE data section and write to header."""
        # print("  🔍 Computing final checksum...")
        import hashlib
        
        hasher = hashlib.blake2b(digest_size=8)
        
        with open(self.vectors_path, 'rb') as f:
            # Skip header
            f.seek(self.HEADER_SIZE)
            
            # Read in chunks to handle large files
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        
        # Write checksum to header (bytes 8-15)
        with open(self.vectors_path, 'r+b') as f:
            f.seek(8)  # Position after magic (4B) and version (4B)
            f.write(hasher.digest()[:8])
