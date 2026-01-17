# core/preprocessing/unified_vector_writer.py
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple
import os
import gc
import shutil
import time
import heapq
from config import PathConfig, EXPECTED_VECTORS
from core.utilities.region_utils import REGION_MAPPING

class UnifiedVectorWriter:
    """Optimized vector writer with batched I/O and vectorized operations."""
    
    HEADER_FORMAT = "<4sI8s"
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
        
        # SIMPLE buffers - just store raw data
        self.track_ids = []
        self.vectors_buffer = []  # Store raw 32D vectors
        self.isrcs = []
        self.regions = []
        self.batch_count = 0
        
        self.temp_index_dir = self.output_dir / "temp_index"
        self.temp_index_dir.mkdir(exist_ok=True)
    
    def __enter__(self):
        """Open files for writing."""
        gc.disable()
        
        if self.resume_from > 0 and self.vectors_path.exists():
            self.vector_file = open(self.vectors_path, "ab")
        else:
            self.vector_file = open(self.vectors_path, "wb")
            self._write_header()
        
        temp_index_path = self.temp_index_dir / f"temp_index_{self.resume_from}.bin"
        self.temp_index_file = open(temp_index_path, "ab" if self.resume_from > 0 else "wb")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize files."""
        try:
            if exc_type is None:
                self._flush_buffers()
                self._write_index()
        finally:
            if self.vector_file:
                self.vector_file.close()
            if self.temp_index_file:
                self.temp_index_file.close()
            gc.enable()
            try:
                shutil.rmtree(self.temp_index_dir)
            except:
                pass
        return False
    
    def _write_header(self):
        header = struct.pack(self.HEADER_FORMAT, self.MAGIC, 1, b"\0" * 8)
        self.vector_file.write(header)
    
    def write_record(self, track_id: str, vector: np.ndarray, isrc: str = "", region: int = 7):
        """Store raw data (NO packing here!)."""
        self.track_ids.append(track_id)
        self.vectors_buffer.append(vector)  # Store raw vector
        self.isrcs.append(isrc)
        self.regions.append(region)
        
        # Write temp index
        tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
        self.temp_index_file.write(tid_bytes)
        self.temp_index_file.write(struct.pack("<I", self.total_records))
        
        self.total_records += 1
        self.batch_count += 1
        
        if self.batch_count >= self.WRITE_BATCH_SIZE:
            self._flush_buffers()
    
    def _flush_buffers(self):
        """OPTIMIZATION: Vectorized packing of ENTIRE batch at once."""
        if not self.batch_count:
            return
        
        print(f"  🔍 Debug: Packing {self.batch_count:,} vectors...")
        pack_start = time.time()
        
        # Stack all vectors into a single array (batch_count x 32)
        vectors_array = np.stack(self.vectors_buffer)
        
        # Create structured array for batch
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
        
        # === VECTORIZED packing (no Python loops!) ===
        
        # Binary dims (mode, time signatures)
        binary_vals = np.zeros(self.batch_count, dtype=np.uint8)
        binary_vals |= (vectors_array[:, 9] >= 0.5).astype(np.uint8) << 0   # mode
        binary_vals |= (vectors_array[:, 11] >= 0.5).astype(np.uint8) << 1  # ts_4_4
        binary_vals |= (vectors_array[:, 12] >= 0.5).astype(np.uint8) << 2 # ts_3_4
        binary_vals |= (vectors_array[:, 13] >= 0.5).astype(np.uint8) << 3 # ts_5_4
        binary_vals |= (vectors_array[:, 14] >= 0.5).astype(np.uint8) << 4 # ts_other
        records['binary'] = binary_vals
        
        # Scaled dims (22 values)
        scaled_indices = [0,1,2,3,4,5,6,8,16] + list(range(19,32))
        for i, idx in enumerate(scaled_indices[:9]):
            vals = vectors_array[:, idx]
            vals = np.where(vals == -1.0, 0, vals * 10000)
            records['scaled'][:, i] = vals.astype(np.uint16)
        
        for i, idx in enumerate(range(19, 32), start=9):
            vals = vectors_array[:, idx]
            vals = np.where(vals == -1.0, 0, vals * 10000)
            records['scaled'][:, i] = vals.astype(np.uint16)
        
        # FP32 dims (5 values)
        fp32_indices = [7, 10, 15, 17, 18]
        for i, idx in enumerate(fp32_indices):
            records['fp32'][:, i] = vectors_array[:, idx].astype(np.float32)
        
        # Validity masks
        valid_mask = (vectors_array != -1.0) & ~np.isnan(vectors_array)
        for j in range(32):
            records['mask'] |= (valid_mask[:, j].astype(np.uint32) << j)
        
        # Regions
        records['region'] = np.array(self.regions, dtype=np.uint8)
        
        # String cleaning (still needed, but now vectorized)
        print(f"  🔍 Debug: Cleaning {self.batch_count} strings...")
        clean_start = time.time()
        records['track_id'] = self._clean_strings_batch(self.track_ids, 22)
        records['isrc'] = self._clean_strings_batch(self.isrcs, 12)
        print(f"  ✅ Debug: Strings cleaned in {time.time() - clean_start:.2f}s")
        
        # Single write
        print(f"  🔍 Debug: Writing to disk...")
        write_start = time.time()
        self.vector_file.write(records.tobytes())
        print(f"  ✅ Debug: Written in {time.time() - write_start:.2f}s")
        
        # Clear
        self._clear_buffers()
        
        total_time = time.time() - pack_start
        print(f"  ✅ Debug: Total flush: {total_time:.2f}s")
    
    def _clear_buffers(self):
        self.batch_count = 0
        self.track_ids.clear()
        self.vectors_buffer.clear()
        self.isrcs.clear()
        self.regions.clear()
    
    def _clean_strings_batch(self, strings: List[str], max_len: int) -> np.ndarray:
        """Fast batch string cleaning."""
        if not strings:
            return np.array([], dtype=f'S{max_len}')
        
        n = len(strings)
        result = np.empty(n, dtype=f'S{max_len}')
        
        for i, s in enumerate(strings):
            if not s:
                result[i] = b''
                continue
            
            cleaned = bytearray(max_len)
            j = 0
            for c in s:
                code = ord(c)
                if 32 <= code < 127 and j < max_len:
                    cleaned[j] = code
                    j += 1
            
            while j < max_len:
                cleaned[j] = 0
                j += 1
            
            result[i] = bytes(cleaned)
        
        return result
    
    def finalize(self):
        self._flush_buffers()
        self._write_index()
    
    def _write_index(self):
        """Build final sorted index file."""
        import struct
        
        temp_files = list(self.temp_index_dir.glob("temp_index_*.bin"))
        if not temp_files:
            return
        
        # Sort by filename to maintain order
        temp_files.sort(key=lambda f: int(f.stem.split('_')[-1]))
        
        # Merge all temp files into final index
        print(f"  🔍 Debug: Merging {len(temp_files)} temp index files...")
        merge_start = time.time()
        
        with open(self.index_path, "wb") as out_f:
            for temp_file in temp_files:
                with open(temp_file, "rb") as f:
                    shutil.copyfileobj(f, out_f)
        
        print(f"  ✅ Debug: Index merge completed in {time.time() - merge_start:.2f}s")
        
        # Cleanup handled by __exit__
    
    def finalize(self):
        """Finalize processing"""
        self._flush_buffers()
        self._write_index()
    
    def get_stats(self):
        """Get performance statistics for debugging."""
        return {
            "total_records": self.total_records,
            "batches_flushed": self.total_records // self.WRITE_BATCH_SIZE,
            "buffer_utilization": (self.total_records % self.WRITE_BATCH_SIZE) / self.WRITE_BATCH_SIZE
        }
