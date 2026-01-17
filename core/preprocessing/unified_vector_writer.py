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

class UnifiedVectorWriter:
    """Optimized vector writer with batched I/O and robust uint16 packing."""
    
    HEADER_FORMAT = "<4sI8s"
    HEADER_SIZE = 16
    RECORD_SIZE = 104
    MAGIC = b"SPAU"
    WRITE_BATCH_SIZE = 500000
    
    # ===== DIAGNOSTIC CONFIGURATION =====
    # Set to True to log bad values (WARNING: slows preprocessing by ~10%)
    ENABLE_BAD_VALUE_LOGGING = False  # Set to False for production speed
    
    # Track types of bad values
    BAD_VALUE_SUMMARY = {
        'nan': 0,
        'inf': 0,
        'out_of_range': 0,
        'null_response': 0  # Tracks with no audio features
    }
    # =====================================
    
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
        
        self.temp_index_dir = self.output_dir / "temp_index"
        self.temp_index_dir.mkdir(exist_ok=True)
        
        # Diagnostic tracking
        self.bad_value_count = 0
        self.null_response_tracks = []  # Track IDs with null audio features
    
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
        
        # Print diagnostic summary if enabled
        if self.ENABLE_BAD_VALUE_LOGGING and self.bad_value_count > 0:
            print(f"\n" + "═"*65)
            print(f"  📊 Diagnostic Summary:")
            print(f"     Total bad vectors: {self.bad_value_count:,}")
            print(f"     NaN values: {self.BAD_VALUE_SUMMARY['nan']:,}")
            print(f"     Inf values: {self.BAD_VALUE_SUMMARY['inf']:,}")
            print(f"     Out of range: {self.BAD_VALUE_SUMMARY['out_of_range']:,}")
            print(f"     Null audio features: {self.BAD_VALUE_SUMMARY['null_response']:,}")
            print(f"  Details logged to: data/diagnostics/bad_values.log")
            print(f"  Track IDs logged to: data/diagnostics/null_response_tracks.txt")
            print(f"═"*65)
        
        return False
    
    def _write_header(self):
        header = struct.pack(self.HEADER_FORMAT, self.MAGIC, 1, b"\0" * 8)
        self.vector_file.write(header)
    
    def write_record(self, track_id: str, vector: np.ndarray, isrc: str = "", region: int = 7):
        """Store raw data."""
        self.track_ids.append(track_id)
        self.vectors_buffer.append(vector)
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
    
    # ===== ENHANCED DIAGNOSTIC METHOD =====
    def _log_bad_values(self, vectors_array: np.ndarray, track_ids_batch: List[str]):
        """
        Comprehensive logging of bad values for debugging.
        Tracks NaN, Inf, and out-of-range values separately.
        """
        if not self.ENABLE_BAD_VALUE_LOGGING:
            return
        
        # Create diagnostics directory
        diag_dir = Path("data/diagnostics")
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        values_log = diag_dir / "bad_values.log"
        null_log = diag_dir / "null_response_tracks.txt"
        
        batch_bad_count = 0
        
        # Open both files in append mode
        with open(values_log, "a") as val_f, open(null_log, "a") as null_f:
            for i in range(vectors_array.shape[0]):
                vec = vectors_array[i]
                track_id = track_ids_batch[i]
                
                # Check each dimension
                for dim in range(32):
                    val = vec[dim]
                    
                    # Check for NaN
                    if np.isnan(val):
                        val_f.write(f"Track {track_id}: dim {dim+1:02d} = nan (NaN/Inf)\n")
                        self.BAD_VALUE_SUMMARY['nan'] += 1
                        batch_bad_count += 1
                        
                        # Special handling for acousticness (dim 0)
                        if dim == 0:
                            null_f.write(f"{track_id}\n")
                            self.BAD_VALUE_SUMMARY['null_response'] += 1
                        break
                    
                    # Check for Inf
                    elif np.isinf(val):
                        val_f.write(f"Track {track_id}: dim {dim+1:02d} = inf (NaN/Inf)\n")
                        self.BAD_VALUE_SUMMARY['inf'] += 1
                        batch_bad_count += 1
                        break
                    
                    # Check for out-of-range (should never happen with clamping)
                    elif val < -1.0 or val > 1.0:
                        val_f.write(f"Track {track_id}: dim {dim+1:02d} = {val:.3f} (out of range)\n")
                        self.BAD_VALUE_SUMMARY['out_of_range'] += 1
                        batch_bad_count += 1
                        break
        
        self.bad_value_count += batch_bad_count
    
    def _flush_buffers(self):
        """Pack and write entire batch with robust uint16 conversion."""
        if not self.batch_count:
            return
        
        print(f"  🔍 Debug: Packing {self.batch_count:,} vectors...")
        pack_start = time.time()
        
        # Stack vectors
        vectors_array = np.stack(self.vectors_buffer)
        
        # ===== DIAGNOSTIC CALL =====
        if self.ENABLE_BAD_VALUE_LOGGING:
            print(f"  🔍 Debug: Running bad value diagnostic on batch...")
            self._log_bad_values(vectors_array, self.track_ids)
            print(f"  ✅ Debug: Diagnostic complete")
        # ===========================
        
        # Create structured record array
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
        
        # === ROBUST PACKING ===
        
        # Binary dims (mode, time signatures)
        binary_vals = np.zeros(self.batch_count, dtype=np.uint8)
        binary_vals |= (vectors_array[:, 9] >= 0.5).astype(np.uint8) << 0   # mode
        binary_vals |= (vectors_array[:, 11] >= 0.5).astype(np.uint8) << 1  # ts_4_4
        binary_vals |= (vectors_array[:, 12] >= 0.5).astype(np.uint8) << 2  # ts_3_4
        binary_vals |= (vectors_array[:, 13] >= 0.5).astype(np.uint8) << 3  # ts_5_4
        binary_vals |= (vectors_array[:, 14] >= 0.5).astype(np.uint8) << 4  # ts_other
        records['binary'] = binary_vals
        
        # Scaled dimensions - Clamp values to prevent overflow
        scaled_indices = [0,1,2,3,4,5,6,8,16] + list(range(19,32))
        
        for i, idx in enumerate(scaled_indices[:9]):
            vals = vectors_array[:, idx]
            
            # Clamp to valid range to prevent overflow
            vals = np.clip(vals, -1.0, 1.0)
            
            # Convert sentinel -1 to 0, then scale
            scaled_vals = np.where(vals == -1.0, 0.0, vals * 10000.0)
            
            # Ensure values fit in uint16 range
            scaled_vals = np.clip(scaled_vals, 0.0, 65535.0)
            
            records['scaled'][:, i] = scaled_vals.astype(np.uint16)
        
        for i, idx in enumerate(range(19, 32), start=9):
            vals = vectors_array[:, idx]
            
            # FIX: Same clamping for genre dimensions
            vals = np.clip(vals, -1.0, 1.0)
            scaled_vals = np.where(vals == -1.0, 0.0, vals * 10000.0)
            scaled_vals = np.clip(scaled_vals, 0.0, 65535.0)
            
            records['scaled'][:, i] = scaled_vals.astype(np.uint16)
        
        # FP32 dimensions (5 values)
        fp32_indices = [7, 10, 15, 17, 18]
        for i, idx in enumerate(fp32_indices):
            records['fp32'][:, i] = vectors_array[:, idx].astype(np.float32)
        
        # Validity masks - FIX: Check for NaN/Inf
        valid_mask = (vectors_array != -1.0) & np.isfinite(vectors_array)
        for j in range(32):
            records['mask'] |= (valid_mask[:, j].astype(np.uint32) << j)
        
        # Regions
        records['region'] = np.array(self.regions, dtype=np.uint8)
        
        # String cleaning - FIX: More robust ASCII cleaning
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
        """
        Fast ASCII string cleaning that returns a 1D array of fixed-length strings.
        FIX: Returns shape (n,) not (n, max_len) to match structured array dtype.
        """
        if not strings:
            return np.array([], dtype=f'S{max_len}')
        
        n = len(strings)
        # Create 2D buffer first
        buffer = np.zeros((n, max_len), dtype=np.uint8)
        
        for i, s in enumerate(strings):
            if not s:
                continue
            
            # Direct memory copy of ASCII bytes
            try:
                data = s.encode('ascii', 'ignore')[:max_len]
                buffer[i, :len(data)] = np.frombuffer(data, dtype=np.uint8)
            except Exception:
                # Fallback byte-by-byte
                for j, c in enumerate(s[:max_len]):
                    code = ord(c)
                    if 32 <= code < 127:
                        buffer[i, j] = code
        
        # FIX: Convert to 1D array of fixed-length strings
        # View the 2D buffer as a 1D array of S22/S12 strings
        return np.frombuffer(buffer.tobytes(), dtype=f'S{max_len}')
    
    def _write_index(self):
        """Merge all temp index files into final sorted index."""
        import struct
        
        temp_files = sorted(self.temp_index_dir.glob("temp_index_*.bin"),
                           key=lambda f: int(f.stem.split('_')[-1]))
        
        if not temp_files:
            return
        
        print(f"  🔍 Debug: Merging {len(temp_files)} temp index files...")
        merge_start = time.time()
        
        with open(self.index_path, "wb") as out_f:
            for temp_file in temp_files:
                with open(temp_file, "rb") as f:
                    shutil.copyfileobj(f, out_f)
        
        print(f"  ✅ Debug: Index merge completed in {time.time() - merge_start:.2f}s")
    
    def finalize(self):
        """Finalize processing."""
        self._flush_buffers()
        self._write_index()
    
    def get_stats(self):
        """Get performance statistics for debugging."""
        return {
            "total_records": self.total_records,
            "batches_flushed": self.total_records // self.WRITE_BATCH_SIZE,
            "buffer_utilization": (self.total_records % self.WRITE_BATCH_SIZE) / self.WRITE_BATCH_SIZE
        }
