# core/preprocessing/unified_vector_writer.py
import struct
import numpy as np
from pathlib import Path
from typing import Tuple, List
import zlib
import os
import tempfile
import shutil
import sys
import heapq
from config import PathConfig, EXPECTED_VECTORS
from core.utilities.region_utils import REGION_MAPPING
from core.preprocessing.unified_vector_reader import UnifiedVectorReader

class UnifiedVectorWriter:
    """Optimized vector writer with batched I/O and efficient indexing."""
    
    # File header format (16 bytes)
    HEADER_FORMAT = "<4sI8s"  # Magic, version, checksum placeholder
    HEADER_SIZE = 16
    
    # Record format (104 bytes)
    RECORD_SIZE = 104
    
    # Magic number for file identification
    MAGIC = b"SPAU"  # Spaudible Packed Vector
    
    # Batch size for vector packing
    WRITE_BATCH_SIZE = 25_000 # Was 500_000
    
    def __init__(self, output_dir: Path, resume_from=0):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.vectors_path = self.output_dir / "track_vectors.bin"
        self.index_path = self.output_dir / "track_index.bin"
        
        # Resume state
        self.resume_from = resume_from
        self.total_records = resume_from
        
        # File handles
        self.vector_file = None
        self.index_file = None
        
        # Buffers
        self.track_ids = []
        self.vectors = []
        self.isrcs = []
        self.regions = []
        self.batch_count = 0
        
        # Temporary index storage
        self.temp_index_dir = self.output_dir / "temp_index"
        self.temp_index_dir.mkdir(exist_ok=True)
        self.temp_index_file = None

    def __enter__(self):
        """Open files for writing in appropriate mode."""
        # Open vector file in append mode if resuming
        if self.resume_from > 0 and self.vectors_path.exists():
            self.vector_file = open(self.vectors_path, "ab")
        else:
            self.vector_file = open(self.vectors_path, "wb")
            self._write_header()
        
        # Open temporary index file
        temp_index_path = self.temp_index_dir / f"temp_index_{self.resume_from}.bin"
        self.temp_index_file = open(temp_index_path, "ab" if self.resume_from > 0 else "wb")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize files and clean up."""
        try:
            if exc_type is None:
                # Flush any remaining records
                self._flush_buffers()
                self._write_index()
        finally:
            # Close files
            if self.vector_file:
                self.vector_file.close()
            if self.temp_index_file:
                self.temp_index_file.close()
            
            # Clean up temporary directory
            try:
                shutil.rmtree(self.temp_index_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temp directory: {e}")
        
        return False

    def _write_index(self):
        """Write the temporary index to final index file."""
        if not self.temp_index_dir.exists():
            return
            
        temp_files = list(self.temp_index_dir.glob("temp_index_*.bin"))
        if not temp_files:
            return
        
        # Merge all temporary index files
        combined_path = self.temp_index_dir / "combined_temp_index.bin"
        with open(combined_path, "wb") as combined:
            for temp_file in temp_files:
                with open(temp_file, "rb") as f:
                    shutil.copyfileobj(f, combined)
                temp_file.unlink()
        
        # Sort and deduplicate index
        self._sort_index(combined_path, self.index_path)
        combined_path.unlink()

    def _sort_index(self, input_path: Path, output_path: Path):
        """Sort temporary index file by track_id for binary search."""
        records = []
        
        # Read all index entries
        with open(input_path, "rb") as f:
            while True:
                tid_bytes = f.read(22)
                if len(tid_bytes) < 22:
                    break
                index_bytes = f.read(4)
                track_id = tid_bytes.decode('ascii', 'ignore').rstrip('\0')
                vector_index = struct.unpack("<I", index_bytes)[0]
                records.append((track_id, vector_index))
        
        # Sort by track_id
        records.sort(key=lambda x: x[0])
        
        # Write sorted index
        with open(output_path, "wb") as f:
            for track_id, vector_index in records:
                tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                f.write(tid_bytes)
                f.write(struct.pack("<I", vector_index))

    def finalize(self):
        """Finalize processing and merge index files."""
        # Merge all temporary index files
        temp_files = list(self.temp_index_dir.glob("temp_index_*.bin"))
        if temp_files:
            # Create combined temporary file
            combined_path = self.temp_index_dir / "combined_temp_index.bin"
            with open(combined_path, "wb") as combined_file:
                for temp_file in temp_files:
                    with open(temp_file, "rb") as f:
                        shutil.copyfileobj(f, combined_file)
                    temp_file.unlink()
            
            # Sort and write final index
            self._sort_index(combined_path, self.index_path)
            combined_path.unlink()

    def _write_header(self):
        """Write file header with magic and version"""
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
        """Add record to batch buffer"""
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
        binary_bytes = self._pack_binary_dims(vectors)
        
        # Pack scaled dimensions
        scaled_dims = self._pack_scaled_dims(vectors)
        
        # Pack FP32 dimensions
        fp32_dims = self._pack_fp32_dims(vectors)
        
        # Generate validity masks
        validity_masks = self._get_validity_masks(vectors)
        
        # Precompute clean ISRCs and track IDs
        clean_isrcs = [self._clean_isrc(isrc) for isrc in self.isrcs]
        clean_track_ids = [self._clean_track_id(tid) for tid in self.track_ids]
        
        # Write records in bulk
        records = bytearray()
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
            records.extend(record)
        
        # Write all records at once
        self.vector_file.write(records)
        
        # Clear buffers
        self.track_ids.clear()
        self.vectors.clear()
        self.isrcs.clear()
        self.regions.clear()
        self.batch_count = 0
        
        # Flush to disk
        self.vector_file.flush()
        self.temp_index_file.flush()
    
    def _pack_binary_dims(self, vectors: np.ndarray) -> np.ndarray:
        """Pack binary dimensions into a single byte per vector"""
        # Extract relevant dimensions
        mode = vectors[:, 9]
        time_sig_4_4 = vectors[:, 11]
        time_sig_3_4 = vectors[:, 12]
        time_sig_5_4 = vectors[:, 13]
        time_sig_other = vectors[:, 14]
        
        # Convert to binary flags
        binary_bytes = (
            (mode >= 0.5).astype(np.uint8) << 0 |
            (time_sig_4_4 >= 0.5).astype(np.uint8) << 1 |
            (time_sig_3_4 >= 0.5).astype(np.uint8) << 2 |
            (time_sig_5_4 >= 0.5).astype(np.uint8) << 3 |
            (time_sig_other >= 0.5).astype(np.uint8) << 4
        )
        
        return binary_bytes

    def _pack_scaled_dims(self, vectors: np.ndarray) -> np.ndarray:
        """Pack scaled dimensions into uint16 values with NaN handling"""
        # First 9 scaled dimensions
        scaled_indices = [0, 1, 2, 3, 4, 5, 6, 8, 16]
        scaled_part1 = vectors[:, scaled_indices]
        
        # Meta-genres
        scaled_part2 = vectors[:, 19:32]
        
        # Combine and scale
        scaled_dims = np.hstack([scaled_part1, scaled_part2])
        
        # Replace NaNs with 0
        scaled_dims = np.where(np.isnan(scaled_dims), 0.0, scaled_dims)
        
        # Scale and convert to uint16
        scaled_dims = np.where(scaled_dims == -1.0, 0.0, scaled_dims)
        scaled_dims = (scaled_dims * 10000).astype(np.uint16)
        
        return scaled_dims
    
    def _pack_fp32_dims(self, vectors: np.ndarray) -> np.ndarray:
        """Pack FP32 dimensions with NaN handling"""
        fp32_indices = [7, 10, 15, 17, 18]
        fp32_dims = vectors[:, fp32_indices]
        
        # Replace NaNs with -1.0
        fp32_dims = np.where(np.isnan(fp32_dims), -1.0, fp32_dims)
        return fp32_dims

    def _get_validity_masks(self, vectors: np.ndarray) -> np.ndarray:
        """Generate validity masks for each vector with NaN handling"""
        # Mark NaN values as invalid
        invalid_matrix = np.isnan(vectors)
        valid_matrix = np.logical_and(vectors != -1.0, ~invalid_matrix)
        
        validity_masks = np.zeros(vectors.shape[0], dtype=np.uint32)
        
        for j in range(32):
            validity_masks |= (valid_matrix[:, j].astype(np.uint32) << j)
        
        return validity_masks

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

    def _build_index_from_vectors(self, vectors_path, index_path):
        """Build index file from completed vectors file with progress reporting."""
        # Initialize reader
        reader = UnifiedVectorReader(vectors_path)
        total_vectors = reader.get_total_vectors()
        
        # Verify vector count
        if total_vectors != EXPECTED_VECTORS:
            raise ValueError(f"Expected {EXPECTED_VECTORS:,} vectors, found {total_vectors:,}")
        
        # Create temporary directory for sorting
        temp_dir = self.output_dir / "temp_index"
        temp_dir.mkdir(exist_ok=True)
        
        # Process in chunks
        chunk_size = 1_000_000
        chunk_files = []
        num_chunks = (total_vectors + chunk_size - 1) // chunk_size
        
        print(f"  Processing {total_vectors:,} vectors in {num_chunks} chunks")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_vectors)
            num_vectors = end_idx - start_idx
            
            # Print progress
            print(f"  Chunk {chunk_idx+1}/{num_chunks}: vectors {start_idx:,}-{end_idx-1:,}")
            
            # Read metadata for chunk
            metadata = reader.get_vector_metadata_batch(start_idx, num_vectors)
            
            # Sort this chunk
            metadata.sort(key=lambda x: x[0])
            
            # Write sorted chunk to temporary file
            chunk_file = temp_dir / f"chunk_{chunk_idx}.bin"
            with open(chunk_file, "wb") as f:
                for track_id, vector_index in metadata:
                    tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                    f.write(tid_bytes)
                    f.write(struct.pack("<I", vector_index))
            
            chunk_files.append(chunk_file)
        
        # Merge sorted chunks
        print("  Merging sorted chunks...")
        self._merge_chunks(chunk_files, index_path)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        print(f"  ✅ Index file created with {total_vectors:,} entries")

    def _merge_chunks(self, chunk_files, output_path):
        """Merge sorted chunk files into final index file."""
        # Open all chunk files
        files = [open(f, "rb") for f in chunk_files]
        records = [None] * len(files)
        
        # Initialize the records for each file
        for i, f in enumerate(files):
            tid_bytes = f.read(22)
            if tid_bytes:
                index_bytes = f.read(4)
                track_id = tid_bytes.decode('ascii', 'ignore').rstrip('\0')
                vector_index = struct.unpack("<I", index_bytes)[0]
                records[i] = (track_id, vector_index, i)
        
        # Use a heap to merge the records
        heap = []
        for i, rec in enumerate(records):
            if rec is not None:
                heapq.heappush(heap, (rec[0], rec[1], rec[2]))
        
        # Open output file
        with open(output_path, "wb") as out_file:
            processed = 0
            while heap:
                track_id, vector_index, file_idx = heapq.heappop(heap)
                
                # Write to final index
                tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                out_file.write(tid_bytes)
                out_file.write(struct.pack("<I", vector_index))
                
                # Print progress periodically
                processed += 1
                if processed % 5_000_000 == 0:
                    print(f"  Merged {processed:,} records...")
                
                # Get next record from the same file
                tid_bytes = files[file_idx].read(22)
                if tid_bytes:
                    index_bytes = files[file_idx].read(4)
                    track_id = tid_bytes.decode('ascii', 'ignore').rstrip('\0')
                    vector_index = struct.unpack("<I", index_bytes)[0]
                    heapq.heappush(heap, (track_id, vector_index, file_idx))
        
        # Close all files
        for f in files:
            f.close()

    def finalize(self):
        """Finalize processing"""
        self._flush_buffers()
        self._write_index()
