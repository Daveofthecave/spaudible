# core/utilities/setup_validator.py
"""
Vector cache validation utilities for Spaudible.
Validates the unified binary format with 104-byte vectors and 26-byte index entries.
"""
import json
import struct
import hashlib
from pathlib import Path
from typing import Tuple
from config import PathConfig, EXPECTED_VECTORS, VECTOR_RECORD_SIZE, VECTOR_HEADER_SIZE

# Format-specific constants
INDEX_ENTRY_SIZE = 26     # 26 bytes per index (22B track_id + 4B vector_index)

def validate_vector_cache(checksum_validation: bool = True) -> Tuple[bool, str]:
    """
    Validate unified vector cache with optional cryptographic checksum.
    
    Args:
        checksum_validation: If True, performs full BLAKE2b checksum validation.
                           If False, only checks file sizes and basic structure (fast).
    
    Returns:
        Tuple of (is_valid, message)
    """
    
    # --- 1. Validate vector file ---
    vectors_path = PathConfig.get_vector_file()
    if not vectors_path.exists():
        return False, f"Vector file not found: {vectors_path}"
    
    vector_file_size = vectors_path.stat().st_size
    if vector_file_size < VECTOR_HEADER_SIZE:
        return False, f"Vector file too small: {vector_file_size} bytes"
    
    # Calculate expected vector count
    num_vectors = (vector_file_size - VECTOR_HEADER_SIZE) // VECTOR_RECORD_SIZE
    
    if num_vectors != EXPECTED_VECTORS:
        return False, (
            f"Incorrect vector count: expected {EXPECTED_VECTORS:,}, "
            f"got {num_vectors:,} (file size: {vector_file_size:,} bytes)"
        )
    
    # --- 2. Validate index file ---
    index_path = PathConfig.get_index_file()
    if not index_path.exists():
        return False, f"Index file not found: {index_path}"
    
    index_file_size = index_path.stat().st_size
    expected_index_size = EXPECTED_VECTORS * INDEX_ENTRY_SIZE
    
    # Allow 1% margin for processing variations
    size_diff = abs(index_file_size - expected_index_size)
    if size_diff > expected_index_size * 0.01:
        return False, (
            f"Index file size incorrect: expected {expected_index_size:,} bytes, "
            f"got {index_file_size:,} bytes (diff: {size_diff:,})"
        )
    
    # --- 3. Validate header checksum (optional) ---
    if not checksum_validation:
        # Fast path: just verify checksum field is populated
        try:
            with open(vectors_path, 'rb') as f:
                f.seek(8)  # Skip magic (4B) + version (4B)
                stored_checksum = f.read(8)
                if stored_checksum == b"\0" * 8:
                    return False, "Checksum not written (preprocessing incomplete)"
        except IOError as e:
            return False, f"Cannot read vector file header: {e}"
        
        return True, f"Valid vector cache (fast check) with {num_vectors:,} tracks"
    
    # --- 4. Full validation with cryptographic checksum ---
    try:
        with open(vectors_path, 'rb') as f:
            # Skip header
            f.seek(VECTOR_HEADER_SIZE)
            
            # Compute BLAKE2b checksum of data section
            hasher = hashlib.blake2b(digest_size=8)
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
            
            computed_checksum = hasher.digest()
        
        # Read stored checksum from header
        with open(vectors_path, 'rb') as f:
            f.seek(8)  # Position after magic + version
            stored_checksum = f.read(8)
        
        if stored_checksum != computed_checksum:
            return False, (
                f"Checksum mismatch: vector file may be corrupted. "
                f"Stored: {stored_checksum.hex()}, Computed: {computed_checksum.hex()}"
            )
    
    except Exception as e:
        return False, f"Checksum validation failed: {e}"
    
    # --- 5. Validate index sorting (sample-based) ---
    try:
        with open(index_path, 'rb') as f:
            # Sample random entries and verify they're in order
            prev_track_id = None
            
            # Check 3 samples: start, middle, end
            sample_positions = [0, EXPECTED_VECTORS // 2, EXPECTED_VECTORS - 1]
            
            for pos in sample_positions:
                f.seek(pos * INDEX_ENTRY_SIZE)
                track_id_bytes = f.read(22)
                track_id = track_id_bytes.decode('ascii', errors='ignore').rstrip('\0')
                
                if prev_track_id is not None and track_id < prev_track_id:
                    return False, f"Index not sorted at position {pos}: {track_id} < {prev_track_id}"
                
                if not track_id:  # Empty track ID
                    return False, f"Empty track ID at position {pos}"
                
                prev_track_id = track_id
        
    except Exception as e:
        return False, f"Index integrity check failed: {e}"
    
    return True, f"Valid vector cache with {num_vectors:,} tracks (checksum verified)"

def is_setup_complete() -> bool:
    """
    Fast check if setup is complete by verifying all required files exist
    and have reasonable sizes.
    """
    required_files = [
        PathConfig.get_main_db(),
        PathConfig.get_audio_db(),
        PathConfig.get_vector_file(),
        PathConfig.get_index_file(),
    ]
    
    if not all(file.exists() for file in required_files):
        return False
    
    # Quick size check (no checksum)
    valid, _ = validate_vector_cache(checksum_validation=False)
    return valid

def rebuild_index() -> bool:
    """
    Rebuild sorted index from existing vectors file if index is missing or corrupted.
    Returns True if rebuild successful, False otherwise.
    """
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    
    print("\n  🔧 Rebuilding index from vectors file...")
    
    try:
        # Verify vectors file exists and is valid
        if not vectors_path.exists():
            print(f"  ❗ Vector file not found: {vectors_path}")
            return False
        
        valid, msg = validate_vector_cache(checksum_validation=False)
        if not valid:
            print(f"  ❗ Vector file validation failed: {msg}")
            return False
        
        from core.preprocessing.unified_vector_reader import UnifiedVectorReader
        
        reader = UnifiedVectorReader(vectors_path)
        total_vectors = reader.get_total_vectors()
        print(f"  Found {total_vectors:,} vectors in file")
        
        # Create temp directory for sorting
        temp_dir = PathConfig.VECTORS / "temp_index_rebuild"
        temp_dir.mkdir(exist_ok=True)
        
        # Extract metadata in chunks and sort
        chunk_size = 2_000_000  # 2M entries per chunk
        num_chunks = (total_vectors + chunk_size - 1) // chunk_size
        chunk_files = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_vectors)
            num_in_chunk = end_idx - start_idx
            
            print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} ({num_in_chunk:,} vectors)...")
            
            # Extract track_id → vector_index pairs
            metadata = reader.extract_metadata_batch(start_idx, num_in_chunk)
            
            # Sort by track ID (byte comparison)
            metadata.sort(key=lambda x: x[0])
            
            # Write sorted chunk to temp file
            chunk_file = temp_dir / f"chunk_{chunk_idx:04d}.bin"
            with open(chunk_file, "wb") as cf:
                for track_id, vector_index in metadata:
                    # Track ID (22 bytes, null-padded ASCII)
                    tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                    cf.write(tid_bytes)
                    
                    # Vector index (4 bytes, little-endian)
                    cf.write(struct.pack("<I", vector_index))
            
            chunk_files.append(chunk_file)
        
        # Merge sorted chunks using heap sort
        print("  Merging sorted chunks...")
        _merge_sorted_chunks(chunk_files, index_path)
        
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"  ✅ Index rebuilt successfully with {total_vectors:,} entries")
        return True
        
    except Exception as e:
        print(f"  ❗ Failed to rebuild index: {e}")
        import traceback
        traceback.print_exc()
        return False

def _merge_sorted_chunks(chunk_files: list, output_path: Path):
    """
    Merge sorted chunk files using heap sort.
    
    Args:
        chunk_files: List of sorted chunk file paths
        output_path: Path for final merged index file
    """
    import heapq
    
    # Open all chunk files
    files = [open(f, "rb") for f in chunk_files]
    records = []
    
    # Read first record from each file
    for i, f in enumerate(files):
        data = f.read(26)
        if data:
            track_id = data[:22].decode('ascii', 'ignore').rstrip('\0')
            vector_index = struct.unpack("<I", data[22:26])[0]
            records.append((track_id, vector_index, i))
    
    heapq.heapify(records)
    
    # Merge into final index
    with open(output_path, "wb") as out_file:
        while records:
            track_id, vector_index, file_idx = heapq.heappop(records)
            
            # Write merged record
            tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
            out_file.write(tid_bytes)
            out_file.write(struct.pack("<I", vector_index))
            
            # Read next record from the same file
            next_data = files[file_idx].read(26)
            if next_data:
                next_track_id = next_data[:22].decode('ascii', 'ignore').rstrip('\0')
                next_vector_index = struct.unpack("<I", next_data[22:26])[0]
                heapq.heappush(records, (next_track_id, next_vector_index, file_idx))
        
        print(f"    Merged {len(chunk_files)} chunks into final index")
    
    # Close all chunk files
    for f in files:
        f.close()
