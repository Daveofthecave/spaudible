# core/utilities/setup_validator.py
import json
import struct
import hashlib
from pathlib import Path
from config import PathConfig, EXPECTED_VECTORS

def validate_vector_cache(checksum_validation=True):
    """
    Validate vector cache with optional checksum.
    
    Args:
        checksum_validation: If True, perform full cryptographic checksum.
                           If False, only check file size (fast).
    """
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    
    # Fast checks first
    if not vectors_path.exists():
        return False, "Vector file not found"
    
    # Check vector file size
    vector_size = vectors_path.stat().st_size
    header_size = 16
    record_size = 104
    
    if vector_size < header_size:
        return False, f"Vector file too small: {vector_size} bytes"
    
    num_vectors = (vector_size - header_size) // record_size
    
    if num_vectors != EXPECTED_VECTORS:
        return False, (f"Incorrect vector count: expected {EXPECTED_VECTORS:,}, "
                      f"got {num_vectors:,}")
    
    # Skip expensive checksum if requested
    if not checksum_validation:
        # Only check if checksum field in header is non-zero (exists)
        try:
            with open(vectors_path, 'rb') as f:
                f.seek(8)  # Check if checksum bytes are present
                checksum_bytes = f.read(8)
                if checksum_bytes == b'\0' * 8:
                    return False, "Checksum not written (preprocessing incomplete)"
        except:
            pass
        
        # Fast path: Just verify index exists too
        if not index_path.exists():
            return False, "Index file not found"
        
        index_size = index_path.stat().st_size
        expected_index_size = EXPECTED_VECTORS * 26
        if abs(index_size - expected_index_size) > expected_index_size * 0.01:
            return False, "Index file size incorrect"
        
        return True, f"Valid vector cache (fast check) with {num_vectors:,} tracks"
    
    # Full validation with checksum
    try:
        with open(vectors_path, 'rb') as f:
            f.seek(4)  # Skip magic
            version = struct.unpack("<I", f.read(4))[0]
            stored_checksum = f.read(8)
            
            # Recompute checksum of data portion
            f.seek(header_size)
            hasher = hashlib.blake2b(digest_size=8)
            while True:
                chunk = f.read(10 * 1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
            
            computed_checksum = hasher.digest()
            
            if stored_checksum != computed_checksum:
                return False, "Checksum mismatch - vector file may be corrupted"
    except Exception as e:
        return False, f"Checksum validation failed: {e}"
    
    # Check index file
    if not index_path.exists():
        return False, "Index file not found"
    
    index_size = index_path.stat().st_size
    expected_index_size = EXPECTED_VECTORS * 26
    
    if abs(index_size - expected_index_size) > expected_index_size * 0.01:
        return False, f"Index file size incorrect"
    
    # Basic validation that index is sorted
    try:
        with open(index_path, 'rb') as f:
            prev_track_id = None
            for i in range(0, min(1000, EXPECTED_VECTORS)):
                f.seek(i * 26)
                track_id_bytes = f.read(22)
                if not track_id_bytes:
                    break
                track_id = track_id_bytes.decode('ascii', 'ignore').rstrip('\0')
                if prev_track_id and track_id < prev_track_id:
                    return False, "Index file is not properly sorted"
                prev_track_id = track_id
    except Exception as e:
        return False, f"Index validation failed: {e}"
    
    return True, f"Valid vector cache with {num_vectors:,} tracks"

def is_setup_complete():
    """Check if setup is complete using FAST validation."""
    # Check required files exist
    required_files = [
        PathConfig.get_main_db(),
        PathConfig.get_audio_db(),
        PathConfig.get_vector_file(),
        PathConfig.get_index_file()
    ]
    
    if not all(file.exists() for file in required_files):
        return False
    
    # Use FAST validation (skip checksum)
    valid, _ = validate_vector_cache(checksum_validation=False)
    return valid

def rebuild_index():
    """
    Rebuild sorted index from existing vectors file.
    Used when index is missing or corrupted.
    Returns True if successful.
    """
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    
    print("\n  🔧 Rebuilding index from vectors file...")
    
    try:
        # Verify vectors file
        if not vectors_path.exists():
            print(f"  ❗ Vector file not found: {vectors_path}")
            return False
        
        reader = UnifiedVectorReader(vectors_path)
        total_vectors = reader.get_total_vectors()
        
        print(f"  Found {total_vectors:,} vectors in file")
        
        # Create temp directory for sorting
        temp_dir = PathConfig.VECTORS / "temp_index"
        temp_dir.mkdir(exist_ok=True)
        
        # Extract and sort in chunks (same logic as preprocessing)
        chunk_size = 2_000_000
        chunk_files = []
        num_chunks = (total_vectors + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_vectors)
            num_in_chunk = end_idx - start_idx
            
            print(f"  Extracting chunk {chunk_idx+1}/{num_chunks}...")
            metadata = reader.extract_metadata_batch(start_idx, num_in_chunk)
            metadata.sort(key=lambda x: x[0])
            
            chunk_file = temp_dir / f"chunk_{chunk_idx:04d}.bin"
            with open(chunk_file, "wb") as f:
                for track_id, vector_index in metadata:
                    tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                    f.write(tid_bytes)
                    f.write(struct.pack("<I", vector_index))
            
            chunk_files.append(chunk_file)
        
        # Merge chunks
        print("  Merging sorted chunks...")
        _merge_sorted_chunks(chunk_files, index_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"  ✅ Index rebuilt with {total_vectors:,} entries")
        return True
        
    except Exception as e:
        print(f"  ❗ Failed to rebuild index: {e}")
        import traceback
        traceback.print_exc()
        return False


def _merge_sorted_chunks(chunk_files, output_path):
    """Helper to merge sorted chunk files."""
    import heapq
    import struct
    
    files = [open(f, "rb") for f in chunk_files]
    records = []
    
    for i, f in enumerate(files):
        data = f.read(26)
        if data:
            track_id = data[:22].decode('ascii', 'ignore').rstrip('\0')
            vector_index = struct.unpack("<I", data[22:26])[0]
            records.append((track_id, vector_index, i))
    
    heapq.heapify(records)
    
    with open(output_path, "wb") as out_file:
        while records:
            track_id, vector_index, file_idx = heapq.heappop(records)
            
            tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
            out_file.write(tid_bytes)
            out_file.write(struct.pack("<I", vector_index))
            
            next_data = files[file_idx].read(26)
            if next_data:
                next_track_id = next_data[:22].decode('ascii', 'ignore').rstrip('\0')
                next_vector_index = struct.unpack("<I", next_data[22:26])[0]
                heapq.heappush(records, (next_track_id, next_vector_index, file_idx))
    
    for f in files:
        f.close()
