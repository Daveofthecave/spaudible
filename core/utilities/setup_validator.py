# core/utilities/setup_validator.py
import json
from pathlib import Path
from config import PathConfig, EXPECTED_VECTORS
import os
import sys
import shutil
import struct
import heapq
from core.preprocessing.unified_vector_reader import UnifiedVectorReader
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter  # Add this import

def validate_vector_cache():
    """Validate vector cache completeness with exact vector count."""
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    
    # Check vector file existence
    if not vectors_path.exists():
        return False, "Vector file not found"
    
    # Validate vector file size
    vector_size = vectors_path.stat().st_size
    header_size = 16
    record_size = 104
    
    if vector_size < header_size:
        return False, f"Vector file too small: {vector_size} bytes"
    
    # Calculate number of vectors from file size
    num_vectors = (vector_size - header_size) // record_size
    
    # Verify exact vector count
    if num_vectors != EXPECTED_VECTORS:
        return False, (f"Incorrect vector count: expected {EXPECTED_VECTORS:,}, "
                      f"got {num_vectors:,}")
    
    # Check index file
    index_exists = index_path.exists()
    index_valid = False
    
    if index_exists:
        # Validate index file size
        index_size = index_path.stat().st_size
        expected_index_size = EXPECTED_VECTORS * 26  # 22B track ID + 4B index
        
        # Allow 1% variance
        if index_size > 0 and abs(index_size - expected_index_size) <= expected_index_size * 0.01:
            index_valid = True
    
    # Return status based on index validity
    if index_valid:
        return True, f"Valid vector cache with {num_vectors:,} tracks"
    else:
        status = "Vector file complete but index missing" if not index_exists else "Index file incomplete"
        return False, f"{status} ({num_vectors:,} vectors)"

def rebuild_index():
    """Robust index file rebuilding with progress reporting."""
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    
    print("\n  🔧 Rebuilding index file...")
    
    try:
        # Verify vectors file contains exactly EXPECTED_VECTORS
        reader = UnifiedVectorReader(vectors_path)
        total_vectors = reader.get_total_vectors()
        
        if total_vectors != EXPECTED_VECTORS:
            print(f"  ❗ Vector file has {total_vectors:,} vectors, expected {EXPECTED_VECTORS:,}")
            return False
        
        # Create temporary writer to handle sorting
        writer = UnifiedVectorWriter(PathConfig.VECTORS)
        writer._build_index_from_vectors(vectors_path, index_path)
        print("  ✅ Index file successfully rebuilt")
        return True
    except Exception as e:
        print(f"  ❗ Failed to rebuild index: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up any temporary files
        temp_dir = PathConfig.VECTORS / "temp_index"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def is_setup_complete():
    """Check if setup is complete and valid."""
    # Check required files exist
    required_files = [
        PathConfig.get_main_db(),
        PathConfig.get_audio_db(),
        PathConfig.get_vector_file(),
        PathConfig.get_index_file()
    ]
    
    if not all(file.exists() for file in required_files):
        return False
    
    # Validate vector cache
    valid, _ = validate_vector_cache()
    return valid
