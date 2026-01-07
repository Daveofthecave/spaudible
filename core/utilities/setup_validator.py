# core/utilities/setup_validator.py
import json
from pathlib import Path
from config import PathConfig

def validate_vector_cache():
    """Comprehensive validation of vector cache completeness."""
    vectors_path = PathConfig.get_vector_file()
    metadata_path = PathConfig.get_metadata_file()
    
    # Check file existence
    if not vectors_path.exists():
        return False, "Vector file not found"
    if not metadata_path.exists():
        return False, "Metadata file not found"
    
    # Validate vector file size
    vector_size = vectors_path.stat().st_size
    if vector_size % 128 != 0:
        return False, f"Vector file size {vector_size} not divisible by 128 bytes"
    
    num_vectors = vector_size // 128
    
    # Validate metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata_count = metadata.get('total_tracks', 0)
        if metadata_count <= 0:
            return False, f"Invalid track count in metadata: {metadata_count}"
            
        # Allow 5% variance from metadata claim
        if abs(num_vectors - metadata_count) > metadata_count * 0.05:
            return False, (f"Vector count mismatch: file has {num_vectors:,} vectors, "
                          f"metadata claims {metadata_count:,}")
        
        # Minimum track count threshold (95% of 256M)
        min_tracks = 256_000_000
        if metadata_count < min_tracks:
            return False, (f"Insufficient tracks processed: {metadata_count:,} < {min_tracks:,.0f}")
            
        return True, f"Valid vector cache with {metadata_count:,} tracks"
        
    except Exception as e:
        return False, f"Error validating metadata: {str(e)}"

def is_setup_complete():
    """Comprehensive check if setup is complete and valid."""
    # Check required files exist
    required_files = PathConfig.all_required_files()
    if not all(file.exists() for file in required_files):
        return False
    
    # Validate vector cache completeness
    valid, _ = validate_vector_cache()
    return valid
