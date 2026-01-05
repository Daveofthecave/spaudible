# core/preprocessing/mask_generator.py
import struct
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_mask_file(vectors_path: Path, masks_path: Path):
    """
    Generate a mask file from an existing vector file.
    
    Args:
        vectors_path: Path to track_vectors.bin
        masks_path: Path to output track_masks.bin
        
    Returns:
        True if successful, False otherwise
    """
    # Verify vectors file exists
    if not vectors_path.exists():
        print(f"❌ Vector file not found: {vectors_path}")
        return False
    
    # Calculate total vectors
    file_size = vectors_path.stat().st_size
    bytes_per_vector = 128  # 32 dimensions * 4 bytes each
    total_vectors = file_size // bytes_per_vector
    
    print(f"🔍 Found vector file: {vectors_path}")
    print(f"   File size: {file_size / (1024**3):.1f} GB")
    print(f"   Total vectors: {total_vectors:,}")
    
    # Create masks directory if needed
    masks_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in chunks for memory efficiency
    chunk_size = 1000000  # 1 million vectors per chunk
    total_chunks = (total_vectors + chunk_size - 1) // chunk_size
    
    print(f"\n⚙️  Generating masks file: {masks_path}")
    print(f"   Processing in chunks of {chunk_size:,} vectors")
    
    try:
        with open(vectors_path, 'rb') as vectors_file, \
             open(masks_path, 'wb') as masks_file:
            
            progress_bar = tqdm(total=total_vectors, unit='vectors', unit_scale=True)
            
            for _ in range(total_chunks):
                # Read chunk of vectors
                vectors_data = vectors_file.read(chunk_size * bytes_per_vector)
                if not vectors_data:
                    break
                    
                # Convert to numpy array for efficient processing
                vectors = np.frombuffer(vectors_data, dtype=np.float32)
                vectors = vectors.reshape(-1, 32)
                
                # Create masks
                masks = np.zeros(len(vectors), dtype=np.uint32)
                for i in range(32):
                    # Check which vectors have valid values in this dimension
                    valid_mask = vectors[:, i] != -1.0
                    masks |= (valid_mask << i).astype(np.uint32)
                
                # Write masks to file
                masks_bytes = masks.tobytes()
                masks_file.write(masks_bytes)
                
                # Update progress
                progress_bar.update(len(vectors))
            
            progress_bar.close()
        
        # Verify file size
        expected_size = total_vectors * 4  # 4 bytes per mask
        actual_size = masks_path.stat().st_size
        if actual_size != expected_size:
            print(f"⚠️  File size mismatch: expected {expected_size:,} bytes, got {actual_size:,}")
            return False
        else:
            print(f"✅ Successfully generated masks file: {masks_path}")
            print(f"   File size: {actual_size / (1024**3):.1f} GB")
            return True
            
    except Exception as e:
        print(f"💥 Error during mask generation: {e}")
        return False
