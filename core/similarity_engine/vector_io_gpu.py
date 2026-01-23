# core/similarity_engine/vector_io_gpu.py
"""
GPU-accelerated vector I/O for unified track_vectors.bin format.
Memory-mapped GPU tensors with CUDA kernel unpacking.
"""
import torch
import numpy as np
import mmap
import struct
from pathlib import Path
from typing import Optional, Union, List
from config import PathConfig, VRAM_SAFETY_FACTOR

# Define the record structure (104 bytes)
RECORD_DTYPE = np.dtype([
    ('binary', np.uint8),           # Byte 0
    ('scaled', np.uint16, (22,)),   # Bytes 1-44 (22 uint16 values)
    ('fp32', np.float32, (5,)),     # Bytes 45-64 (5 fp32 values)
    ('mask', np.uint32),            # Bytes 65-68 (uint32 validity mask)
    ('region', np.uint8),           # Byte 69
    ('isrc', 'S12'),                # Bytes 70-81 (12 bytes)
    ('track_id', 'S22')             # Bytes 82-103 (22 bytes)
])

VECTOR_RECORD_SIZE = 104
VECTOR_HEADER_SIZE = 16

class VectorReaderGPU:
    """GPU-optimized unified vector reader with proper structured array parsing."""
    
    def __init__(self, 
                vectors_path: Optional[str] = None,
                device: str = "cuda",
                vram_scaling_factor_mb: Optional[int] = None):
        self.vectors_path = Path(vectors_path) if vectors_path else PathConfig.get_vector_file()
        self.device = device
        
        # Open file handle and memory map
        self._file = open(self.vectors_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._records = np.frombuffer(self._mmap, dtype=RECORD_DTYPE, 
                                    offset=VECTOR_HEADER_SIZE)
        
        self.total_vectors = len(self._records)
        self.RECORD_SIZE = VECTOR_RECORD_SIZE
        self.data_start = VECTOR_HEADER_SIZE
        
        # Use Path.stat() directly (not the file handle)
        self._file_size = self.vectors_path.stat().st_size
        
        # Calculate VRAM allocation
        self.max_batch_size = self._calculate_max_batch_size(vram_scaling_factor_mb)
        
        print(f"✅ GPU Vector Reader initialized:")
        print(f"   Records: {self.total_vectors:,}")
        print(f"   Max batch: {self.max_batch_size:,}")
    
    def _calculate_max_batch_size(self, vram_override: Optional[int] = None) -> int:
        """Calculate maximum vectors that fit in VRAM."""
        bytes_per_vector = self.RECORD_SIZE
        
        if vram_override:
            usable_vram = vram_override * 1024 * 1024
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for GPU reader")
            free_vram = torch.cuda.mem_get_info()[0]
            usable_vram = int(free_vram * VRAM_SAFETY_FACTOR)
        
        return min(int((usable_vram // bytes_per_vector) // 1.5), 50_000_000)

    def _load_cuda_unpacker(self):
        """Load CUDA kernel for unpacking records."""
        try:
            from torch.utils.cpp_extension import load
            
            # Look for CUDA source file
            cuda_source = Path(__file__).parent / "cuda_vector_unpacker.cu"
            if not cuda_source.exists():
                print("  ⚠️  CUDA kernel not found, will use Python fallback")
                return None
            
            # Compile and load
            unpacker = load(
                name="cuda_unpacker",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False
            )
            return unpacker
        
        except Exception as e:
            print(f"  ⚠️  CUDA kernel compilation failed: {e}")
            print("  Will use Python fallback for unpacking")
            return None
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read and convert vectors to torch tensor."""
        if start_idx + num_vectors > self.total_vectors:
            raise ValueError("Range exceeds file bounds")
        
        # Extract records from structured array
        records = self._records[start_idx:start_idx + num_vectors]
        
        # Initialize output
        vectors = torch.full((num_vectors, 32), -1.0, dtype=torch.float32, device=self.device)
        
        # === Unpack binary dimensions (use .copy() to make writable) ===
        binary_numpy = records['binary'].astype(np.int64).copy()
        binary_vals = torch.from_numpy(binary_numpy).to(self.device)
        vectors[:, 9]  = (binary_vals >> 0) & 1
        vectors[:, 11] = (binary_vals >> 1) & 1
        vectors[:, 12] = (binary_vals >> 2) & 1
        vectors[:, 13] = (binary_vals >> 3) & 1
        vectors[:, 14] = (binary_vals >> 4) & 1
        
        # === Unpack scaled dimensions (uint16) ===
        scaled_numpy = records['scaled'].astype(np.float32).copy()
        scaled = torch.from_numpy(scaled_numpy).to(self.device)
        vectors[:, 0]  = scaled[:, 0] / 10000.0
        vectors[:, 1]  = scaled[:, 1] / 10000.0
        vectors[:, 2]  = scaled[:, 2] / 10000.0
        vectors[:, 3]  = scaled[:, 3] / 10000.0
        vectors[:, 4]  = scaled[:, 4] / 10000.0
        vectors[:, 5]  = scaled[:, 5] / 10000.0
        vectors[:, 6]  = scaled[:, 6] / 10000.0
        vectors[:, 8]  = scaled[:, 7] / 10000.0
        vectors[:, 16] = scaled[:, 8] / 10000.0
        vectors[:, 19:32] = scaled[:, 9:22] / 10000.0
        
        # === Unpack fp32 dimensions ===
        fp32_numpy = records['fp32'].astype(np.float32).copy()
        fp32 = torch.from_numpy(fp32_numpy).to(self.device)
        vectors[:, 7]  = fp32[:, 0]
        vectors[:, 10] = fp32[:, 1]
        vectors[:, 15] = fp32[:, 2]
        vectors[:, 17] = fp32[:, 3]
        vectors[:, 18] = fp32[:, 4]
        
        # === Apply validity mask (convert to int64 for bitwise ops) ===
        mask_numpy = records['mask'].astype(np.int64).copy()
        mask = torch.from_numpy(mask_numpy).to(self.device)
        
        for dim in range(32):
            bit = (mask & (1 << dim)) == 0
            vectors[bit, dim] = -1.0
        
        # === DEBUG ===
        if start_idx == 0:
            print(f"  🔍 First vector: {vectors[0, :10]}")
            valid_count = (vectors[0] != -1.0).sum()
            print(f"  🔍 Valid dimensions: {valid_count}/32")
            print(f"  🔍 Vector values range: [{vectors.min():.3f}, {vectors.max():.3f}]")
        
        return vectors

    def read_masks(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read mask values as int32 tensor."""
        records = self._records[start_idx:start_idx + num_vectors]
        mask_numpy = records['mask'].astype(np.int32).copy()
        return torch.from_numpy(mask_numpy).to(self.device)

    def read_regions(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read region codes from byte 69 of each record."""
        records = self._records[start_idx:start_idx + num_vectors]
        return torch.from_numpy(records['region'].astype(np.int32)).to(self.device)

    def get_total_vectors(self) -> int:
        """Get total number of vectors in the file."""
        return self.total_vectors

    def get_track_ids_batch(self, indices: Union[List[int], np.ndarray]) -> List[str]:
        """Batch read track IDs directly from vector file (offset 82-103)."""
        # Handle both list and numpy array inputs
        if isinstance(indices, np.ndarray):
            if indices.size == 0:
                return []
            indices_list = indices.tolist()
        else:
            if len(indices) == 0:
                return []
            indices_list = indices

        # Sort indices for sequential access and to restore original order
        indexed_indices = sorted(enumerate(indices_list), key=lambda x: x[1])
        results = [None] * len(indices_list)
        
        for orig_pos, idx in indexed_indices:
            # Calculate offset: header + (record_index * record_size) + track_id_offset
            offset = self.data_start + idx * self.RECORD_SIZE + 82
            
            # Read 22-byte track ID
            track_id_bytes = self._mmap[offset:offset + 22]
            track_id = track_id_bytes.decode('ascii', 'ignore').rstrip('\0')
            
            results[orig_pos] = track_id
        
        return results

    def get_isrcs_batch(self, indices: Union[List[int], np.ndarray]) -> List[str]:
        """Batch read ISRCs directly from vector file (offset 70-81)."""
        # Handle both list and numpy array inputs
        if isinstance(indices, np.ndarray):
            if indices.size == 0:
                return []
            indices_list = indices.tolist()
        else:
            if len(indices) == 0:
                return []
            indices_list = indices

        indexed_indices = sorted(enumerate(indices_list), key=lambda x: x[1])
        results = [None] * len(indices_list)
        
        for orig_pos, idx in indexed_indices:
            offset = self.data_start + idx * self.RECORD_SIZE + 70
            
            # Read 12-byte ISRC
            isrc_bytes = self._mmap[offset:offset + 12]
            isrc = isrc_bytes.decode('ascii', 'ignore').rstrip('\0')
            
            results[orig_pos] = isrc
        
        return results
    
    def get_max_batch_size(self) -> int:
        """Get maximum batch size for this VRAM configuration."""
        return self.max_batch_size
    
    def close(self):
        """Clean up resources - but NOT the mmap (let GC handle it)."""
        if hasattr(self, '_file') and self._file:
            self._file.close()
    
    def __del__(self):
        """Let Python's garbage collector handle mmap cleanup."""
        self.close()


class RegionReaderGPU:
    """
    Legacy region reader for external track_regions.bin file.
    
    Note: In unified format, regions are embedded in track_vectors.bin.
    This class is kept for backwards compatibility but is not used
    in the new architecture.
    """
    
    def __init__(self, region_path: Optional[str] = None):
        self.region_path = Path(region_path) if region_path else PathConfig.VECTORS / "track_regions.bin"
        self.packed_data = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.total_regions = 0
        self.is_loaded = False
    
    def load(self) -> bool:
        """Load external region file if it exists."""
        if not self.region_path.exists():
            return False
        
        try:
            file_size = self.region_path.stat().st_size
            self.total_regions = int(file_size / 0.375)  # 3 bits per region
            
            with open(self.region_path, 'rb') as f:
                self.packed_data = torch.tensor(
                    np.frombuffer(f.read(), dtype=np.uint8),
                    dtype=torch.uint8,
                    device=self.device
                )
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"  ⚠️  Could not load region file: {e}")
            return False
    
    def get_region_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Extract regions for batch of indices."""
        if not self.is_loaded:
            raise RuntimeError("Region file not loaded")
        
        # Bit extraction logic for external file
        byte_offsets = (indices * 3) // 8
        bit_offsets = (indices * 3) % 8
        
        packed_bytes = self.packed_data[byte_offsets]
        mask_low = bit_offsets <= 5
        
        regions = torch.zeros_like(indices, dtype=torch.uint8)
        
        if torch.any(mask_low):
            shifts = 5 - bit_offsets[mask_low]
            regions[mask_low] = (packed_bytes[mask_low] >> shifts) & 0x07
        
        if torch.any(~mask_low):
            # Handle cross-byte case
            next_bytes = self.packed_data[byte_offsets[~mask_low] + 1]
            regions[~mask_low] = ((packed_bytes[~mask_low] << (bit_offsets[~mask_low] - 5)) | 
                                (next_bytes >> (13 - bit_offsets[~mask_low]))) & 0x07
        
        return regions
    
    def is_available(self) -> bool:
        """Check if external region data is available."""
        return self.is_loaded
