# core/similarity_engine/vector_io_gpu.py
"""
GPU-accelerated vector I/O for unified track_vectors.bin format.
Memory-mapped GPU tensors with CUDA kernel unpacking.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import os
import mmap
from config import PathConfig, VRAM_SAFETY_FACTOR

# Constants for unified format
VECTOR_RECORD_SIZE = 104    # Total bytes per vector record
VECTOR_HEADER_SIZE = 16     # Header size at start of file
MASK_OFFSET_IN_RECORD = 65  # 4-byte mask starts at byte 65 of each record
REGION_OFFSET_IN_RECORD = 69  # 1-byte region code at byte 69
ISRC_OFFSET_IN_RECORD = 70  # 12-byte ISRC at bytes 70-81
TRACK_ID_OFFSET_IN_RECORD = 82  # 22-byte track ID at bytes 82-103

class VectorReaderGPU:
    """GPU-optimized unified vector reader with embedded metadata."""
    
    VECTOR_DIMENSIONS = 32
    BYTES_PER_VECTOR = VECTOR_RECORD_SIZE  # 104 bytes per unified record
    BYTES_PER_MASK = 4      # Mask is embedded, just for reference
    DTYPE = torch.float32
    
    def __init__(self, 
                 vectors_path: Optional[str] = None,
                 device: str = "cuda",
                 vram_scaling_factor_mb: Optional[int] = None):
        """
        Initialize GPU vector reader for unified file format.
        
        Args:
            vectors_path: Path to unified track_vectors.bin
            device: GPU device to use
            vram_scaling_factor_mb: Override VRAM allocation limit
        """
        # Use PathConfig default if not provided
        self.vectors_path = Path(vectors_path) if vectors_path else PathConfig.get_vector_file()
        self.device = device
        
        if not self.vectors_path.exists():
            raise FileNotFoundError(f"Vector file not found: {self.vectors_path}")
        
        # Get file sizes
        self.file_size = self.vectors_path.stat().st_size
        self.total_vectors = (self.file_size - VECTOR_HEADER_SIZE) // VECTOR_RECORD_SIZE
        
        # Validate file structure
        if self.file_size < VECTOR_HEADER_SIZE:
            raise ValueError(f"Vector file too small: {self.file_size} bytes")
        
        if self.total_vectors != (self.file_size - VECTOR_HEADER_SIZE) / VECTOR_RECORD_SIZE:
            raise ValueError(
                f"File size mismatch: expected {VECTOR_RECORD_SIZE} bytes per record, "
                f"but file size {self.file_size} doesn't divide evenly"
            )
        
        # Initialize memory map
        try:
            self._file = open(self.vectors_path, 'rb')
            self._mmap = mmap.mmap(
                self._file.fileno(),
                self.file_size,
                access=mmap.ACCESS_READ
            )
        except Exception as e:
            raise RuntimeError(f"Failed to memory-map vector file: {e}")
        
        # Calculate VRAM allocation
        self.max_batch_size = self._calculate_max_batch_size(vram_scaling_factor_mb)
        
        # Initialize CUDA unpacker if available
        self.cuda_unpacker = self._load_cuda_unpacker()
        
        print(f"✅ GPU Vector Reader initialized:")
        print(f"   File: {self.vectors_path.name}")
        print(f"   Size: {self.file_size / (1024**3):.1f} GB")
        print(f"   Vectors: {self.total_vectors:,}")
        print(f"   Max batch: {self.max_batch_size:,}")
        print(f"   Device: {self.device}")
    
    def _calculate_max_batch_size(self, vram_scaling_factor_mb: Optional[int]) -> int:
        """Calculate maximum vectors that fit in VRAM."""
        bytes_per_vector = VECTOR_RECORD_SIZE  # Full record size
        
        if vram_scaling_factor_mb:
            usable_vram = vram_scaling_factor_mb * 1024 * 1024
        else:
            # Auto-detect available VRAM
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for GPU reader")
            
            free_vram = torch.cuda.mem_get_info()[0]
            usable_vram = int(free_vram * VRAM_SAFETY_FACTOR)
        
        # Account for unpacking overhead (1.5x safety margin)
        max_batch = int((usable_vram // bytes_per_vector) // 1.5)
        
        # Clamp to reasonable range
        return min(max_batch, 50_000_000)  # Max 50M vectors per batch
    
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
        """
        Read and unpack a chunk of vectors to GPU.
        
        Args:
            start_idx: Starting vector index (0-based)
            num_vectors: Number of vectors to read
            
        Returns:
            Tensor of shape (num_vectors, 32) on GPU
        """
        # Validate bounds
        if start_idx < 0 or start_idx + num_vectors > self.total_vectors:
            raise ValueError(
                f"Invalid range: [{start_idx}, {start_idx + num_vectors}) "
                f"outside [0, {self.total_vectors})"
            )
        
        # Process in sub-batches if needed
        if num_vectors <= self.max_batch_size:
            return self._read_chunk_internal(start_idx, num_vectors)
        
        # Split into sub-batches
        results = []
        remaining = num_vectors
        current_start = start_idx
        
        while remaining > 0:
            batch_size = min(remaining, self.max_batch_size)
            batch = self._read_chunk_internal(current_start, batch_size)
            results.append(batch)
            
            current_start += batch_size
            remaining -= batch_size
        
        return torch.cat(results, dim=0)
    
    def _read_chunk_internal(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Internal method to read a single batch."""
        # Calculate byte offset (skip header + jump to record)
        offset = VECTOR_HEADER_SIZE + start_idx * VECTOR_RECORD_SIZE
        num_bytes = num_vectors * VECTOR_RECORD_SIZE
        
        # Read raw bytes from memory map
        raw_data = torch.frombuffer(
            self._mmap,
            dtype=torch.uint8,
            count=num_bytes,
            offset=offset
        ).cuda()
        
        # Unpack via CUDA kernel or fallback
        if self.cuda_unpacker:
            return self._unpack_via_cuda(raw_data, num_vectors)
        else:
            return self._unpack_via_python(raw_data, num_vectors, start_idx)
    
    def _unpack_via_cuda(self, raw_data: torch.Tensor, num_vectors: int) -> torch.Tensor:
        """Use CUDA kernel for high-speed unpacking."""
        # Allocate output tensor
        output = torch.empty((num_vectors, self.VECTOR_DIMENSIONS), 
                           dtype=self.DTYPE,
                           device=self.device)
        
        # Launch kernel
        threads_per_block = 256
        blocks = (num_vectors + threads_per_block - 1) // threads_per_block
        
        self.cuda_unpacker.unpack_vectors_kernel(
            raw_data.data_ptr(),
            output.data_ptr(),
            num_vectors
        )
        
        # Synchronize
        torch.cuda.synchronize()
        
        return output
    
    def _unpack_via_python(self, raw_data: torch.Tensor, num_vectors: int, start_idx: int) -> torch.Tensor:
        """Fallback Python unpacking using numpy for misaligned memory."""
        # Move to CPU to allow flexible memory operations
        records_cpu = raw_data.cpu()
        
        # Use numpy to handle the misaligned views
        import numpy as np
        records_np = records_cpu.numpy().view(np.uint8)
        records_np = records_np.reshape(num_vectors, VECTOR_RECORD_SIZE)
        
        # Allocate torch output on GPU
        vectors = torch.zeros((num_vectors, self.VECTOR_DIMENSIONS), 
                            dtype=self.DTYPE,
                            device=self.device)
        
        # Binary dimensions - these work on GPU
        binary_byte = raw_data[::VECTOR_RECORD_SIZE].to(self.device)
        vectors[:, 9] = ((binary_byte >> 0) & 1).float()
        vectors[:, 11] = ((binary_byte >> 1) & 1).float()
        vectors[:, 12] = ((binary_byte >> 2) & 1).float()
        vectors[:, 13] = ((binary_byte >> 3) & 1).float()
        vectors[:, 14] = ((binary_byte >> 4) & 1).float()
        
        # Scaled dimensions - use numpy then convert to torch
        scaled_np = records_np[:, 1:45].view(np.uint16).reshape(num_vectors, 22)
        scaled_float = scaled_np.astype(np.float32) / 10000.0
        scaled_float = np.where(scaled_float == 0.0, -1.0, scaled_float)
        scaled_gpu = torch.from_numpy(scaled_float).to(self.device)
        
        vectors[:, 0] = scaled_gpu[:, 0]   # acousticness
        vectors[:, 1] = scaled_gpu[:, 1]   # instrumentalness
        vectors[:, 2] = scaled_gpu[:, 2]   # speechiness
        vectors[:, 3] = scaled_gpu[:, 3]   # valence
        vectors[:, 4] = scaled_gpu[:, 4]   # danceability
        vectors[:, 5] = scaled_gpu[:, 5]   # energy
        vectors[:, 6] = scaled_gpu[:, 6]   # liveness
        vectors[:, 8] = scaled_gpu[:, 7]   # key
        vectors[:, 16] = scaled_gpu[:, 8]  # release_date
        vectors[:, 19:32] = scaled_gpu[:, 9:22]  # meta-genres
        
        # FP32 dimensions - use numpy then convert to torch
        fp32_np = records_np[:, 45:65].view(np.float32).reshape(num_vectors, 5)
        fp32_gpu = torch.from_numpy(fp32_np).to(self.device)
        
        vectors[:, 7] = fp32_gpu[:, 0]   # loudness
        vectors[:, 10] = fp32_gpu[:, 1]  # tempo
        vectors[:, 15] = fp32_gpu[:, 2]  # duration
        vectors[:, 17] = fp32_gpu[:, 3]  # popularity
        vectors[:, 18] = fp32_gpu[:, 4]  # followers
        
        # Debug first vector
        if start_idx == 0:
            print(f"  🔍 First vector: {vectors[0, :10]}")
            valid_count = (vectors[0] != -1.0).sum()
            print(f"  🔍 Valid dimensions: {valid_count}/32")
            print(f"  🔍 Vector values range: [{vectors.min():.3f}, {vectors.max():.3f}]")
        
        return vectors

    def read_masks(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """
        Read validity masks from embedded position in vector records.
        Masks are at offset 65 within each 104-byte record.
        """
        # Read the entire chunk first
        file_offset = VECTOR_HEADER_SIZE + start_idx * VECTOR_RECORD_SIZE
        total_bytes = num_vectors * VECTOR_RECORD_SIZE
        
        chunk_bytes = torch.frombuffer(
            self._mmap,
            dtype=torch.uint8,
            count=total_bytes,
            offset=file_offset
        ).cuda()
        
        # View as records, then extract mask bytes (65-68)
        records = chunk_bytes.view(num_vectors, VECTOR_RECORD_SIZE)
        mask_uint8 = records[:, MASK_OFFSET_IN_RECORD:MASK_OFFSET_IN_RECORD+4]
        
        # Debug: Check first few masks
        if start_idx == 0:
            print(f"  🔍 First mask bytes: {mask_uint8[0].cpu().numpy()}")
            print(f"  🔍 Second mask bytes: {mask_uint8[1].cpu().numpy()}")
        
        # Unpack little-endian bytes
        byte0 = mask_uint8[:, 0].to(torch.int32)
        byte1 = mask_uint8[:, 1].to(torch.int32)
        byte2 = mask_uint8[:, 2].to(torch.int32)
        byte3 = mask_uint8[:, 3].to(torch.int32)
        
        masks_int32 = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
        return masks_int32.to(torch.uint32)

    def read_regions(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """Read region codes from byte 69 of each record."""
        file_offset = VECTOR_HEADER_SIZE + start_idx * VECTOR_RECORD_SIZE
        total_bytes = num_vectors * VECTOR_RECORD_SIZE
        
        chunk_bytes = torch.frombuffer(
            self._mmap,
            dtype=torch.uint8,
            count=total_bytes,
            offset=file_offset
        ).cuda()
        
        records = chunk_bytes.view(num_vectors, VECTOR_RECORD_SIZE)
        regions = records[:, REGION_OFFSET_IN_RECORD]  # Byte 69
        
        return regions

    def read_vector_metadata(self, index: int) -> dict:
        """
        Read metadata for a single vector from unified record.
        
        Args:
            index: Vector index to read
            
        Returns:
            dict with mask, region, isrc, track_id
        """
        if index < 0 or index >= self.total_vectors:
            raise ValueError(f"Index {index} out of bounds")
        
        offset = VECTOR_HEADER_SIZE + index * VECTOR_RECORD_SIZE
        
        # Read mask (4 bytes)
        mask_bytes = self._mmap[offset + MASK_OFFSET_IN_RECORD:
                                offset + MASK_OFFSET_IN_RECORD + 4]
        mask = struct.unpack("<I", mask_bytes)[0]
        
        # Read region (1 byte)
        region = self._mmap[offset + REGION_OFFSET_IN_RECORD]
        
        # Read ISRC (12 bytes)
        isrc_bytes = self._mmap[offset + ISRC_OFFSET_IN_RECORD:
                                offset + ISRC_OFFSET_IN_RECORD + 12]
        isrc = isrc_bytes.decode('ascii', 'ignore').rstrip('\0')
        
        # Read track ID (22 bytes)
        track_id_bytes = self._mmap[offset + TRACK_ID_OFFSET_IN_RECORD:
                                    offset + TRACK_ID_OFFSET_IN_RECORD + 22]
        track_id = track_id_bytes.decode('ascii', 'ignore').rstrip('\0')
        
        return {
            'mask': mask,
            'region': region,
            'isrc': isrc,
            'track_id': track_id
        }
    
    def get_total_vectors(self) -> int:
        """Get total number of vectors in the file."""
        return self.total_vectors
    
    def get_max_batch_size(self) -> int:
        """Get maximum batch size for this VRAM configuration."""
        return self.max_batch_size
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()
        if hasattr(self, '_file') and self._file:
            self._file.close()


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
