# core/similarity_engine/vector_io_gpu.py
"""
Vector file reader using PyTorch operations run on the GPU.
"""
import torch
import numpy as np
import struct
import mmap
import warnings
from pathlib import Path
from typing import Optional, Union
from config import PathConfig

# Constants (must match unified vector format)
VECTOR_RECORD_SIZE = 104
VECTOR_HEADER_SIZE = 16

class VectorReaderGPU:
    """GPU-accelerated vector reader using PyTorch tensor operations."""
    
    def __init__(self, 
                vectors_path: Optional[str] = None,
                device: str = "cuda",
                vram_scaling_factor_mb: Optional[int] = None):
        """
        Initialize GPU vector reader.
        
        Args:
            vectors_path: Path to unified track_vectors.bin
            device: Device to use ('cuda' or 'cpu')
            vram_scaling_factor_mb: Legacy parameter (ignored)
        """
        self.vectors_path = Path(vectors_path) if vectors_path else PathConfig.get_vector_file()
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Check CUDA availability first
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            print("  ⚠️ CUDA not available; falling back to CPU mode")
            self.device = "cpu"
        else:
            self.device = device
        
        # Open and memory-map the file
        self._file = open(self.vectors_path, 'rb')
        self.file_size = self.vectors_path.stat().st_size
        self.total_vectors = (self.file_size - VECTOR_HEADER_SIZE) // VECTOR_RECORD_SIZE
        
        # Create persistent memory map
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create numpy view instead of tensor to allow slicing on Windows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._numpy_array = np.frombuffer(
                self._mmap, 
                dtype=np.uint8, 
                offset=VECTOR_HEADER_SIZE
            ).reshape(-1, VECTOR_RECORD_SIZE)
        
        # Pre-allocate GPU unpacking buffer for reuse
        self._gpu_unpack_buffer = None
        
        # Auto-detect safe batch size - use 75% of free VRAM
        if self.device == "cuda":
            free_vram = torch.cuda.mem_get_info()[0]
            # Each vector needs: 104B (raw) + 128B (unpacked) + ~256B (intermediates) = ~488B
            # Add 75% safety margin
            safe_batch = int(free_vram * 0.75 / 488)
             # Empirically-determined ideal size: 500k uses <1GB VRAM
            self.max_batch_size = min(safe_batch, 500_000)
            # print(f"   Loaded {self.total_vectors:,} track vectors ({self.file_size/1000**3:.1f} GB)")
            # print(f"   Loaded {self.total_vectors:,} track vectors")
        else:
            self.max_batch_size = 200_000
            # print(f"   Loaded {self.total_vectors:,} vectors")
            print(f"   Using PyTorch unpacking on {self.device}")

    def read_chunk(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """
        Read and unpack a chunk of vectors safely.
        
        Args:
            start_idx: Starting vector index
            num_vectors: Number of vectors to read
            
        Returns:
            Tensor of shape (num_vectors, 32)
        """
        # Enforce safe batch size
        if num_vectors > self.max_batch_size:
            num_vectors = self.max_batch_size
        
        if start_idx + num_vectors > self.total_vectors:
            num_vectors = self.total_vectors - start_idx
        
        # Slice numpy array first, then convert to tensor
        cpu_records = self._numpy_array[start_idx:start_idx + num_vectors].copy()
        
        # Move to GPU with async transfer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gpu_records = torch.from_numpy(cpu_records).clone().to(self.device, non_blocking=True)
        
        # Unpack using optimized manual method
        return self._unpack_vectors(gpu_records, num_vectors)

    def _unpack_vectors(self, records_gpu: torch.Tensor, num_vectors: int) -> torch.Tensor:
        """
        Unpack 104-byte records from track_vectors.bin into 32D vectors 
        using PyTorch tensor ops. All operations are parallelized
        automatically by PyTorch's CUDA backend.
        
        This version avoids torch.compile to prevent alignment view errors.
        All operations are standard PyTorch ops that work with misaligned memory.
        """
        # Pre-allocate or reuse GPU buffer (crucial for performance)
        if self._gpu_unpack_buffer is None or self._gpu_unpack_buffer.size(0) < num_vectors:
            self._gpu_unpack_buffer = torch.empty(
                num_vectors, 32, 
                dtype=torch.float32, 
                device=records_gpu.device
            )
        vectors = self._gpu_unpack_buffer[:num_vectors].fill_(-1.0)
        
        # === BINARY DIMENSIONS (single byte at offset 0) ===
        # These are always aligned and can be extracted directly
        binary_byte = records_gpu[:, 0]
        vectors[:, 9]  = (binary_byte & 1).float()
        vectors[:, 11] = ((binary_byte >> 1) & 1).float()
        vectors[:, 12] = ((binary_byte >> 2) & 1).float()
        vectors[:, 13] = ((binary_byte >> 3) & 1).float()
        vectors[:, 14] = ((binary_byte >> 4) & 1).float()
        
        # === SCALED DIMENSIONS (22 uint16 values starting at byte 1) ===
        # Byte 1 is at offset 1 in the tensor, which is NOT 2-byte aligned
        # Solution: clone the slice to get aligned memory, then view
        scaled_bytes = records_gpu[:, 1:45].clone().contiguous()
        scaled_int16 = scaled_bytes.view(torch.int16).view(num_vectors, 22)
        scaled_float = scaled_int16.float() * 0.0001
        
        # Copy to output (all these destinations are properly indexed)
        vectors[:, 0:7] = scaled_float[:, 0:7]
        vectors[:, 8] = scaled_float[:, 7]
        vectors[:, 16] = scaled_float[:, 8]
        vectors[:, 19:32] = scaled_float[:, 9:22]
        
        # === FP32 DIMENSIONS (5 float32 values starting at byte 45) ===
        # Byte 45 is NOT 4-byte aligned, so we MUST clone first
        fp32_bytes = records_gpu[:, 45:65].clone().contiguous()
        fp32_section = fp32_bytes.view(torch.float32).view(num_vectors, 5)
        
        vectors[:, 7]  = fp32_section[:, 0]
        vectors[:, 10] = fp32_section[:, 1]
        vectors[:, 15] = fp32_section[:, 2]
        vectors[:, 17] = fp32_section[:, 3]
        vectors[:, 18] = fp32_section[:, 4]
        
        return vectors

    def read_masks(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """
        Read validity masks where bit j indicates dimension j is valid.
        Mask is stored as 4-byte integer at bytes 65-68 of each record.
        """
        # Enforce safe batch size SAME as read_chunk
        if num_vectors > self.max_batch_size:
            print(f"  ⚠️ Requested {num_vectors:,} masks, limiting to {self.max_batch_size:,} for safety")
            num_vectors = self.max_batch_size
        
        if start_idx + num_vectors > self.total_vectors:
            num_vectors = self.total_vectors - start_idx
        
        # Slice numpy array first, then convert and move to GPU
        # Byte 65 may not be 4-byte aligned, so clone first
        mask_bytes = self._numpy_array[start_idx:start_idx + num_vectors, 65:69].copy()
        masks = torch.from_numpy(mask_bytes).clone().contiguous()
        
        # Reinterpret as int32 and return
        return masks.view(torch.int32).view(num_vectors).to(self.device, non_blocking=True)

    def read_regions(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """
        Read region codes (single byte at position 69).
        No alignment issues since we're reading single bytes.
        """
        # Enforce safe batch size same as read_chunk
        if num_vectors > self.max_batch_size:
            print(f"  ⚠️ Requested {num_vectors:,} regions, limiting to {self.max_batch_size:,} for safety")
            num_vectors = self.max_batch_size
        
        if start_idx + num_vectors > self.total_vectors:
            num_vectors = self.total_vectors - start_idx
        
        # Slice numpy array first, then convert and move to GPU
        # Direct slice and return (single bytes don't have alignment issues)
        region_bytes = self._numpy_array[start_idx:start_idx + num_vectors, 69].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            regions = torch.from_numpy(region_bytes).clone()
        
        return regions.view(torch.uint8).view(num_vectors).to(self.device, non_blocking=True)

    def get_total_vectors(self) -> int:
        return self.total_vectors

    def get_max_batch_size(self) -> int:
        """Return the safe batch size that was calculated during initialization."""
        return self.max_batch_size

    def get_isrcs_batch(self, indices: Union[list, torch.Tensor]) -> list:
        """Extract ISRCs directly from vector file."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        results = []
        for idx in indices:
            # ISRC is at bytes 70-81 (12 bytes)
            isrc_bytes = self._numpy_array[idx, 70:82]
            isrc = isrc_bytes.tobytes().decode('ascii', errors='ignore').rstrip('\0')
            results.append(isrc)
        
        return results

    def get_track_ids_batch(self, indices: Union[list, torch.Tensor]) -> list:
        """Extract track IDs directly from vector file."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        results = []
        for idx in indices:
            track_id_bytes = self._numpy_array[idx, 82:104]
            track_id = track_id_bytes.tobytes().decode('ascii', errors='ignore').rstrip('\0')
            results.append(track_id)
        
        return results

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_numpy_array'):
            del self._numpy_array
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()
        if hasattr(self, '_gpu_unpack_buffer'):
            del self._gpu_unpack_buffer
