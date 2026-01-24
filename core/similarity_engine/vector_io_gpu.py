# core/similarity_engine/vector_io_gpu.py
"""
Vector file reader using PyTorch operations run on the GPU.
"""
import torch
import numpy as np
import struct
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
        
        # Open and memory-map the file
        self._file = open(self.vectors_path, 'rb')
        self.file_size = self.vectors_path.stat().st_size
        self.total_vectors = (self.file_size - VECTOR_HEADER_SIZE) // VECTOR_RECORD_SIZE
        
        # Use numpy memmap to avoid PyTorch's non-writable buffer warning
        self.mmap = np.memmap(self._file, mode='r', dtype=np.uint8, offset=0)
        
        # Auto-detect safe batch size - use 25% of free VRAM (more conservative)
        if self.device == "cuda":
            free_vram = torch.cuda.mem_get_info()[0]
            # Each vector needs: 104B (raw) + 128B (unpacked) + ~256B (intermediates) = ~488B
            # Add 25% safety margin
            safe_batch = int(free_vram * 0.25 / 488)
            self.max_batch_size = min(safe_batch, 2_000_000)  # Cap at 2M for stability
            print(f"  📊 Loaded {self.total_vectors:,} vectors ({self.file_size/1024**3:.1f} GB)")
            print(f"  🚀 Using PyTorch unpacking on {self.device}")
            print(f"  ⚙️  Auto-limited batch size to {self.max_batch_size:,} vectors for safety")
        else:
            self.max_batch_size = 250_000
            print(f"  📊 Loaded {self.total_vectors:,} vectors ({self.file_size/1024**3:.1f} GB)")
            print(f"  🚀 Using PyTorch unpacking on {self.device}")

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
            print(f"  ⚠️  Requested {num_vectors:,} vectors, limiting to {self.max_batch_size:,} for safety")
            num_vectors = self.max_batch_size
        
        if start_idx + num_vectors > self.total_vectors:
            num_vectors = self.total_vectors - start_idx
        
        # Calculate byte range
        byte_offset = VECTOR_HEADER_SIZE + start_idx * VECTOR_RECORD_SIZE
        num_bytes = num_vectors * VECTOR_RECORD_SIZE
        
        # Create explicit writable copy to avoid PyTorch warning/implicit copies
        numpy_data = np.array(self.mmap[byte_offset:byte_offset + num_bytes], copy=True)
        raw_bytes = torch.from_numpy(numpy_data).to(self.device, non_blocking=True)
        
        # Unpack and return
        return self._unpack_vectors(raw_bytes, num_vectors)
    
    def _unpack_vectors(self, raw_bytes: torch.Tensor, num_vectors: int) -> torch.Tensor:
        """
        Unpack 104-byte records from track_vectors.bin into 32D vectors 
        using PyTorch tensor ops. All operations are parallelized
        automatically by PyTorch's CUDA backend.
        """
        # Ensure tensor is contiguous before any view operations
        records = raw_bytes.contiguous().view(num_vectors, VECTOR_RECORD_SIZE)
        
        # Initialize output tensor with sentinel value -1.0
        vectors = torch.full(
            (num_vectors, 32), 
            -1.0, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Binary dimensions (byte 0)
        binary_byte = records[:, 0].to(torch.uint8)
        vectors[:, 9]  = (binary_byte & 1).float()
        vectors[:, 11] = ((binary_byte >> 1) & 1).float()
        vectors[:, 12] = ((binary_byte >> 2) & 1).float()
        vectors[:, 13] = ((binary_byte >> 3) & 1).float()
        vectors[:, 14] = ((binary_byte >> 4) & 1).float()
        
        # Scaled dimensions - use narrow for zero-copy slicing, then contiguous for alignment
        scaled_section = torch.narrow(records, 1, 1, 44).contiguous()  # bytes 1-44
        scaled_int16 = scaled_section.view(torch.int16).view(num_vectors, 22)
        scaled = scaled_int16.float() / 10000.0
        
        # Map scaled values to their vector positions
        vectors[:, 0]  = scaled[:, 0]   # acousticness
        vectors[:, 1]  = scaled[:, 1]   # instrumentalness
        vectors[:, 2]  = scaled[:, 2]   # speechiness
        vectors[:, 3]  = scaled[:, 3]   # valence
        vectors[:, 4]  = scaled[:, 4]   # danceability
        vectors[:, 5]  = scaled[:, 5]   # energy
        vectors[:, 6]  = scaled[:, 6]   # liveness
        vectors[:, 8]  = scaled[:, 7]   # key
        vectors[:, 16] = scaled[:, 8]   # release_date
        vectors[:, 19:32] = scaled[:, 9:22]  # meta-genres 1-13
        
        # FP32 dimensions - use narrow for zero-copy slicing, then contiguous for alignment
        fp32_section = torch.narrow(records, 1, 45, 20).contiguous()  # bytes 45-64
        fp32_tensor = fp32_section.view(torch.float32).view(num_vectors, 5)
        vectors[:, 7]  = fp32_tensor[:, 0]  # loudness
        vectors[:, 10] = fp32_tensor[:, 1]  # tempo
        vectors[:, 15] = fp32_tensor[:, 2]  # duration
        vectors[:, 17] = fp32_tensor[:, 3]  # popularity
        vectors[:, 18] = fp32_tensor[:, 4]  # followers
        
        return vectors
    
    def read_masks(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """
        Read validity masks where bit j indicates dimension j is valid.
        Mask is stored as 4-byte integer at bytes 65-68 of each record.
        """
        # Enforce safe batch size SAME as read_chunk
        if num_vectors > self.max_batch_size:
            print(f"  ⚠️  Requested {num_vectors:,} masks, limiting to {self.max_batch_size:,} for safety")
            num_vectors = self.max_batch_size
        
        if start_idx + num_vectors > self.total_vectors:
            num_vectors = self.total_vectors - start_idx
        
        byte_offset = VECTOR_HEADER_SIZE + start_idx * VECTOR_RECORD_SIZE
        numpy_data = np.array(self.mmap[byte_offset:byte_offset + num_vectors * VECTOR_RECORD_SIZE], copy=True)
        records = torch.from_numpy(numpy_data).to(self.device)
        
        # Reshape to 2D and then use narrow
        records = records.contiguous().view(num_vectors, VECTOR_RECORD_SIZE)
        mask_section = torch.narrow(records, 1, 65, 4).contiguous()
        return mask_section.view(torch.int32).view(num_vectors)

    def read_regions(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """
        Read region codes (single byte at position 69).
        No alignment issues since we're reading single bytes.
        """
        # Enforce safe batch size SAME as read_chunk
        if num_vectors > self.max_batch_size:
            print(f"  ⚠️  Requested {num_vectors:,} regions, limiting to {self.max_batch_size:,} for safety")
            num_vectors = self.max_batch_size
        
        if start_idx + num_vectors > self.total_vectors:
            num_vectors = self.total_vectors - start_idx
        
        byte_offset = VECTOR_HEADER_SIZE + start_idx * VECTOR_RECORD_SIZE
        numpy_data = np.array(self.mmap[byte_offset:byte_offset + num_vectors * VECTOR_RECORD_SIZE], copy=True)
        records = torch.from_numpy(numpy_data).to(self.device)
        
        # Reshape to 2D before accessing column
        records = records.contiguous().view(num_vectors, VECTOR_RECORD_SIZE)
        return records[:, 69].view(torch.uint8)
    
    def get_total_vectors(self) -> int:
        return self.total_vectors
    
    def get_max_batch_size(self) -> int:
        """Estimate max batch size based on available memory."""
        if self.device == "cpu":
            return 500_000  # Conservative for CPU memory
        
        # For GPU, calculate based on free VRAM
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        bytes_per_vector = VECTOR_RECORD_SIZE + (32 * 4)  # Raw + unpacked
        max_batch = int(free_memory * 0.85 / bytes_per_vector)
        return min(max_batch, 50_000_000)  # Cap at 50M
    
    def get_isrcs_batch(self, indices: Union[list, torch.Tensor]) -> list:
        """Extract ISRCs directly from vector file."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        results = []
        for idx in indices:
            # ISRC is at bytes 70-81 (12 bytes)
            byte_offset = VECTOR_HEADER_SIZE + idx * VECTOR_RECORD_SIZE + 70
            isrc_bytes = self.mmap[byte_offset:byte_offset + 12]
            # FIX: Convert numpy array slice to bytes before decoding
            isrc = isrc_bytes.tobytes().decode('ascii', errors='ignore').rstrip('\0')
            results.append(isrc)
        
        return results
    
    def get_track_ids_batch(self, indices: Union[list, torch.Tensor]) -> list:
        """Extract track IDs directly from vector file."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        results = []
        for idx in indices:
            byte_offset = VECTOR_HEADER_SIZE + idx * VECTOR_RECORD_SIZE + 82
            track_id_bytes = self.mmap[byte_offset:byte_offset + 22]
            # FIX: Convert numpy array slice to bytes before decoding
            track_id = track_id_bytes.tobytes().decode('ascii', errors='ignore').rstrip('\0')
            results.append(track_id)
        
        return results
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'mmap'):
            self.mmap._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()
