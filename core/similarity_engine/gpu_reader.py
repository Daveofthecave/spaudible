# core/similarity_engine/gpu_reader.py
import torch
import numpy as np
from pathlib import Path
import os
import mmap
from torch.utils.cpp_extension import load
from config import PathConfig

# Load CUDA extension
try:
    cuda_unpacker = load(
        name="cuda_unpacker",
        sources=["core/similarity_engine/cuda_unpacker.cu"],
        extra_cuda_cflags=["-O3"],
        verbose=True
    )
except Exception as e:
    print(f"⚠️ Failed to load CUDA unpacker: {e}")
    cuda_unpacker = None

class UnifiedVectorReaderGPU:
    """GPU-optimized reader for unified vector format with direct unpacking."""
    
    HEADER_SIZE = 16  # SPAU magic (4B) + version (4B) + checksum (8B)
    RECORD_SIZE = 104  # Fixed record size
    
    def __init__(self, file_path: str):
        """
        Initialize GPU reader.
        
        Args:
            file_path: Path to track_vectors.bin
        """
        self.path = file_path
        self.file = open(file_path, "rb")
        self.file_size = os.path.getsize(file_path)
        
        # Validate file size
        if (self.file_size - self.HEADER_SIZE) % self.RECORD_SIZE != 0:
            raise ValueError(f"Invalid file size: {self.file_size} bytes")
        
        # Calculate number of vectors
        self.num_vectors = (self.file_size - self.HEADER_SIZE) // self.RECORD_SIZE
        
        # Memory map the file for efficient reading
        self.mmap = mmap.mmap(
            self.file.fileno(), 
            self.file_size, 
            access=mmap.ACCESS_READ
        )
        
        # Skip header
        self.data_start = self.HEADER_SIZE
        self.data_size = self.file_size - self.HEADER_SIZE
        
        # Print initialization info
        print(f"✅ GPU reader initialized: {self.num_vectors:,} vectors")
    
    def read_chunk(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        """
        Read and unpack a chunk of vectors directly on GPU.
        
        Args:
            start_idx: Starting vector index
            num_vectors: Number of vectors to read
            
        Returns:
            Tensor of shape (num_vectors, 32) on GPU
        """
        # Validate indices
        if start_idx < 0 or start_idx + num_vectors > self.num_vectors:
            raise ValueError(f"Invalid indices: {start_idx} to {start_idx+num_vectors} (max {self.num_vectors})")
        
        # Calculate byte range
        start_byte = self.data_start + start_idx * self.RECORD_SIZE
        num_bytes = num_vectors * self.RECORD_SIZE
        
        # Create a tensor directly from the memory-mapped region
        # Note: This avoids an extra copy by using from_buffer
        input_tensor = torch.frombuffer(
            self.mmap, 
            dtype=torch.uint8, 
            count=num_bytes,
            offset=start_byte
        ).cuda()
        
        # Allocate output tensor on GPU
        output = torch.empty((num_vectors, 32), dtype=torch.float32, device="cuda")
        
        # Launch CUDA kernel if available
        if cuda_unpacker:
            threads_per_block = 256
            blocks = (num_vectors + threads_per_block - 1) // threads_per_block
            
            cuda_unpacker.unpack_vectors_kernel(
                input_tensor.data_ptr(),
                output.data_ptr(),
                num_vectors
            )
        else:
            # Fallback to CPU unpacking (slow!)
            print("⚠️ Using CPU fallback for vector unpacking")
            cpu_data = np.frombuffer(
                self.mmap[start_byte:start_byte+num_bytes], 
                dtype=np.uint8
            )
            unpacked = self._unpack_cpu(cpu_data, num_vectors)
            output = torch.tensor(unpacked, dtype=torch.float32, device="cuda")
        
        return output
    
    def _unpack_cpu(self, data: np.ndarray, num_vectors: int) -> np.ndarray:
        """Fallback CPU implementation for unpacking vectors"""
        vectors = np.zeros((num_vectors, 32), dtype=np.float32)
        
        for i in range(num_vectors):
            record = data[i * self.RECORD_SIZE: (i+1) * self.RECORD_SIZE]
            
            # Unpack binary dimensions
            binary_byte = record[0]
            vectors[i, 9] = 1.0 if binary_byte & 1 else 0.0      # mode
            vectors[i, 11] = 1.0 if binary_byte & 2 else 0.0    # time_sig 4/4
            vectors[i, 12] = 1.0 if binary_byte & 4 else 0.0   # time_sig 3/4
            vectors[i, 13] = 1.0 if binary_byte & 8 else 0.0    # time_sig 5/4
            vectors[i, 14] = 1.0 if binary_byte & 16 else 0.0   # time_sig other
            
            # Unpack scaled dimensions
            scaled = np.frombuffer(record[1:45], dtype=np.uint16)
            # First 9 scaled dimensions (0.0001 precision)
            vectors[i, 0] = scaled[0] / 10000.0   # acousticness
            vectors[i, 1] = scaled[1] / 10000.0   # instrumentalness
            vectors[i, 2] = scaled[2] / 10000.0   # speechiness
            vectors[i, 3] = scaled[3] / 10000.0   # valence
            vectors[i, 4] = scaled[4] / 10000.0   # danceability
            vectors[i, 5] = scaled[5] / 10000.0   # energy
            vectors[i, 6] = scaled[6] / 10000.0   # liveness
            vectors[i, 8] = scaled[7] / 10000.0   # key
            vectors[i, 16] = scaled[8] / 10000.0  # release_date
            
            # Next 13 scaled dimensions - meta-genre (0.0001 precision)
            vectors[i, 19] = scaled[9] / 10000.0   # meta-genre1
            vectors[i, 20] = scaled[10] / 10000.0  # meta-genre2
            vectors[i, 21] = scaled[11] / 10000.0  # meta-genre3
            vectors[i, 22] = scaled[12] / 10000.0  # meta-genre4
            vectors[i, 23] = scaled[13] / 10000.0  # meta-genre5
            vectors[i, 24] = scaled[14] / 10000.0  # meta-genre6
            vectors[i, 25] = scaled[15] / 10000.0  # meta-genre7
            vectors[i, 26] = scaled[16] / 10000.0  # meta-genre8
            vectors[i, 27] = scaled[17] / 10000.0  # meta-genre9
            vectors[i, 28] = scaled[18] / 10000.0  # meta-genre10
            vectors[i, 29] = scaled[19] / 10000.0  # meta-genre11
            vectors[i, 30] = scaled[20] / 10000.0  # meta-genre12
            vectors[i, 31] = scaled[21] / 10000.0  # meta-genre13
            
            # Unpack FP32 dimensions
            fp32 = np.frombuffer(record[45:65], dtype=np.float32)
            vectors[i, 7] = fp32[0]   # loudness
            vectors[i, 10] = fp32[1]  # tempo
            vectors[i, 15] = fp32[2]  # duration
            vectors[i, 17] = fp32[3]  # popularity
            vectors[i, 18] = fp32[4]  # followers
        
        return vectors
    
    def get_total_vectors(self) -> int:
        return self.num_vectors
    
    def get_vector_metadata(self, index: int) -> dict:
        """Read metadata for a single vector"""
        if index < 0 or index >= self.num_vectors:
            raise ValueError(f"Invalid index: {index}")
        
        start_byte = self.data_start + index * self.RECORD_SIZE
        record = self.mmap[start_byte:start_byte+self.RECORD_SIZE]
        
        return {
            "validity_mask": struct.unpack("<I", record[65:69])[0],
            "region": record[69],
            "isrc": record[70:82].decode("ascii").rstrip("\0"),
            "track_id": record[82:104].decode("ascii").rstrip("\0")
        }
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, "mmap") and self.mmap:
            self.mmap.close()
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()
