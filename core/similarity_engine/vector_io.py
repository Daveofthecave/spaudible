# core/similarity_engine/vector_io.py
import mmap
import torch
import sys
import warnings
import numpy as np
from pathlib import Path
from typing import Union

class VectorReader:
    """
    Unified vector reader using CPU-based PyTorch with zero-copy memory mapping.
    PyTorch's CPU backend has superior SIMD and multithreading vs NumPy 
    for byte-manipulation workloads.
    """
    
    VECTOR_RECORD_SIZE = 104
    VECTOR_HEADER_SIZE = 16
    
    def __init__(self, vectors_path: Union[str, Path], device: str = "cpu"):
        self.vectors_path = Path(vectors_path)
        self.device = device
        self._is_windows = sys.platform == "win32"  # To disable memory-mapping on Windows
        
        # Open and memory-map
        self._file = open(self.vectors_path, 'rb')
        self.file_size = self.vectors_path.stat().st_size
        self.total_vectors = (self.file_size - self.VECTOR_HEADER_SIZE) // self.VECTOR_RECORD_SIZE
        
        # Create memory map
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create numpy view instead of a tensor to allow slicing before tensor conversion.
        # This prevents Windows from loading the entire 26GB track_vectors.bin file into RAM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._numpy_array = np.frombuffer(
                self._mmap, 
                dtype=np.uint8, 
                offset=self.VECTOR_HEADER_SIZE
            ).reshape(-1, self.VECTOR_RECORD_SIZE)

    def read_chunk(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        if start_idx >= self.total_vectors:
            return torch.empty((0, 32), dtype=torch.float32, device=self.device)
        
        num_vectors = min(num_vectors, self.total_vectors - start_idx)

        if self._is_windows:
            # On Windows, ditch memory-mapping for file I/O to prevent RAM accumulation
            byte_offset = self.VECTOR_HEADER_SIZE + start_idx * self.VECTOR_RECORD_SIZE
            num_bytes = num_vectors * self.VECTOR_RECORD_SIZE
            self._file.seek(byte_offset)
            raw_bytes = self._file.read(num_bytes)
            records_numpy = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(num_vectors, self.VECTOR_RECORD_SIZE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                records = torch.from_numpy(records_numpy)

            return self._unpack_vectors(records, num_vectors)
        
        # Slice numpy array first, then convert to tensor
        records_numpy = self._numpy_array[start_idx:start_idx + num_vectors]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            records = torch.from_numpy(records_numpy).clone()
        
        return self._unpack_vectors(records, num_vectors)
    
    def _unpack_vectors(self, records: torch.Tensor, num_vectors: int) -> torch.Tensor:
        vectors = torch.full((num_vectors, 32), -1.0, dtype=torch.float32)
        
        # Binary dims (byte 0) - no alignment issues
        binary_byte = records[:, 0]
        vectors[:, 9] = (binary_byte & 1).float()
        vectors[:, 11] = ((binary_byte >> 1) & 1).float()
        vectors[:, 12] = ((binary_byte >> 2) & 1).float()
        vectors[:, 13] = ((binary_byte >> 3) & 1).float()
        vectors[:, 14] = ((binary_byte >> 4) & 1).float()
        
        # Scaled dims (bytes 1-44) - must clone for alignment
        scaled_bytes = records[:, 1:45].clone().contiguous()
        scaled = scaled_bytes.view(torch.int16).view(num_vectors, 22).float() * 0.0001
        vectors[:, 0:7] = scaled[:, 0:7]
        vectors[:, 8] = scaled[:, 7]
        vectors[:, 16] = scaled[:, 8]
        vectors[:, 19:32] = scaled[:, 9:22]
        
        # FP32 dims (bytes 45-64) - must clone for alignment
        fp32_bytes = records[:, 45:65].clone().contiguous()
        fp32 = fp32_bytes.view(torch.float32).view(num_vectors, 5)
        vectors[:, 7] = fp32[:, 0]
        vectors[:, 10] = fp32[:, 1]
        vectors[:, 15] = fp32[:, 2]
        vectors[:, 17] = fp32[:, 3]
        vectors[:, 18] = fp32[:, 4]
        
        return vectors.to(self.device)

    def read_masks(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        if start_idx >= self.total_vectors:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        
        num_vectors = min(num_vectors, self.total_vectors - start_idx)

        # Avoid memory-mapping on Windows, since it triggers excessive RAM growth
        if self._is_windows:
            # Read full chunk, then extract mask bytes (65-68) - avoids seek overhead
            byte_offset = self.VECTOR_HEADER_SIZE + start_idx * self.VECTOR_RECORD_SIZE
            num_bytes = num_vectors * self.VECTOR_RECORD_SIZE
            self._file.seek(byte_offset)
            raw_bytes = self._file.read(num_bytes)
            records = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(num_vectors, self.VECTOR_RECORD_SIZE)
            mask_data = records[:, 65:69]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                masks = torch.from_numpy(mask_data)
            
            return masks.view(torch.int32).view(num_vectors).to(self.device)
        
        # Slice numpy array first, then convert to torch
        mask_bytes = self._numpy_array[start_idx:start_idx + num_vectors, 65:69]
        masks = torch.from_numpy(mask_bytes).clone().contiguous()
        
        return masks.view(torch.int32).view(num_vectors).to(self.device)

    def read_regions(self, start_idx: int, num_vectors: int) -> torch.Tensor:
        if start_idx >= self.total_vectors:
            return torch.empty(0, dtype=torch.uint8, device=self.device)
        
        num_vectors = min(num_vectors, self.total_vectors - start_idx)

        # Avoid memory-mapping on Windows, since it triggers excessive RAM growth
        if self._is_windows:
            # Read full chunk, then extract region byte (69) - avoids seek overhead
            byte_offset = self.VECTOR_HEADER_SIZE + start_idx * self.VECTOR_RECORD_SIZE
            num_bytes = num_vectors * self.VECTOR_RECORD_SIZE
            self._file.seek(byte_offset)
            raw_bytes = self._file.read(num_bytes)
            records = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(num_vectors, self.VECTOR_RECORD_SIZE)
            region_data = records[:, 69]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                regions = torch.from_numpy(region_data)
            
            return regions.view(torch.uint8).view(num_vectors).to(self.device)
        
        # Slice numpy array first, then convert to torch
        region_bytes = self._numpy_array[start_idx:start_idx + num_vectors, 69]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            regions = torch.from_numpy(region_bytes).clone()
        
        return regions.view(torch.uint8).view(num_vectors).to(self.device)

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

    def get_total_vectors(self) -> int:
        return self.total_vectors
    
    def get_max_batch_size(self) -> int:
        if self.device == "cpu":
            return 500_000  # Conservative CPU limit
        
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0]
            safe_batch = int(free_vram * 0.75 / 256)
            return min(safe_batch, 2_000_000)
        
        return 100_000
    
    def close(self):
        if hasattr(self, '_numpy_array'):
            del self._numpy_array
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()
