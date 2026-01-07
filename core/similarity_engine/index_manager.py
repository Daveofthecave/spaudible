# core/similarity_engine/index_manager.py
"""
Manage bidirectional mapping between track IDs and vector indices
from the track_index.bin file with ISRC support.
"""
import struct
from config import PathConfig
from pathlib import Path
from typing import List, Optional

class IndexManager:
    """
    Manages the bidirectional mapping between Spotify track IDs and their positions
    in the vector cache (track_vectors.bin) with ISRC support.
    
    The index file (track_index.bin) stores fixed-size records containing:
    - ISRC (12 bytes, UTF-8 encoded)
    - Track ID (22 bytes, UTF-8 encoded)
    - Vector offset (8 bytes, unsigned long long)
    
    Each record is 42 bytes total (12 + 22 + 8).
    """
    
    ISRC_SIZE = 12
    TRACK_ID_SIZE = 22
    OFFSET_SIZE = 8
    INDEX_ENTRY_SIZE = ISRC_SIZE + TRACK_ID_SIZE + OFFSET_SIZE
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize index manager.
        
        Args:
            index_path: Optional custom path to track_index.bin file.
                        Defaults to PathConfig.get_index_file()
        """
        self.index_path = Path(index_path) if index_path else PathConfig.get_index_file()
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector index file not found: {self.index_path}")
    
    def get_track_id(self, index: int) -> str:
        """
        Get track ID for a given vector index.
        
        Args:
            index: Position in vector cache (0-based index)
            
        Returns:
            Spotify track ID string
        """
        with open(self.index_path, 'rb') as f:
            offset = index * self.INDEX_ENTRY_SIZE + self.ISRC_SIZE  # Skip ISRC
            f.seek(offset)
            
            # Read track ID (22 bytes after ISRC)
            track_id_bytes = f.read(self.TRACK_ID_SIZE)
            return track_id_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
    
    def get_isrc(self, index: int) -> str:
        """
        Get ISRC for a given vector index.
        
        Args:
            index: Position in vector cache (0-based index)
            
        Returns:
            ISRC string or empty string if not available
        """
        with open(self.index_path, 'rb') as f:
            offset = index * self.INDEX_ENTRY_SIZE
            f.seek(offset)
            
            # Read ISRC (first 12 bytes of entry)
            isrc_bytes = f.read(self.ISRC_SIZE)
            return isrc_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
    
    def get_track_ids_batch(self, indices: List[int]) -> List[str]:
        """
        Get multiple track IDs at once.
        
        Args:
            indices: List of vector cache positions
            
        Returns:
            List of Spotify track IDs in same order as input indices
        """
        track_ids = []
        with open(self.index_path, 'rb') as f:
            file_size = self.index_path.stat().st_size
            max_offset = file_size - self.INDEX_ENTRY_SIZE
            
            for idx in indices:
                offset = idx * self.INDEX_ENTRY_SIZE + self.ISRC_SIZE
                if offset > max_offset or offset < 0:
                    track_ids.append("")
                    continue
                    
                f.seek(offset)
                track_id_bytes = f.read(self.TRACK_ID_SIZE)
                track_id = track_id_bytes.decode('utf-8', errors='ignore').rstrip('\0')
                track_ids.append(track_id)
                
        return track_ids
    
    def get_isrcs_batch(self, indices: List[int]) -> List[str]:
        """
        Get multiple ISRCs at once.
        
        Args:
            indices: List of vector cache positions
            
        Returns:
            List of ISRC strings in same order as input indices
        """
        isrcs = []
        with open(self.index_path, 'rb') as f:
            file_size = self.index_path.stat().st_size
            max_offset = file_size - self.INDEX_ENTRY_SIZE
            
            for idx in indices:
                offset = idx * self.INDEX_ENTRY_SIZE
                if offset > max_offset or offset < 0:
                    isrcs.append("")
                    continue
                    
                f.seek(offset)
                isrc_bytes = f.read(self.ISRC_SIZE)
                isrc = isrc_bytes.decode('utf-8', errors='ignore').rstrip('\0')
                isrcs.append(isrc)
                
        return isrcs
    
    def get_index_from_track_id(self, track_id: str) -> Optional[int]:
        """
        Find vector index for a given track ID (slow - linear scan).
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Vector index or None if not found
        """
        track_id_bytes = track_id.ljust(self.TRACK_ID_SIZE, '\x00').encode('utf-8')
        
        with open(self.index_path, 'rb') as f:
            file_size = self.index_path.stat().st_size
            num_entries = file_size // self.INDEX_ENTRY_SIZE
            
            for i in range(num_entries):
                offset = i * self.INDEX_ENTRY_SIZE + self.ISRC_SIZE
                f.seek(offset)
                
                entry_track_id_bytes = f.read(self.TRACK_ID_SIZE)
                if entry_track_id_bytes == track_id_bytes:
                    return i
        
        return None
