# core/preprocessing/querying/query_index_searcher.py
"""
Query Index Searcher
====================
Fast text search using MARISA trie + memory-mapped inverted index.
"""
import struct
import mmap
import marisa_trie
from pathlib import Path
from typing import List, Set, Optional
from config import PathConfig
from core.preprocessing.querying.query_tokenizer import tokenize

class QueryIndexSearcher:
    """Fast text search using inverted index"""
    
    # Must match the value in build_query_index.py
    TOKEN_TABLE_ENTRY_SIZE = 80
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if index files exist"""
        return (
            PathConfig.get_query_marisa_file().exists() and 
            PathConfig.get_query_postings_file().exists()
        )
    
    def __init__(self):
        """Load index into memory-mapped files"""
        # Load MARISA trie
        marisa_path = PathConfig.get_query_marisa_file()
        self.trie = marisa_trie.Trie()
        self.trie.load(str(marisa_path))
        
        # Memory-map postings file
        postings_path = PathConfig.get_query_postings_file()
        self._file = open(postings_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read header
        magic, version, token_count, token_table_offset, postings_offset = struct.unpack(
            "<7sBQQQ", self._mmap[:32]
        )
        assert magic == b"SPAUIDX", "Invalid index magic"
        
        self.token_count = token_count
        self.token_table_offset = token_table_offset
        self.postings_offset = postings_offset
    
    def _read_varint(self, offset: int) -> tuple[int, int]:
        """Read varint from mmap, return (value, bytes_read)"""
        result = 0
        shift = 0
        pos = offset
        
        while True:
            byte = self._mmap[pos]
            result |= (byte & 0x7F) << shift
            pos += 1
            if not (byte & 0x80):
                break
            shift += 7
        
        return result, pos - offset
    
    def _read_posting_list(self, offset: int, length: int) -> List[int]:
        """Read delta-encoded varint list and decode to absolute indices"""
        if length == 0:
            return []
        
        indices = []
        current_pos = self.postings_offset + offset
        prev = 0
        
        for _ in range(length):
            delta, bytes_read = self._read_varint(current_pos)
            current_pos += bytes_read
            prev += delta
            indices.append(prev)
        
        return indices
    
    def _get_token_info(self, token: str) -> Optional[tuple[int, int]]:
        """Get (postings_offset, postings_length) for token"""
        if token not in self.trie:
            return None
        
        # Calculate token table entry position
        token_idx = self.trie.key_id(token)
        entry_offset = self.token_table_offset + token_idx * self.TOKEN_TABLE_ENTRY_SIZE
        
        # Skip token string (64 bytes), read metadata
        meta_offset = entry_offset + 64
        doc_freq = struct.unpack("<I", self._mmap[meta_offset:meta_offset+4])[0]
        postings_offset = struct.unpack("<Q", self._mmap[meta_offset+4:meta_offset+12])[0]
        postings_length = struct.unpack("<I", self._mmap[meta_offset+12:meta_offset+16])[0]
        
        return postings_offset, postings_length
    
    def search(self, query: str, limit: int = 50) -> List[int]:
        """
        Search for tracks matching query.
        Returns list of vector indices, sorted by relevance.
        """
        # Tokenize query with all field prefixes
        tokens = tokenize(query, field_prefix="")  # Track tokens
        tokens += tokenize(query, field_prefix="a_")  # Artist tokens
        tokens += tokenize(query, field_prefix="al_")  # Album tokens
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = [t for t in tokens if not (t in seen or seen.add(t))]
        
        # Get posting lists for each token
        posting_lists = []
        for token in unique_tokens:
            info = self._get_token_info(token)
            if info:
                offset, length = info
                indices = self._read_posting_list(offset, length)
                posting_lists.append(set(indices))
        
        if not posting_lists:
            return []
        
        # Intersect all lists (start with smallest for efficiency)
        posting_lists.sort(key=len)
        result = posting_lists[0]
        for lst in posting_lists[1:]:
            result.intersection_update(lst)
            if not result:
                break
        
        # Convert to sorted list and return top N
        return sorted(list(result))[:limit]
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
