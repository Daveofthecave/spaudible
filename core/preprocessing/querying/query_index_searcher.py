# core/preprocessing/querying/query_index_searcher.py
""" 
Low-Level Query Index Searcher
====================
Fast text search using MARISA RecordTrie + memory-mapped inverted index.
Fixed: Now uses RecordTrie with embedded indices to ensure alignment with token table.
"""
import struct
import mmap
import marisa_trie
from pathlib import Path
from typing import List, Set, Optional, Tuple
from config import PathConfig
from core.preprocessing.querying.query_tokenizer import tokenize


class QueryIndexSearcher:
    """Fast text search using inverted index with RecordTrie alignment."""

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
        # Load MARISA RecordTrie (not regular Trie)
        # FIX: Must provide format string '<I' even when loading from file
        marisa_path = PathConfig.get_query_marisa_file()
        self.trie = marisa_trie.RecordTrie('<I')  # Fixed: added '<I' argument
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

    def _read_varint(self, offset: int) -> Tuple[int, int]:
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
        current_pos = offset
        prev = 0
        for _ in range(length):
            delta, bytes_read = self._read_varint(current_pos)
            current_pos += bytes_read
            prev += delta
            indices.append(prev)
        return indices

    def get_df(self, token: str) -> int:
        """Get document frequency (number of tracks containing this token).
        Returns 0 if token not found.
        """
        info = self._get_token_info(token)
        if not info:
            return 0
        _, length = info
        return length

    def _get_token_info(self, token: str) -> Optional[Tuple[int, int]]:
        """ 
        Get (postings_offset, postings_length) for token using stored index.
        Fixed: Uses RecordTrie to retrieve the token table index stored during building.
        """
        if token not in self.trie:
            return None
        
        # Retrieve the stored token table index from the RecordTrie
        # RecordTrie with format '<I' returns list of tuples like [(idx,)]
        records = self.trie[token]
        if not records:
            return None
        
        # Extract the integer from the first tuple
        token_idx = records[0][0]
        
        # Calculate token table entry position using the stored index
        entry_offset = self.token_table_offset + token_idx * self.TOKEN_TABLE_ENTRY_SIZE
        
        # Skip token string (64 bytes), read metadata
        meta_offset = entry_offset + 64
        doc_freq = struct.unpack("<I", self._mmap[meta_offset:meta_offset+4])[0]
        postings_offset = struct.unpack("<Q", self._mmap[meta_offset+4:meta_offset+12])[0]
        postings_length = struct.unpack("<I", self._mmap[meta_offset+12:meta_offset+16])[0]
        
        return postings_offset, postings_length

    def search(self, query: str, limit: int = 50, artist_query: str = "", 
        album_query: str = "", max_df: Optional[int] = None) -> List[int]:
        """ 
        Field-aware search supporting track, artist, and album queries.
        Uses AND logic across fields (track AND artist AND album).
        
        Args:
            query: Track name query
            limit: Max results to return
            artist_query: Artist field query
            album_query: Album field query
            max_df: Skip tokens appearing in more than this many documents (stopword filter)
        """
        posting_lists = []
        
        # Track tokens (no prefix)
        if query:
            track_tokens = tokenize(query, field_prefix="")
            track_postings = self._get_postings_for_tokens(track_tokens, max_df=max_df)
            if track_postings:
                posting_lists.append(track_postings)
        
        # Artist tokens (with a_ prefix)
        if artist_query:
            artist_tokens = tokenize(artist_query, field_prefix="a_")
            artist_postings = self._get_postings_for_tokens(artist_tokens, max_df=max_df)
            if artist_postings:
                posting_lists.append(artist_postings)
        
        # Album tokens (with al_ prefix)
        if album_query:
            album_tokens = tokenize(album_query, field_prefix="al_")
            album_postings = self._get_postings_for_tokens(album_tokens, max_df=max_df)
            if album_postings:
                posting_lists.append(album_postings)
        
        if not posting_lists:
            return []
        
        # Intersect across fields (AND logic)
        posting_lists.sort(key=len)
        result = posting_lists[0]
        for lst in posting_lists[1:]:
            result.intersection_update(lst)
            if not result:
                break
        
        return sorted(list(result))[:limit]

    def _get_postings_for_tokens(self, tokens: List[str], 
        max_df: Optional[int] = None) -> Optional[Set[int]]:
        """Get intersection of posting lists for a set of tokens.
        
        Args:
            tokens: List of tokens to intersect
            max_df: Maximum document frequency. Tokens appearing in more tracks than this
                   are treated as stopwords and skipped entirely. If all tokens are
                   stopwords, falls back to the rarest token.
        """
        if not tokens:
            return None
        
        posting_lists = []
        rarest_token = None
        rarest_df = float('inf')
        rarest_postings = None
        
        for token in tokens:
            info = self._get_token_info(token)
            if not info:
                return set()  # If any token missing, this partition fails
            
            offset, length = info
            
            # Track the rarest token for fallback
            if length < rarest_df:
                rarest_df = length
                rarest_token = token
                # Don't read yet, just store info
            
            # Skip stopwords (high frequency tokens)
            if max_df and length > max_df:
                continue
            
            # Load posting list for this token
            indices = self._read_posting_list(offset, length)
            posting_lists.append(set(indices))
        
        # If all tokens were stopwords, use the rarest one
        if not posting_lists and rarest_token:
            offset, length = self._get_token_info(rarest_token)
            indices = self._read_posting_list(offset, length)
            return set(indices)
        
        # If still no posting lists (shouldn't happen unless empty input), return None
        if not posting_lists:
            return None
        
        # Intersect (AND logic)
        posting_lists.sort(key=len)
        result = posting_lists[0]
        for lst in posting_lists[1:]:
            result.intersection_update(lst)
            if not result:
                break
        
        return result

    def validate_index(self, test_token: str = "rock") -> bool:
        """ 
        Validate that a common token can be found and its postings read correctly.
        Returns True if valid, raises exception with details if not.
        """
        if test_token not in self.trie:
            raise ValueError(f"Test token '{test_token}' not found in trie")
        
        # Get token info
        info = self._get_token_info(test_token)
        if not info:
            raise ValueError(f"Could not retrieve token info for '{test_token}'")
        
        offset, length = info
        
        # Verify offset is within file bounds
        if offset >= len(self._mmap):
            raise ValueError(
                f"Token '{test_token}' offset {offset} exceeds file size {len(self._mmap)}"
            )
        
        # Verify we can read the posting list
        try:
            indices = self._read_posting_list(offset, length)
            if len(indices) != length:
                raise ValueError(
                    f"Expected {length} postings, got {len(indices)}"
                )
            return True
        except Exception as e:
            raise ValueError(
                f"Failed to read postings for token '{test_token}': {e}"
            )

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
