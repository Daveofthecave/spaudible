# core/preprocessing/querying/query_index_format.py
"""
Query Index Format Specification
================================

This module documents the binary format for the inverted index used by
the semantic search system. The index consists of two files:

1. data/vectors/query_index/marisa_trie.bin - MARISA trie mapping tokens to metadata
2. data/vectors/query_index/inverted_index.bin - Concatenated posting lists

File Structure: inverted_index.bin
----------------------------------

[Header: 32 bytes]
    Offset  Type    Description
    0       char[7] Magic bytes: "SPAUIDX"
    7       uint8   Format version (currently 1)
    8       uint64  Total number of tokens in the index
    16      uint64  Byte offset to postings section
    24      uint64  Total size of postings section in bytes

[Token Table: 80 bytes per token]
    Each entry maps a token to its posting list location:
    
    Offset  Type    Description
    0       char[64] Null-terminated token string (max 63 chars + null)
    64      uint32  Document frequency (number of tracks containing this token)
    68      uint64  Byte offset to posting list in postings section
    76      uint32  Length of posting list (number of uint32 entries)

[Postings Section: variable length]
    Concatenated delta-encoded varint lists. Each list corresponds to
    one token and contains sorted vector indices where the token appears.
    
    Encoding scheme:
    - Indices are stored as delta-encoded varints (Protocol Buffer style)
    - Example: [1000, 1005, 1010] → encodes as [1000, 5, 5]
    - Each varint uses 7-bit chunks with continuation bit
    - Maximum varint length: 5 bytes for 32-bit integers

Tokenization Rules
------------------
- Lowercase all text
- Preserve apostrophes: "rock'n'roll" → ["rock'n'roll"]
- Unigrams only: "red hot" → ["red", "hot"]
- Field prefixes: "a_" for artist, "al_" for album, no prefix for track
- No stopword removal - "the", "a" are kept
- First artist only: "Keane, The Fray" → ["a_keane"]

Example
-------
For track "Perfect Symmetry" by Keane:
- Track tokens: ["perfect", "symmetry"]
- Artist tokens: ["a_keane"]
- Album tokens: ["al_perfect", "al_symmetry"]
- Combined: ["perfect", "symmetry", "a_keane", "al_perfect", "al_symmetry"]

Posting list for "a_keane": [vector_idx_1, vector_idx_2, ...]

Notes
-----
- All integers are little-endian
- Token strings are UTF-8 encoded (ASCII-clean for Spotify data)
- Posting lists are sorted and delta-encoded for space efficiency
- MARISA trie is built separately from unique tokens
- Average tokens per track: 9.4 (track + artist + album)
- Average postings per token: ~170 (2.41B pairs ÷ 14.2M tokens)

Stats & Performance Characteristics
---------------------------
- Index size: 3.3 GB for 256M tracks (90% compression from raw pairs)
- Build time: ~3-5 hours on an NVMe SSD
- Expected search latency: <50ms cold, <10ms warm
- Memory usage: <1GB during search
- Token lookup: O(1) via MARISA trie
- Intersection: O(k) where k = average postings per token
"""
