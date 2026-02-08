# core/preprocessing/querying/query_index_format.py
"""
Query Index Format Specifications
=================================

This module documents the binary format for the inverted index used by the 
text search system. The index enables fast plaintext search across track names,
artist names, and album titles using field-weighted tokenization.

The index consists of two files in data/vectors/query_index/:
1. marisa_trie.bin - MARISA RecordTrie mapping tokens to token table indices
2. inverted_index.bin - Inverted index with token table + concatenated posting lists


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
    
    Offset  Type     Description
    0       char[64] Null-terminated token string (max 63 chars + null)
    64      uint32   Document frequency (number of tracks containing this token)
    68      uint64   Byte offset to posting list in postings section
    76      uint32   Length of posting list (number of uint32 entries)

[Postings Section: variable length]
    Concatenated and sorted delta-encoded varint lists. Each list corresponds to 
    one token and contains sorted vector indices where the token appears.
    Indices are stored as delta-encoded "varints" in the Protocol Buffer format.
    For example: [1000, 1005, 1010] encodes as [1000, 5, 5].
    Each varint uses 7-bit chunks with a continuation bit.
    The maximum length of a varint is 5 bytes for 32-bit integers.


Tokenization Properties
-----------------------

The method normalize_token() in query_tokenizer.py handles the
tokenization of a string to the proper format (see its docstring for details).

Field Prefixes:
- No prefix: Track name tokens
- "a_": Artist name tokens
- "al_": Album name tokens

Only unigrams are used because bigrams would take up too much disk space.

If the artist field contains multiple artists, only the first one is
preserved to save space (eg. "Coldplay, Rihanna" → ["a_coldplay"]).

While no tokens are removed during indexing, search-time filtering supports
max_df (maximum document frequency) thresholds. Tokens appearing in more
than max_df (eg. 6.5M) tracks (eg. "the", "a", "in", "of") are usually skipped 
at query time to improve performance.

The tokenization phase converts all Unicode characters into ASCII to
enable more flexible searches. This allows the user to type plain
ASCII and still find the intended songs, even if they originally
contained non-ASCII characters (eg. with diacritics).

Tokenization Example
--------------------

For the track "Shattered (Turn The Car Around)" by O.A.R.
from the album All Sides, it will try various combinations
of prefix assignments until it finds the one with the highest score:
- Track tokens: ["shattered", "turn", "car", "around"] ("the" is ignored for speed)
- Artist tokens: ["a_oar"] (dots removed from acronym)
- Album tokens: ["al_all", "al_sides"]


Stats & Performance Characteristics
-----------------------------------

- Total tracks indexed: 256,039,007
- Unique tokens: 17,070,914
- Raw token pairs: ~2.41 billion
- MARISA trie filesize: 147.5 MB
- Inverted index filesize: 4.7 GB (~85% compression from raw pairs)
- Postings section starting address: 0x516684C0
- Integer endianness: little
- Build time: ~4 hours on an NVMe SSD
- Memory usage during search: <1.5 GB
- Expected search latency: <100ms - 5s (slower with more common tokens)
- Token lookup: O(1) via MARISA trie
- Intersection complexity: O(k) where k = size of smallest posting list
"""
