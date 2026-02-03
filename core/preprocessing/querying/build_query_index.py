# core/preprocessing/querying/build_query_index.py
"""
Song Query Index Builder
===================
Streaming builder for semantic search index.
Uses external merge sort to avoid memory explosion.
"""
import sqlite3
import struct
import marisa_trie
import heapq
import shutil
import gc
import time
import psutil
import signal
import sys
from pathlib import Path
from typing import Tuple, Iterator, Optional, List
from config import PathConfig
from core.preprocessing.querying.query_tokenizer import (
    tokenize_track_name,
    tokenize_artist_name,
    tokenize_album_name
)

# Configuration
CHUNK_SIZE = 1_000_000  # Pairs per chunk during sort
MAX_TOKEN_LENGTH = 255
TOKEN_TABLE_ENTRY_SIZE = 80

def get_memory_usage() -> str:
    """Get current memory usage in human-readable format"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"{mem_info.rss / (1024**3):.1f} GB"

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n  🛑 Received interrupt signal. Cleaning up...")
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def deduplicate_tokens(tokens: List[str]) -> List[str]:
    """Remove duplicate tokens while preserving order (faster than set conversion)."""
    seen = set()
    unique = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique

def write_varint(f, value: int):
    """Write unsigned 32-bit integer as varint"""
    while value >= 0x80:
        f.write(bytes([value & 0x7F | 0x80]))
        value >>= 7
    f.write(bytes([value]))

def write_delta_encoded_list(f, indices: List[int]):
    """Write sorted list of uint32 as delta-encoded varints"""
    prev = 0
    for idx in indices:
        delta = idx - prev
        write_varint(f, delta)
        prev = idx

def write_token_pair(f, token: str, vector_idx: int):
    """Write (token, vector_idx) pair to binary file"""
    token_bytes = token.encode('utf-8')
    if len(token_bytes) > MAX_TOKEN_LENGTH:
        token_bytes = token_bytes[:MAX_TOKEN_LENGTH]
    f.write(struct.pack("<H", len(token_bytes)))
    f.write(token_bytes)
    f.write(struct.pack("<I", vector_idx))

def read_token_pair(f) -> Tuple[Optional[str], Optional[int]]:
    """Read (token, vector_idx) pair from binary file"""
    length_data = f.read(2)
    if not length_data:
        return None, None
    token_length = struct.unpack("<H", length_data)[0]
    
    token_bytes = f.read(token_length)
    vector_idx_data = f.read(4)
    if not vector_idx_data:
        return None, None
    
    token = token_bytes.decode('utf-8', errors='ignore')
    vector_idx = struct.unpack("<I", vector_idx_data)[0]
    return token, vector_idx

def stream_token_pairs() -> Iterator[Tuple[str, int]]:
    """
    Stream tracks from database and yield (token, vector_idx) pairs.
    Uses chunked queries to prevent memory buildup.
    """
    main_db = PathConfig.get_main_db()
    if not main_db.exists():
        raise FileNotFoundError(f"Main database not found: {main_db}")
    
    # Get total track count
    conn = sqlite3.connect(main_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tracks")
    total_tracks = cursor.fetchone()[0]
    conn.close()
    
    processed = 0
    total_pairs = 0
    start_time = time.time()
    
    print(f"  Memory usage: {get_memory_usage()}")
    print(f"  Streaming token pairs from {total_tracks:,} tracks...")
    
    # Process in batches of 500K tracks
    batch_size = 500_000
    
    for batch_start in range(0, total_tracks, batch_size):
        # Create new connection for each batch to mitigate memory leaks
        conn = sqlite3.connect(main_db)
        conn.row_factory = None
        conn.execute("PRAGMA cache_size = -200000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA synchronous = OFF")  # Faster, less memory
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.rowid, t.name,
                   GROUP_CONCAT(DISTINCT art.name),
                   alb.name
            FROM tracks t
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists art ON ta.artist_rowid = art.rowid
            JOIN albums alb ON t.album_rowid = alb.rowid
            WHERE t.rowid BETWEEN ? AND ?
            GROUP BY t.rowid
            ORDER BY t.rowid
        """, (batch_start + 1, batch_start + batch_size))
        
        batch_pairs = 0
        for row in cursor:
            vector_idx = row[0] - 1
            
            # Tokenize
            track_tokens = tokenize_track_name(row[1])
            artist_tokens = tokenize_artist_name(row[2] or "")
            album_tokens = tokenize_album_name(row[3] or "")
            
            # Deduplicate tokens per track; helps reduce filesize
            all_tokens = track_tokens + artist_tokens + album_tokens
            unique_tokens = deduplicate_tokens(all_tokens)
            
            for token in unique_tokens:
                yield token, vector_idx
            
            processed += 1
            total_pairs += len(unique_tokens)
            batch_pairs += len(unique_tokens)
            
            # Print progress every 1M tracks
            if processed % 1_000_000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                avg_tokens = total_pairs / processed
                print(f"    Processed {processed:,} tracks ({rate:.0f} tracks/sec) | "
                      f"{total_pairs:,} pairs | Avg tokens/track: {avg_tokens:.1f} | Memory: {get_memory_usage()}")
        
        # Explicit cleanup to avoid memory leaks
        cursor.close()
        conn.close()
        del cursor
        del conn
        gc.collect()
        
        print(f"    Batch complete. Memory: {get_memory_usage()}")
    
    elapsed = time.time() - start_time
    print(f"  ✓ Streaming complete: {processed:,} tracks, {total_pairs:,} pairs")
    print(f"  ✓ Elapsed time: {elapsed:.1f}s | Rate: {processed/elapsed:.0f} tracks/sec")

def external_sort_token_pairs(temp_dir: Path) -> Path:
    """
    Sort token pairs using external merge sort.
    Returns path to sorted file.
    """
    print("\n" + "=" * 65)
    print("  Phase 1: Writing unsorted token pairs")
    print("=" * 65)
    
    temp_unsorted = temp_dir / "token_pairs_unsorted.bin"
    pair_count = 0
    start_time = time.time()
    
    # Process in batches to control memory
    batch_size = 10_000_000
    
    with open(temp_unsorted, "wb") as f:
        batch = []
        for token, vector_idx in stream_token_pairs():
            batch.append((token, vector_idx))
            pair_count += 1
            
            if len(batch) >= batch_size:
                # Write batch
                for t, v in batch:
                    write_token_pair(f, t, v)
                batch = []
                gc.collect()
                
                if pair_count % 10_000_000 == 0:
                    print(f"    Written {pair_count:,} pairs | Memory: {get_memory_usage()}")
        
        # Write remaining batch
        if batch:
            for t, v in batch:
                write_token_pair(f, t, v)
            batch = None
            gc.collect()
    
    elapsed = time.time() - start_time
    file_size = temp_unsorted.stat().st_size
    print(f"  ✓ Wrote {pair_count:,} pairs")
    print(f"  ✓ File size: {file_size / (1024**3):.2f} GB")
    print(f"  ✓ Write rate: {pair_count/elapsed:.0f} pairs/sec")
    print(f"  ✓ Memory after write: {get_memory_usage()}")
    
    return temp_unsorted

def external_merge_sort(temp_dir: Path, unsorted_path: Path) -> Path:
    """
    Phase 2: External merge sort for token pairs.
    Splits into chunks, sorts each, then merges with heap.
    """
    print("\n" + "=" * 65)
    print("  Phase 2: External Merge Sort")
    print("=" * 65)
    
    # Configuration
    chunk_size = 50_000_000  # 50M pairs per chunk (~500MB)
    unsorted_file = open(unsorted_path, "rb")
    chunk_files = []
    chunk_idx = 0
    
    print("  Splitting into sorted chunks...")
    while True:
        batch = []
        for _ in range(chunk_size):
            token, vector_idx = read_token_pair(unsorted_file)
            if token is None:
                break
            batch.append((token, vector_idx))
        
        if not batch:
            break
        
        # Sort chunk
        batch.sort(key=lambda x: x[0])
        
        # Write chunk
        chunk_path = temp_dir / f"sort_chunk_{chunk_idx:04d}.bin"
        with open(chunk_path, "wb") as cf:
            for token, vector_idx in batch:
                write_token_pair(cf, token, vector_idx)
        
        chunk_files.append(chunk_path)
        chunk_idx += 1
        print(f"    Chunk {chunk_idx}: {len(batch):,} pairs")
        gc.collect()
    
    unsorted_file.close()
    
    # Phase 2b: Merge chunks
    print("  Merging sorted chunks...")
    sorted_path = temp_dir / "token_pairs_sorted.bin"
    with open(sorted_path, "wb") as out_f:
        # Open all chunk files
        files = [open(f, "rb") for f in chunk_files]
        entries = []
        
        # Read first entry from each chunk
        for i, f in enumerate(files):
            token, vector_idx = read_token_pair(f)
            if token is not None:
                heapq.heappush(entries, (token, vector_idx, i))
        
        # Merge with heap
        current_token = None
        current_batch = []
        total_pairs = 0
        
        while entries:
            token, vector_idx, chunk_idx = heapq.heappop(entries)
            
            if token != current_token and current_token is not None:
                # Write batch for previous token
                for v in current_batch:
                    write_token_pair(out_f, current_token, v)
                current_batch = []
                total_pairs += 1
            
            current_token = token
            current_batch.append(vector_idx)
            
            # Read next from same chunk
            next_token, next_idx = read_token_pair(files[chunk_idx])
            if next_token is not None:
                heapq.heappush(entries, (next_token, next_idx, chunk_idx))
        
        # Write last batch
        if current_token is not None:
            for v in current_batch:
                write_token_pair(out_f, current_token, v)
            total_pairs += 1
        
        # Close chunk files
        for f in files:
            f.close()
        
        # Cleanup chunk files
        for f in chunk_files:
            f.unlink()
    
    print(f"  ✓ Sorted file: {sorted_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  ✓ Total pairs: {total_pairs:,}")
    return sorted_path

def build_inverted_index(sorted_path: Path, output_dir: Path) -> Tuple[int, int]:
    """
    Phase 3: Build inverted index from sorted token pairs.
    Returns (token_count, posting_bytes_written)
    """
    print("\n" + "=" * 65)
    print("  Phase 3: Building inverted index")
    print("=" * 65)
    
    marisa_path = output_dir / "marisa_trie.bin"
    postings_path = output_dir / "inverted_index.bin"
    
    # Use generator for tokens to avoid memory leak
    token_generator = _token_generator(sorted_path)
    
    print("  Building posting lists...")
    start_time = time.time()
    
    with open(postings_path, "wb") as postings_f:
        # Write header placeholder
        postings_f.write(b"\0" * 32)
        
        token_table_offset = 32
        postings_offset = token_table_offset
        
        token_count = 0
        total_posting_bytes = 0
        
        for token, indices in token_generator:
            # Write token table entry
            _write_token_entry(postings_f, token, indices,
                              token_table_offset, postings_offset, token_count)
            
            postings_offset = postings_f.tell()
            token_count += 1
            total_posting_bytes += len(indices) * 1.5  # Estimated varint size
            
            # Progress every 100K tokens
            if token_count % 100_000 == 0:
                print(f"    Processed {token_count:,} tokens | "
                      f"Memory: {get_memory_usage()} | "
                      f"Postings: {total_posting_bytes / (1024**3):.2f} GB")
                gc.collect()
        
        # Write final header
        postings_f.seek(0)
        header = struct.pack(
            "<7sBQQQ",
            b"SPAUIDX",
            1,
            token_count,
            token_table_offset,
            postings_offset
        )
        postings_f.write(header)
    
    elapsed = time.time() - start_time
    print(f"  ✓ Inverted index built in {elapsed:.1f}s")
    print(f"  ✓ Unique tokens: {token_count:,}")
    print(f"  ✓ Postings file size: {postings_path.stat().st_size / (1024**3):.2f} GB")
    
    return token_count, postings_path.stat().st_size

def _token_generator(sorted_path: Path) -> Iterator[Tuple[str, List[int]]]:
    """
    Generator that yields (token, [indices]) from sorted file.
    Avoids loading all tokens into memory.
    """
    with open(sorted_path, "rb") as f:
        current_token = None
        current_indices = []
        
        while True:
            token, vector_idx = read_token_pair(f)
            if token is None:
                break
            
            if token != current_token and current_token is not None:
                yield current_token, current_indices
                current_indices = []
            
            current_token = token
            current_indices.append(vector_idx)
        
        if current_token is not None:
            yield current_token, current_indices

def build_marisa_trie(tokens: Iterator[str], output_path: Path) -> None:
    """
    Phase 4: Build MARISA trie from token iterator.
    Memory-efficient: doesn't load all tokens at once.
    """
    print("\n" + "=" * 65)
    print("  Phase 4: Building MARISA trie")
    print("=" * 65)
    
    print(f"  Memory before trie build: {get_memory_usage()}")
    start_time = time.time()
    
    # Build trie from iterator (streaming)
    trie = marisa_trie.Trie(tokens)
    trie.save(str(output_path))
    
    elapsed = time.time() - start_time
    print(f"  ✓ Trie built in {elapsed:.1f}s")
    print(f"  ✓ Trie size: {output_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  ✓ Memory after trie build: {get_memory_usage()}")

def build_query_index():
    """Main entry point with full phase structure and resumability"""
    print("\n" + "=" * 65)
    print("  Building Query Index")
    print("=" * 65)
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Initial memory: {get_memory_usage()}")
    
    overall_start = time.time()
    
    output_dir = PathConfig.get_query_index_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Check for resume
        unsorted_path = temp_dir / "token_pairs_unsorted.bin"
        sorted_path = temp_dir / "token_pairs_sorted.bin"
        
        if unsorted_path.exists() and not sorted_path.exists():
            print(f"  🔍 Resuming from existing unsorted file: {unsorted_path}")
            print(f"  ✓ File size: {unsorted_path.stat().st_size / (1024**3):.2f} GB")
        elif unsorted_path.exists() and sorted_path.exists():
            print(f"  🔍 Resuming from existing sorted file: {sorted_path}")
            print(f"  ✓ File size: {sorted_path.stat().st_size / (1024**3):.2f} GB")
        else:
            # Phase 1: Write unsorted pairs
            unsorted_path = external_sort_token_pairs(temp_dir)
        
        # Phase 2: Sort if needed
        if not sorted_path.exists():
            sorted_path = external_merge_sort(temp_dir, unsorted_path)
        else:
            print(f"  ⏭️  Skipping sort, using existing sorted file")
        
        # Phase 3: Build inverted index
        token_count, postings_size = build_inverted_index(sorted_path, output_dir)
        
        # Phase 4: Build MARISA trie (streaming)
        token_iterator = (token for token, _ in _token_generator(sorted_path))
        marisa_path = output_dir / "marisa_trie.bin"
        build_marisa_trie(token_iterator, marisa_path)
        
        # Final statistics
        total_elapsed = time.time() - overall_start
        
        print("\n" + "=" * 65)
        print("  ✅ Query Index Built Successfully")
        print("=" * 65)
        print(f"  Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f} hours)")
        print(f"  Final memory: {get_memory_usage()}")
        print("\n  📊 Index Statistics:")
        print(f"  • Unique tokens: {token_count:,}")
        print(f"  • MARISA trie: {marisa_path.stat().st_size / (1024**3):.2f} GB")
        print(f"  • Postings file: {postings_size / (1024**3):.2f} GB")
        print(f"  • Total size: {(marisa_path.stat().st_size + postings_size) / (1024**3):.2f} GB")
        
    finally:
        # Clean up temp directory (keep unsorted for resume)
        if sorted_path.exists():
            sorted_path.unlink()
        gc.collect()

def _write_token_entry(f, token: str, indices: List[int], 
                      token_table_offset: int, postings_offset: int, token_idx: int):
    """Write token table entry and posting list"""
    # Calculate entry position
    entry_offset = token_table_offset + token_idx * TOKEN_TABLE_ENTRY_SIZE
    
    # Token string (64 bytes, null-padded)
    token_bytes = token.encode('utf-8')[:63] + b'\0'
    f.seek(entry_offset)
    f.write(token_bytes.ljust(64, b'\0'))
    
    # Document frequency (4 bytes)
    f.write(struct.pack("<I", len(indices)))
    
    # Postings offset (8 bytes)
    f.write(struct.pack("<Q", postings_offset))
    
    # Postings length (4 bytes)
    f.write(struct.pack("<I", len(indices)))
    
    # Write posting list at end of file (delta-encoded)
    f.seek(postings_offset)
    write_delta_encoded_list(f, sorted(indices))

if __name__ == "__main__":
    build_query_index()
