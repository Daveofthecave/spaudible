# core/utilities/text_search_utils.py
""" Text-based search utilities for Spaudible.
Provides fast semantic search using an inverted index with a MARISA trie.
"""
import difflib
import math
import re
import shutil
import struct
import mmap
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path
import sqlite3
from config import (
    PathConfig,
    VECTOR_HEADER_SIZE,
    VECTOR_RECORD_SIZE,
    TRACK_ID_OFFSET_IN_RECORD,
    ISRC_OFFSET_IN_RECORD
    )
from core.preprocessing.querying.query_index_searcher import QueryIndexSearcher
from core.preprocessing.querying.query_tokenizer import tokenize

TOTAL_TRACKS = 256_039_007  # From config.EXPECTED_VECTORS

@dataclass
class SearchResult:
    """Represents a single search result."""
    track_id: str
    track_name: str
    artist_name: str
    popularity: int
    isrc: Optional[str] = None
    confidence: float = 1.0
    final_score: float = 0.0

    @property
    def display_text(self) -> str:
        """Formatted display string for CLI."""
        year = self.extract_year()
        year_str = f" ({year})" if year else ""
        return f"{self.track_name} - {self.artist_name}{year_str}"

    @property
    def signature(self) -> str:
        """Unique signature for deduplication."""
        # Normalize: lowercase, remove parenthetical content, strip whitespace
        artist = self.artist_name.lower().strip()
        title = self.track_name.lower().split('(')[0].split('[')[0].strip()
        return f"{artist}|{title}"

    def extract_year(self) -> Optional[str]:
        """Extract year from release_date if available."""
        # This would be fetched from the database if needed
        return None


class VectorMetadataCache:
    """Fast metadata extraction from track_vectors.bin without SQLite"""
    
    def __init__(self):
        self.vectors_path = PathConfig.get_vector_file()
        self._file = open(self.vectors_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def get_popularity(self, vector_idx: int) -> float:
        """Extract popularity (dimension 17) from packed vector record"""
        # FP32 section starts at offset 45 in record
        # Index mapping: 0=loudness(7), 1=tempo(10), 2=duration(15), 
        # 3=popularity(17), 4=followers(18)
        # So popularity is at offset 45 + 3*4 = 57
        record_offset = VECTOR_HEADER_SIZE + (vector_idx * VECTOR_RECORD_SIZE)
        popularity_offset = record_offset + 45 + (3 * 4)  # FP32 section + index 3
        
        self._mmap.seek(popularity_offset)
        pop_float = struct.unpack('<f', self._mmap.read(4))[0]
        
        # Convert from normalized sqrt(popularity/100) 
        # back to approximate 0-100 scale if needed
        # Or just return the normalized value. 
        # BM25 expects raw popularity for boosting.
        # The vector stores sqrt(popularity/100.0), 
        # so popularity = (val^2) * 100
        if pop_float < 0:
            return 0.0
        return (pop_float ** 2) * 100.0
    
    def get_isrc(self, vector_idx: int) -> str:
        """Extract ISRC from record (offset 70-81)"""
        record_offset = VECTOR_HEADER_SIZE + (vector_idx * VECTOR_RECORD_SIZE)
        isrc_offset = record_offset + 70
        isrc_bytes = self._mmap[isrc_offset:isrc_offset+12]
        return isrc_bytes.decode('ascii', 'ignore').rstrip('\x00')
    
    def close(self):
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()


class BM25Scorer:
    """BM25 scoring with field weights and coordination"""
    
    # Average field lengths (approximate from corpus analysis)
    AVGDL = {
        'track': 3.5,
        'artist': 2.1,
        'album': 2.8
    }
    
    # Field weights: artist matches are most discriminative
    FIELD_WEIGHTS = {
        'artist': 3.0,
        'track': 2.5,
        'album': 1.0
    }
    
    def __init__(self, searcher: QueryIndexSearcher, k1: float = 1.2, b: float = 0.75):
        self.searcher = searcher
        self.k1 = k1
        self.b = b
        self.idf_cache = {}
    
    def idf(self, token: str) -> float:
        """Robertson/Spark Jones IDF with caching"""
        if token not in self.idf_cache:
            info = self.searcher._get_token_info(token)
            if info:
                df = info[1]  # document frequency
                # IDF = log((N - df + 0.5) / (df + 0.5))
                self.idf_cache[token] = math.log((TOTAL_TRACKS - df + 0.5) / (df + 0.5))
            else:
                self.idf_cache[token] = -10.0  # Very rare/unknown term
        return self.idf_cache[token]
    
    def score_field(self, query_tokens: List[str], 
                    field_tokens: List[str], field_name: str) -> float:
        """BM25 score for a single field"""
        if not field_tokens or not query_tokens:
            return 0.0
        
        field_len = len(field_tokens)
        avgdl = self.AVGDL.get(field_name, field_len)
        
        # Count term frequencies in field
        term_counts = {}
        for tok in field_tokens:
            term_counts[tok] = term_counts.get(tok, 0) + 1
        
        score = 0.0
        for token in query_tokens:
            if token not in term_counts:
                continue
            
            tf = term_counts[token]
            idf = self.idf(token)
            
            # BM25 term saturation and length normalization
            denom = tf + self.k1 * (1 - self.b + self.b * (field_len / avgdl))
            score += idf * (tf * (self.k1 + 1)) / denom
        
        return score * self.FIELD_WEIGHTS[field_name]
    
    def score_document(self, query_tokens: List[str], track_name: str, artist_name: str, 
                      album_name: str, popularity: float) -> float:
        """Compute final BM25 score with coordination and popularity"""
        # Tokenize fields
        track_toks = tokenize(track_name, "") if track_name else []
        artist_toks = tokenize(artist_name, "") if artist_name else []
        album_toks = tokenize(album_name, "") if album_name else []
        
        # Field scores
        score = 0.0
        score += self.score_field(query_tokens, artist_toks, 'artist')
        score += self.score_field(query_tokens, track_toks, 'track')
        score += self.score_field(query_tokens, album_toks, 'album')
        
        # Coordination: boost documents matching all query terms
        all_field_tokens = set(track_toks + artist_toks + album_toks)
        matched_terms = sum(1 for t in query_tokens if t in all_field_tokens)
        if len(query_tokens) > 0:
            coord = (matched_terms / len(query_tokens)) ** 2
            score *= coord
        
        # Popularity dampening (log scale to prevent popularity bias)
        # Convert popularity 0-100 to boost factor 1.0-1.5
        pop_boost = 1.0 + (math.log10(popularity + 1) / 10.0)
        score *= pop_boost
        
        return score


def generate_field_partitions(tokens: List[str], 
    max_field_len: int = 5) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    Generate all meaningful partitions of tokens into (Track, Artist, Album).
    Tries fields at the beginning AND end of the token list to handle 
    both "Artist Track" and "Track Artist" orders.
    Returns list of (track_tokens, artist_tokens, album_tokens).
    """
    n = len(tokens)
    partitions = []
    seen = set()
    
    def add_partition(track, artist, album):
        key = (tuple(track), tuple(artist), tuple(album))
        if key not in seen:
            seen.add(key)
            partitions.append((track, artist, album))
    
    # Strategy 1: Artist first (from start), then Album, then Track
    for a_len in range(0, min(max_field_len, n) + 1):
        for al_len in range(0, min(max_field_len, n - a_len) + 1):
            t_len = n - a_len - al_len
            if t_len < 0 or t_len > max_field_len:
                continue
            artist = tokens[:a_len]
            album = tokens[a_len:a_len+al_len]
            track = tokens[a_len+al_len:]
            add_partition(track, artist, album)
    
    # Strategy 2: Album first (from start), then Artist, then Track
    for al_len in range(0, min(max_field_len, n) + 1):
        for a_len in range(0, min(max_field_len, n - al_len) + 1):
            t_len = n - al_len - a_len
            if t_len < 0 or t_len > max_field_len:
                continue
            album = tokens[:al_len]
            artist = tokens[al_len:al_len+a_len]
            track = tokens[al_len+a_len:]
            add_partition(track, artist, album)
    
    # Strategy 3: Track first (from start), then Artist, then Album
    for t_len in range(0, min(max_field_len, n) + 1):
        for a_len in range(0, min(max_field_len, n - t_len) + 1):
            al_len = n - t_len - a_len
            if al_len < 0 or al_len > max_field_len:
                continue
            track = tokens[:t_len]
            artist = tokens[t_len:t_len+a_len]
            album = tokens[t_len+a_len:]
            add_partition(track, artist, album)
    
    # Strategy 4: Artist from END (e.g., "Echo Black Rebel Motorcycle Club")
    for a_len in range(1, min(max_field_len, n) + 1):
        artist = tokens[-a_len:]
        remaining = tokens[:-a_len]
        # Split remaining into Track (start) and Album (end of remaining)
        for al_len in range(0, min(max_field_len, len(remaining)) + 1):
            album = remaining[-al_len:] if al_len > 0 else []
            track = remaining[:-al_len] if al_len > 0 else remaining
            if len(track) > max_field_len:
                continue
            add_partition(track, artist, album)
    
    # Strategy 5: Album from END
    for al_len in range(1, min(max_field_len, n) + 1):
        album = tokens[-al_len:]
        remaining = tokens[:-al_len]
        for a_len in range(0, min(max_field_len, len(remaining)) + 1):
            artist = remaining[-a_len:] if a_len > 0 else []
            track = remaining[:-a_len] if a_len > 0 else remaining
            if len(track) > max_field_len or len(artist) > max_field_len:
                continue
            add_partition(track, artist, album)

    for p in partitions:
        print(p)

    return partitions


def calculate_token_rarity(token: str, searcher: QueryIndexSearcher) -> float:
    """Calculate rarity score for a token (higher = rarer)."""
    info = searcher._get_token_info(token)
    if not info:
        return 1.0  # Very rare (doesn't exist in index)
    offset, length = info
    # length is document frequency; rare tokens have low length
    return 1.0 / math.log(length + 2)


def search_tracks_flexible(query: str, limit: int = 50) -> List[SearchResult]:
    """Main flexible search entry point."""
    if not query or not query.strip():
        return []
    
    if not QueryIndexSearcher.is_available():
        raise RuntimeError("Query index not found.")

    raw_tokens = tokenize(query, "")
    if not raw_tokens:
        return []
    
    print(f"DEBUG: Query tokens: {raw_tokens}")  # DEBUG
    
    searcher = QueryIndexSearcher()
    vector_cache = VectorMetadataCache()
    
    try:
        all_partitions = generate_field_partitions(raw_tokens)
        
        # Prioritize partitions with artist + track
        priority_partitions = []
        for p in all_partitions:
            track, artist, album = p
            if track and artist and not album:
                priority_partitions.insert(0, p)
            elif artist and track and not album:
                priority_partitions.insert(1, p)
            elif track and not artist and not album:
                priority_partitions.insert(2, p)
            elif artist and not track and not album:
                priority_partitions.insert(3, p)
        
        # Take unique partitions
        seen = set()
        partitions = []
        for p in priority_partitions[:10]:
            key = (tuple(p[0]), tuple(p[1]), tuple(p[2]))
            if key not in seen:
                seen.add(key)
                partitions.append(p)
                print(f"DEBUG: Using partition: {p}")  # DEBUG
            if len(partitions) >= 4:
                break
        
        candidate_data = defaultdict(lambda: {
            'max_coverage': 0.0,
            'total_rarity': 0.0,
            'match_count': 0,
            'best_partition': None
        })
        
        # Dynamic stopword threshold: tokens appearing in >1M tracks
        MAX_DF = 1_000_000
        
        for track_toks, artist_toks, album_toks in partitions:
            if not track_toks and not artist_toks and not album_toks:
                continue
            
            # Check coverage
            partition_tokens = set(track_toks + artist_toks + album_toks)
            if not partition_tokens.issuperset(set(raw_tokens)):
                print(f"DEBUG: Skipping partition {track_toks}/{artist_toks}/{album_toks} - missing tokens")  # DEBUG
                continue
            
            matched_count = len(track_toks) + len(artist_toks) + len(album_toks)
            coverage = matched_count / len(raw_tokens)
            
            # Calculate rarity
            rarity = 0.0
            for t in track_toks + artist_toks + album_toks:
                rarity += calculate_token_rarity(t, searcher)
            
            print(f"DEBUG: Searching track={track_toks}, artist={artist_toks}, album={album_toks}")  # DEBUG
            
            # Search
            indices = searcher.search(
                query=" ".join(track_toks),
                artist_query=" ".join(artist_toks),
                album_query=" ".join(album_toks),
                limit=limit * 20,  # Increased
                max_df=None  # DISABLED max_df filtering for debugging
            )
            
            print(f"DEBUG: Found {len(indices)} results for this partition")  # DEBUG
            
            if not indices:
                continue
            
            # Show first few results for debugging
            for idx in indices[:3]:
                track_id = _get_track_id_from_vector_idx(idx)
                print(f"DEBUG:   Result: {track_id}")  # DEBUG
            
            for idx in indices:
                cand = candidate_data[idx]
                # Update if this partition has better coverage
                if coverage > cand['max_coverage']:
                    cand['max_coverage'] = coverage
                    cand['best_partition'] = (track_toks, artist_toks, album_toks)
                cand['total_rarity'] += rarity
                cand['match_count'] += 1
        
        print(f"DEBUG: Total unique candidates: {len(candidate_data)}")  # DEBUG
        
        if not candidate_data:
            return []
        
        return _process_candidates_bm25(
            candidate_data, query, raw_tokens, limit, searcher, vector_cache
        )
    finally:
        searcher.close()
        vector_cache.close()

def _process_candidates_bm25(
    candidate_data: Dict[int, Dict],
    original_query: str,
    raw_tokens: List[str],
    limit: int,
    searcher: QueryIndexSearcher,
    vector_cache: VectorMetadataCache
) -> List[SearchResult]:
    """Convert candidate indices to ranked SearchResults using BM25 scoring."""
    
    # Initialize BM25 scorer
    scorer = BM25Scorer(searcher, k1=1.2, b=0.75)
    
    # Score all candidates using vector metadata (fast, no DB)
    scored_candidates = []
    for idx in candidate_data.keys():
        # Get popularity from vector file (fast)
        popularity = vector_cache.get_popularity(idx)
        
        # Get track ID for metadata fetch later
        track_id = _get_track_id_from_vector_idx(idx)
        if not track_id:
            continue
        
        scored_candidates.append({
            'idx': idx,
            'track_id': track_id,
            'popularity': popularity,
            'cand_info': candidate_data[idx]
        })
    
    # Only pre-filter if we have an enormous number of candidates
    if len(scored_candidates) > 2000:
        scored_candidates.sort(key=lambda x: x['popularity'], reverse=True)
        top_candidates = scored_candidates[:2000]
    else:
        top_candidates = scored_candidates
    
    # Fetch metadata for top 200 only
    track_ids = [c['track_id'] for c in top_candidates]
    metadata_map = _fetch_metadata_batch(track_ids)
    
    # Compute BM25 scores
    final_results = []
    for cand in top_candidates:
        meta = metadata_map.get(cand['track_id'])
        if not meta:
            continue
        
        # Compute BM25 score
        bm25_score = scorer.score_document(
            query_tokens=raw_tokens,
            track_name=meta.get('track_name', ''),
            artist_name=meta.get('artist_name', ''),
            album_name=meta.get('album_name', ''),
            popularity=cand['popularity']
        )
        
        final_results.append({
            'track_id': cand['track_id'],
            'score': bm25_score,
            'meta': meta,
            'popularity': cand['popularity']
        })
    
    # Sort by BM25 score
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Convert to SearchResult objects
    results = []
    for item in final_results[:limit]:
        meta = item['meta']
        results.append(SearchResult(
            track_id=item['track_id'],
            track_name=meta.get('track_name', 'Unknown'),
            artist_name=meta.get('artist_name', 'Unknown'),
            popularity=meta.get('popularity', 0),
            isrc=meta.get('isrc'),
            confidence=item['score'],
            final_score=item['score']
        ))
    
    return results


def _fetch_metadata_batch(track_ids: List[str]) -> Dict[str, Dict]:
    """Fetch metadata for multiple tracks efficiently in one query."""
    if not track_ids:
        return {}
    
    main_db = PathConfig.get_main_db()
    conn = sqlite3.connect(main_db)
    conn.row_factory = sqlite3.Row

    try:
        placeholders = ','.join(['?'] * len(track_ids))
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT t.id, t.name, t.popularity, t.external_id_isrc as isrc,
                   alb.name as album_name,
                   GROUP_CONCAT(DISTINCT art.name) as artist_names
            FROM tracks t
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists art ON ta.artist_rowid = art.rowid
            JOIN albums alb ON t.album_rowid = alb.rowid
            WHERE t.id IN ({placeholders})
            GROUP BY t.id
        """, track_ids)
        
        results = {}
        for row in cursor.fetchall():
            results[row['id']] = {
                'track_name': row['name'],
                'artist_name': row['artist_names'],
                'album_name': row['album_name'],
                'popularity': row['popularity'],
                'isrc': row['isrc']
            }
        return results
    finally:
        conn.close()

def _get_track_id_from_vector_idx(vector_idx: int) -> Optional[str]:
    """Extract track ID from vector file using direct offset"""
    try:
        vectors_path = PathConfig.get_vector_file()
        with open(vectors_path, "rb") as f:
            # Calculate offset: header + (vector_idx * record_size) + track_id_offset
            offset = (
                VECTOR_HEADER_SIZE +
                (vector_idx * VECTOR_RECORD_SIZE) +
                TRACK_ID_OFFSET_IN_RECORD
            )
            f.seek(offset)
            track_id_bytes = f.read(22)
            return track_id_bytes.decode('ascii', 'ignore').rstrip('\0')
    except Exception:
        return None

def _search_exact(
    conn: sqlite3.Connection, track_name: str, artist_name: str
) -> List[SearchResult]:
    """ Exact match search using case-insensitive collation. 
    Includes ISRC for deduplication. """
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT t.id, t.name, art.name as artist_name, t.popularity, t.external_id_isrc as isrc
        FROM tracks t
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        JOIN artists art ON ta.artist_rowid = art.rowid
        WHERE t.name = ? COLLATE NOCASE
    """
    params = [track_name]
    if artist_name:
        query += " AND art.name = ? COLLATE NOCASE"
        params.append(artist_name)
    query += " ORDER BY t.popularity DESC LIMIT 10"
    cursor.execute(query, params)
    
    return [
        SearchResult(
            track_id=row["id"],
            track_name=row["name"],
            artist_name=row["artist_name"],
            popularity=row["popularity"],
            isrc=row["isrc"],
            confidence=1.0
        )
        for row in cursor.fetchall()
    ]


def _search_prefix(
    conn: sqlite3.Connection, track_name: str, artist_name: str
) -> List[SearchResult]:
    """ Prefix match search using case-insensitive collation. Includes ISRC for deduplication. """
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT t.id, t.name, art.name as artist_name, t.popularity, t.external_id_isrc as isrc
        FROM tracks t
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        JOIN artists art ON ta.artist_rowid = art.rowid
        WHERE t.name LIKE ? || '%' COLLATE NOCASE
    """
    params = [track_name]
    if artist_name:
        query += " AND art.name LIKE ? || '%' COLLATE NOCASE"
        params.append(artist_name)
    query += " ORDER BY t.popularity DESC LIMIT 10"
    cursor.execute(query, params)
    
    return [
        SearchResult(
            track_id=row["id"],
            track_name=row["name"],
            artist_name=row["artist_name"],
            popularity=row["popularity"],
            isrc=row["isrc"],
            confidence=0.7
        )
        for row in cursor.fetchall()
    ]


def extract_track_id_from_result(result: SearchResult) -> str:
    """Extract track_id for similarity search."""
    return result.track_id


def format_search_results(results: List[SearchResult]) -> str:
    """Format results for display in CLI."""
    if not results:
        return "No results found."
    
    lines = []
    for idx, result in enumerate(results[:20], 1):
        lines.append(f"{idx:2d}. {result.display_text}")
    return "\n".join(lines)


# =============================================================================
# Interactive UI with prompt_toolkit
# =============================================================================

def interactive_text_search(initial_query: str = "") -> Optional[str]:
    """
    Interactive text search with arrow-key navigation and query editing.
    Fixed for prompt_toolkit compatibility and terminal size requirements.
    Prevents unnecessary re-searching when query hasn't changed.
    """
    import shutil
    
    # Check terminal size before creating UI
    terminal_size = shutil.get_terminal_size()
    if terminal_size.lines < 25 or terminal_size.columns < 80:
        print(f"\n ⚠️ Terminal too small ({terminal_size.columns}x{terminal_size.lines})")
        print(" Minimum: 80x25. Using simple search...")
        return simple_text_search_fallback(initial_query)
    
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.layout import Layout, HSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style
        
        # State management
        results: List[SearchResult] = []
        selected_idx = 0
        query = initial_query
        last_search_query = ""  # Track what we last searched for
        
        # Create query buffer for editing
        query_buffer = Buffer(
            multiline=False,
            accept_handler=lambda buf: perform_search()
        )
        
        # Set initial text if provided
        if initial_query:
            query_buffer.text = initial_query
        
        # UI Components - Store WINDOW references, not just controls
        query_window = Window(
            height=1,
            content=BufferControl(buffer=query_buffer),
            style="class:query-field",
            cursorline=True
        )
        results_window = Window(
            height=20,
            content=FormattedTextControl(text=""),
            style="class:results-list",
            cursorline=False,
            always_hide_cursor=True
        )
        status_window = Window(
            height=1,
            content=FormattedTextControl(text=""),
            style="class:status-bar"
        )
        
        # Key bindings
        kb = KeyBindings()
        
        def perform_search():
            """Execute search and update results display"""
            nonlocal results, selected_idx, query, last_search_query
            query = query_buffer.text.strip()
            if not query:
                results_window.content.text = "Enter a search query above..."
                return
            
            # Only search if query actually changed
            if query == last_search_query and results:
                return  # Don't re-search the same query
            
            last_search_query = query
            
            try:
                # Use new flexible search
                results = search_tracks_flexible(query, limit=50)
                selected_idx = 0
                if not results:
                    results_window.content.text = f"No results found for '{query}'"
                else:
                    update_results_display()
            except Exception as e:
                results_window.content.text = f"Search error: {str(e)}"
                results = []
        
        def update_results_display():
            """Update the results list with current selection"""
            if not results:
                results_window.content.text = "No results"
                return
            
            lines = []
            display_count = min(len(results), 20)
            for i in range(display_count):
                prefix = "→" if i == selected_idx else " "
                result = results[i]
                lines.append(f"{prefix} {result.display_text}")
            
            if len(results) > 20:
                lines.append(f" ... and {len(results) - 20} more")
            
            results_window.content.text = "\n".join(lines)
        
        def update_status_bar():
            """Update status bar text"""
            if not query:
                status = "Enter a song/artist query"
            else:
                status = f"Query: {query} | {len(results)} results"
            status += " | ↑↓ Navigate | Enter=Select | Ctrl+C=Cancel | Backspace=Edit"
            status_window.content.text = status
        
        @kb.add('up')
        def move_up(event):
            """Navigate up in results list"""
            nonlocal selected_idx
            if results:
                selected_idx = max(0, selected_idx - 1)
                update_results_display()
        
        @kb.add('down')
        def move_down(event):
            """Navigate down in results list"""
            nonlocal selected_idx
            if results:
                selected_idx = min(len(results) - 1, selected_idx + 1)
                update_results_display()
        
        @kb.add('enter')
        def handle_enter(event):
            """Handle Enter key based on context"""
            # If query field is focused, perform search
            if event.app.layout.current_window == query_window:
                perform_search()
                event.app.layout.focus(results_window)
            else:
                # Results field is focused: select track
                if results and selected_idx < len(results):
                    event.app.exit(result=results[selected_idx].track_id)
        
        @kb.add('backspace')
        def handle_backspace(event):
            """Switch focus to query field for editing"""
            event.app.layout.focus(query_window)
        
        @kb.add('c-c')
        def handle_cancel(event):
            """Cancel search and return to main menu"""
            event.app.exit(result=None)
        
        @kb.add('escape')
        def handle_escape(event):
            """Escape key also cancels"""
            event.app.exit(result=None)
        
        # Initial search if query provided
        if query:
            perform_search()
        
        # Create layout
        layout = Layout(
            HSplit([
                # Query label
                Window(
                    height=1,
                    content=FormattedTextControl(text="Search query:"),
                    style="class:query-label"
                ),
                query_window,
                # Results label
                Window(
                    height=1,
                    content=FormattedTextControl(text="Results:"),
                    style="class:results-label"
                ),
                results_window,
                status_window
            ])
        )
        
        # Styling
        style = Style.from_dict({
            'query-label': 'bold ansiblue',
            'query-field': 'bg:ansiblack ansigreen',
            'results-label': 'bold ansiblue',
            'results-list': 'bg:ansiblack ansiwhite',
            'status-bar': 'reverse',
        })
        
        # Create and run application
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False
        )
        
        # Set initial focus
        if query and results:
            app.layout.focus(results_window)
        else:
            app.layout.focus(query_window)
        
        # Run the event loop
        result = app.run()
        return result  # track_id or None
        
    except Exception as e:
        print(f"\n ⚠️ Interactive UI failed: {e}")
        return simple_text_search_fallback(initial_query)


def simple_text_search_fallback(query: str) -> Optional[str]:
    """ Fallback text search without prompt_toolkit. Used if the library is not installed or terminal is too small. """
    from ui.cli.console_utils import print_header
    
    try:
        # Use new flexible search
        results = search_tracks_flexible(query, limit=20)
        if not results:
            print(f"\n ❌ No results found for '{query}'")
            input("\n Press Enter to continue...")
            return None
        
        print_header(f"Search Results for '{query}'")
        print()
        for idx, result in enumerate(results, 1):
            print(f" {idx:2d}. {result.display_text}")
        
        print("\n Options: [1-{}] select, [b]ack".format(len(results)))
        while True:
            choice = input("\n > ").strip().lower()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    return results[idx].track_id
            elif choice == 'b':
                return None
            else:
                print(" ❌ Invalid choice. Try again.")
                
    except Exception as e:
        print(f"\n ❌ Search error: {e}")
        input("\n Press Enter to continue...")
        return None
