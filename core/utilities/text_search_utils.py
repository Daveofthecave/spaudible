# core/utilities/text_search_utils.py
"""Text-based search utilities for Spaudible.
Provides fast semantic search using an inverted index with a MARISA trie."""
import math
import re
import struct
import mmap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import sqlite3
from config import (
    PathConfig, VECTOR_HEADER_SIZE, VECTOR_RECORD_SIZE, 
    TRACK_ID_OFFSET_IN_RECORD, EXPECTED_VECTORS
)
from core.preprocessing.querying.query_index_searcher import QueryIndexSearcher
from core.preprocessing.querying.query_tokenizer import tokenize

TOTAL_TRACKS = EXPECTED_VECTORS

COVER_INDICATORS = frozenset([
    'karaoke', 'cover', 'tribute', 'instrumental', 'made famous by',
    'in the style of', 'originally performed by', 'salute to', 'vs',
    'extended', 'mashup', 'remix', 'edit', 'version', 'live', 'unplugged',
    'chillout', 'lounge', 'drum beats', 'feat', 'featuring', 'tribute'
])

@dataclass
class SearchResult:
    track_id: str
    track_name: str
    artist_name: str
    album_name: str = ""
    album_release_year: Optional[str] = None
    popularity: int = 0
    isrc: Optional[str] = None
    confidence: float = 0.0
    matched_tokens: Dict[str, List[str]] = field(default_factory=dict)  # field -> tokens
    
    @property
    def display_text(self) -> str:
        # Format: "Paradise - Coldplay - Mylo Xyloto (2011)"
        artist_str = ', '.join(a.strip() for a in self.artist_name.split(','))
        album_str = f" - {self.album_name}" if len(self.album_name) > 0 else ""
        year_str = f" ({self.album_release_year})" if self.album_release_year else ""
        return f"{self.track_name} - {artist_str}{album_str}{year_str}"
    
    @property
    def is_cover(self) -> bool:
        combined = (self.track_name + " " + self.artist_name).lower()
        return any(ind in combined for ind in COVER_INDICATORS)
    
    @property
    def signature(self) -> str:
        artist = self.artist_name.lower().strip()
        title = re.sub(r'[\(\[].*?[\)\]]', '', self.track_name).lower().strip()
        return f"{artist}|{title}"
    
    def tokens_matched_in_field(self, field: str) -> List[str]:
        return self.matched_tokens.get(field, [])

class VectorMetadataCache:
    def __init__(self):
        self.vectors_path = PathConfig.get_vector_file()
        self._file = open(self.vectors_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def get_popularity(self, vector_idx: int) -> float:
        """
        Extract popularity from vector cache. 
        Dimension 18 is stored at fp32_dims[4] (offset 45 + 16).
        Reverse the sqrt normalization: stored = sqrt(pop/100), so pop = (stored^2) * 100
        """
        record_offset = VECTOR_HEADER_SIZE + (vector_idx * VECTOR_RECORD_SIZE)
        popularity_offset = record_offset + 45 + (3 * 4) # popularity = 4th 4-byte fp32
        
        self._mmap.seek(popularity_offset)
        pop_float = struct.unpack('<f', self._mmap.read(4))[0]
        
        if pop_float < 0 or pop_float > 1.0:  # -1 sentinel or invalid
            return 0.0
        # Reverse sqrt normalization to get 0-100 scale
        return (pop_float ** 2) * 100.0
    
    def get_track_id(self, vector_idx: int) -> Optional[str]:
        """Fast track ID extraction using existing mmap (no file reopening)."""
        try:
            offset = (VECTOR_HEADER_SIZE + (vector_idx * VECTOR_RECORD_SIZE) + TRACK_ID_OFFSET_IN_RECORD)
            self._mmap.seek(offset)
            track_id_bytes = self._mmap.read(22)
            return track_id_bytes.decode('ascii', 'ignore').rstrip('\0')
        except Exception:
            return None
    
    def close(self):
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()

class FlexibleSearcher:
    """
    Searches across all fields simultaneously without guessing partitions.
    Finds tracks where ALL query tokens appear in ANY field (track, artist, or album).
    """
    
    def __init__(self, searcher: QueryIndexSearcher):
        self.searcher = searcher
        self.idf_cache = {}
    
    def get_idf(self, token: str) -> float:
        """Calculate IDF for scoring."""
        if token not in self.idf_cache:
            info = self.searcher._get_token_info(token)
            if info:
                df = info[1]
                # Smooth IDF
                self.idf_cache[token] = math.log((TOTAL_TRACKS - df + 0.5) / (df + 0.5))
            else:
                self.idf_cache[token] = 10.0  # Very rare
        return self.idf_cache[token]
    
    def search(self, tokens: List[str], 
        max_df: int = 1_000_000) -> Dict[int, Dict[str, List[str]]]:
        """
        Find tracks containing ALL tokens in ANY field.
        Returns: {vector_idx: {'track': [tokens], 'artist': [tokens], 'album': [tokens]}}
        """
        if not tokens:
            return {}
        
        # Filter out high-occurrence stopwords
        filtered_tokens = []
        for tok in tokens:
            info = self.searcher._get_token_info(tok)
            if info and info[1] > max_df:
                continue  # Skip stopwords
            filtered_tokens.append(tok)
        
        if not filtered_tokens:
            # All tokens were stopwords, use them anyway but warn
            filtered_tokens = tokens
        
        print(f"DEBUG: Searching for tokens: {filtered_tokens}")
        
        # For each token, find tracks where it appears in any field
        token_candidates = {}  # token -> {idx: [fields]}
        
        for token in filtered_tokens:
            candidates = defaultdict(list)
            
            # Check track field (no prefix)
            self._add_postings(token, candidates, 'track')
            # Check artist field (a_ prefix)
            self._add_postings(f"a_{token}", candidates, 'artist')
            # Check album field (al_ prefix)
            self._add_postings(f"al_{token}", candidates, 'album')
            
            token_candidates[token] = dict(candidates)
        
        # Find intersection: tracks that have ALL tokens in at least one field
        if not token_candidates:
            return {}
        
        # Start with candidates from first token
        all_candidates = set(token_candidates[filtered_tokens[0]].keys())
        
        # Intersect with candidates from other tokens
        for token in filtered_tokens[1:]:
            all_candidates &= set(token_candidates[token].keys())
            if not all_candidates:
                print(f"DEBUG: No candidates after filtering for token '{token}'")
                return {}
        
        print(f"DEBUG: Found {len(all_candidates)} candidates with all tokens")
        
        # Build result with field assignments
        result = {}
        for idx in all_candidates:
            result[idx] = {'track': [], 'artist': [], 'album': []}
            for token in filtered_tokens:
                fields = token_candidates[token].get(idx, [])
                for field in fields:
                    result[idx][field].append(token)
        
        return result
    
    def _add_postings(self, lookup_token: str, candidates: Dict, field_name: str):
        """Helper to add postings for a token to candidate dict."""
        info = self.searcher._get_token_info(lookup_token)
        if not info:
            return
        
        offset, length = info
        try:
            postings = self.searcher._read_posting_list(offset, length)
            for idx in postings:
                candidates[idx].append(field_name)
        except Exception:
            pass

class UnifiedScorer:
    """
    Scores results based on:
    1. Coverage (did we match all tokens?)
    2. Field accuracy (artist tokens in artist field, etc.)
    3. Phrase matching (consecutive tokens)
    4. Quality (originals > covers)
    5. Popularity
    """
    
    def __init__(self, query_tokens: List[str]):
        self.query_tokens = query_tokens
        self.query_set = set(query_tokens)
        self.cover_penalty = CoverPenalty(query_tokens)
    
    def score(self, result: SearchResult) -> float:
        # 1. Coverage (0.0 to 1.0)
        total_matched = len(
            set(result.tokens_matched_in_field('track')) |
            set(result.tokens_matched_in_field('artist')) |
            set(result.tokens_matched_in_field('album'))
        )
        coverage = total_matched / len(self.query_tokens) if self.query_tokens else 0
        
        # 2. Field Accuracy
        # Bonus for tokens appearing in the "right" field based on IDF heuristics
        field_score = 0.0
        
        # Check if it looks like "Track by Artist" format
        track_tokens = set(result.tokens_matched_in_field('track'))
        artist_tokens = set(result.tokens_matched_in_field('artist'))
        
        if track_tokens and artist_tokens:
            # Ideal case: some tokens in track, some in artist
            field_score = 0.3
        
        # 3. Exact phrase matching (significant bonus)
        query_str = " ".join(self.query_tokens)
        track_lower = result.track_name.lower()
        artist_lower = result.artist_name.lower()
        
        phrase_bonus = 0.0
        if query_str in track_lower:
            phrase_bonus = 2.0  # Exact full match in title
        elif " ".join(self.query_tokens[:2]) in track_lower:  # First 2 words
            phrase_bonus = 1.0
        elif " ".join(self.query_tokens[-2:]) in track_lower:  # Last 2 words
            phrase_bonus = 0.8
        
        # 4. Quality (cover/tribute/remix penalty)
        quality_mult = self.cover_penalty.calculate(result)
        
        # 5. Popularity (less fields occupied => greater popularity weight)
        num_fields = sum([
            bool(result.tokens_matched_in_field('track')),
            bool(result.tokens_matched_in_field('artist')),
            bool(result.tokens_matched_in_field('album'))
        ])
        # Determine how many fields have tokens
        if num_fields == 1:
            # Track, Artist, or Album ONLY (eg. "Mr. Brightside") → Heavy boost
            pop_boost = 1.0 + 3.0 * (result.popularity / 100.0)      # 1.0-4.0x
        elif num_fields == 2:
            # Track+Artist or Track+Album → Moderate boost
            pop_boost = 1.0 + 1.0 * (result.popularity / 100.0)      # 1.0-2.0x
        else: # num_fields == 3
            # Track+Artist+Album → Minimal boost
            pop_boost = 1.0 + 0.3 * (result.popularity / 100.0)      # 1.0-1.3x
        
        # Combine individual scores into final score
        # Coverage is quadratic: 100% coverage = 1.0, 90% = 0.81, 80% = 0.64
        coverage_component = coverage ** 2
        
        score = (coverage_component * 0.6 + field_score * 0.2 + phrase_bonus * 0.2)
        final_score = score * quality_mult * pop_boost
        
        return final_score

class CoverPenalty:
    def __init__(self, query_tokens: List[str]):
        self.query_has_cover = bool(set(query_tokens) & COVER_INDICATORS)
    
    def calculate(self, result: SearchResult) -> float:
        if self.query_has_cover:
            return 1.0
        if not result.is_cover:
            return 1.0
        
        # Check if it's just "Live" (less penalty)
        track_lower = result.track_name.lower()
        if 'live' in track_lower and not any(x in track_lower for x in ['karaoke', 'tribute']):
            return 0.6
        
        return 0.1  # Heavy penalty for karaoke/tribute

def search_tracks_flexible(query: str, limit: int = 50) -> List[SearchResult]:
    """
    Main search entry point with popularity pre-filtering.
    Reduces SQL metadata lookups from 300k+ candidates to top 200 by popularity.
    """
    if not query or not query.strip():
        return []
    
    if not QueryIndexSearcher.is_available():
        raise RuntimeError("Query index not found.")
    
    # Tokenize
    raw_tokens = tokenize(query, "")
    if not raw_tokens:
        return []
    
    print(f"DEBUG: Query tokens: {raw_tokens}")
    
    # Initialize
    searcher = QueryIndexSearcher()
    vector_cache = VectorMetadataCache()
    flex_searcher = FlexibleSearcher(searcher)
    scorer = UnifiedScorer(raw_tokens)
    
    try:
        # Phase 1: Find all candidates with matching tokens
        candidates = flex_searcher.search(raw_tokens, max_df=1_000_000)
        if not candidates:
            print("DEBUG: No candidates found")
            return []
        
        vector_indices = list(candidates.keys())
        print(f"DEBUG: {len(vector_indices):,} raw candidates from inverted index")
        
        # Phase 2: Pre-filter by popularity using fast vector cache
        print(f"DEBUG: Ranking candidates by popularity...")
        
        indexed_pops = []
        for idx in vector_indices:
            pop = vector_cache.get_popularity(idx)
            indexed_pops.append((idx, pop))
        
        # Sort by popularity descending and take top N
        # 200 gives us plenty of headroom for the final limit (50) while being fast
        PRE_FILTER_LIMIT = 200
        indexed_pops.sort(key=lambda x: x[1], reverse=True)
        top_candidates = indexed_pops[:PRE_FILTER_LIMIT]
        
        print(f"DEBUG: Selected top {len(top_candidates)} candidates by popularity for metadata lookup")
        
        # Phase 3: Get track IDs only for the top candidates (fast mmap, no SQL yet)
        track_id_pairs = []  # (vector_idx, track_id, popularity)
        for idx, pop in top_candidates:
            track_id = vector_cache.get_track_id(idx)
            if track_id:
                track_id_pairs.append((idx, track_id, pop))
        
        if not track_id_pairs:
            return []
        
        # Phase 4: Batch fetch metadata only for the top 200
        track_ids = [tid for _, tid, _ in track_id_pairs]
        metadata_map = _fetch_metadata_batch(track_ids)
        
        # Phase 5: Build SearchResult objects and score
        results = []
        for idx, track_id, pop in track_id_pairs:
            if track_id not in metadata_map:
                continue
            
            meta = metadata_map[track_id]
            matched = candidates[idx]
            
            result = SearchResult(
                track_id=track_id,
                track_name=meta.get('track_name', 'Unknown'),
                artist_name=meta.get('artist_name', 'Unknown'),
                album_name=meta.get('album_name', ''),
                album_release_year=meta.get('album_release_year'),
                popularity=meta.get('popularity', 0),
                isrc=meta.get('isrc'),
                matched_tokens=matched
            )
            
            result.confidence = scorer.score(result)
            results.append(result)
        
        # Phase 6: Sort by score
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Phase 7: Deduplicate by signature
        seen = set()
        unique_results = []
        for r in results:
            if r.signature not in seen:
                seen.add(r.signature)
                unique_results.append(r)
                if len(unique_results) >= limit:
                    break
        
        print(f"DEBUG: Returning {len(unique_results)} final results")
        return unique_results
        
    finally:
        searcher.close()
        vector_cache.close()

def _fetch_metadata_batch(track_ids: List[str]) -> Dict[str, Dict]:
    """Fetch metadata for multiple tracks."""
    if not track_ids:
        return {}
    
    main_db = PathConfig.get_main_db()
    conn = sqlite3.connect(main_db)
    conn.row_factory = sqlite3.Row
    
    try:
        results = {}
        # Process in chunks of 900 (below SQLite limit)
        for i in range(0, len(track_ids), 900):
            chunk = track_ids[i:i+900]
            placeholders = ','.join(['?'] * len(chunk))
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT t.id, t.name, t.popularity, t.external_id_isrc as isrc, 
                       alb.name as album_name, alb.release_date,
                       GROUP_CONCAT(DISTINCT art.name) as artist_names
                FROM tracks t
                JOIN track_artists ta ON t.rowid = ta.track_rowid
                JOIN artists art ON ta.artist_rowid = art.rowid
                JOIN albums alb ON t.album_rowid = alb.rowid
                WHERE t.id IN ({placeholders})
                GROUP BY t.id
            """, chunk)

            for row in cursor.fetchall():
                # Extract year from release_date (YYYY format)
                year = None
                if row['release_date'] and len(str(row['release_date'])) >= 4:
                    year_str = str(row['release_date'])[:4]
                    if year_str.isdigit():
                        year = year_str
                
                results[row['id']] = {
                    'track_name': row['name'],
                    'artist_name': row['artist_names'],
                    'album_name': row['album_name'],
                    'album_release_year': year,
                    'popularity': row['popularity'],
                    'isrc': row['isrc']
                }
        return results
    finally:
        conn.close()

def _get_track_id_from_vector_idx(vector_idx: int) -> Optional[str]:
    """Extract track ID from vector file."""
    try:
        vectors_path = PathConfig.get_vector_file()
        with open(vectors_path, "rb") as f:
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

def format_search_results(results: List[SearchResult]) -> str:
    """Format results for display."""
    if not results:
        return "No results found."
    
    lines = []
    for idx, result in enumerate(results[:20], 1):
        cover_marker = " [COVER]" if result.is_cover else ""
        lines.append(f"{idx:2d}. {result.display_text}{cover_marker}")
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
        
        # UI components
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
