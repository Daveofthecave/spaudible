# core/utilities/text_search_utils.py
""" Text-based search utilities for Spaudible.
Provides fast semantic search using an inverted index with a MARISA trie.
"""
import difflib
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path
import sqlite3

from config import (
    PathConfig, VECTOR_HEADER_SIZE, VECTOR_RECORD_SIZE, 
    TRACK_ID_OFFSET_IN_RECORD, ISRC_OFFSET_IN_RECORD
)
from core.preprocessing.querying.query_index_searcher import QueryIndexSearcher
from core.preprocessing.querying.query_tokenizer import tokenize


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


def generate_field_partitions(tokens: List[str], max_field_len: int = 5) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    Generate all meaningful partitions of tokens into (Track, Artist, Album).
    Tries fields at the beginning AND end of the token list to handle both
    "Artist Track" and "Track Artist" orders.
    
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
    """
    Main flexible search entry point. Tries all field assignments and scores
    results by text similarity to handle any order of song/artist/album.
    """
    if not query or not query.strip():
        return []
    
    if not QueryIndexSearcher.is_available():
        raise RuntimeError(
            "Query index not found. Please run this command in (venv): \
                python3 -m core.preprocessing.querying.build_query_index"
        )
    
    # Get raw tokens (without field prefixes)
    raw_tokens = tokenize(query, "")
    if not raw_tokens:
        return []
    
    searcher = QueryIndexSearcher()
    try:
        partitions = generate_field_partitions(raw_tokens)
        
        # Aggregate scores across all partitions
        candidate_data = defaultdict(lambda: {
            'max_coverage': 0.0,
            'total_rarity': 0.0,
            'match_count': 0,
            'best_partition': None
        })
        
        for track_toks, artist_toks, album_toks in partitions:
            # Skip empty partitions
            if not track_toks and not artist_toks and not album_toks:
                continue
            
            # Calculate coverage (what fraction of query tokens this partition uses)
            matched_count = len(track_toks) + len(artist_toks) + len(album_toks)
            coverage = matched_count / len(raw_tokens)
            
            # Calculate rarity score (sum of inverse document frequencies)
            rarity = 0.0
            for t in track_toks + artist_toks + album_toks:
                rarity += calculate_token_rarity(t, searcher)
            
            # Query the inverted index with this field assignment
            indices = searcher.search(
                query=" ".join(track_toks),
                artist_query=" ".join(artist_toks),
                album_query=" ".join(album_toks),
                limit=limit * 3  # Get extras for re-ranking
            )
            
            if not indices:
                continue
            
            # Update candidate scores
            for idx in indices:
                cand = candidate_data[idx]
                cand['max_coverage'] = max(cand['max_coverage'], coverage)
                cand['total_rarity'] += rarity
                cand['match_count'] += 1
                if cand['best_partition'] is None:
                    cand['best_partition'] = (track_toks, artist_toks, album_toks)
        
        if not candidate_data:
            return []
        
        # Process and rank candidates
        return _process_candidates(candidate_data, query, raw_tokens, limit)
        
    finally:
        searcher.close()


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


def _process_candidates(
    candidate_data: Dict[int, Dict], 
    original_query: str,
    raw_tokens: List[str],
    limit: int
) -> List[SearchResult]:
    """Convert candidate indices to ranked SearchResults using text similarity."""
    # Map indices to track IDs
    idx_to_track_id = {}
    track_ids = []
    for idx in candidate_data.keys():
        track_id = _get_track_id_from_vector_idx(idx)
        if track_id:
            idx_to_track_id[idx] = track_id
            track_ids.append(track_id)
    
    if not track_ids:
        return []
    
    # Batch fetch metadata
    metadata_map = _fetch_metadata_batch(track_ids)
    
    query_lower = original_query.lower()
    query_tokens_set = set(raw_tokens)
    results = []
    
    for idx, track_id in idx_to_track_id.items():
        meta = metadata_map.get(track_id)
        if not meta:
            continue
        
        cand = candidate_data[idx]
        
        # Build searchable text
        artist = meta['artist_name'] or ''
        track = meta['track_name'] or ''
        album = meta['album_name'] or ''
        full_text = f"{artist} {track} {album}".lower()
        
        # Calculate string similarity (catches exact phrase matches)
        text_sim = difflib.SequenceMatcher(None, query_lower, full_text).ratio()
        
        # Calculate actual token coverage (how many query words appear anywhere)
        matched_tokens = set()
        for tok in raw_tokens:
            if tok in full_text:
                matched_tokens.add(tok)
        actual_coverage = len(matched_tokens) / len(raw_tokens) if raw_tokens else 0
        
        # Popularity normalization
        pop_score = min(meta['popularity'] / 100.0, 1.0)
        
        # Final composite score
        # 40% text similarity (phrase matching)
        # 30% token coverage (keyword matching)
        # 20% rarity (prefer specific/rare matches)
        # 10% consensus (found by multiple partitions)
        final_score = (
            0.40 * text_sim +
            0.30 * actual_coverage +
            0.20 * min(cand['total_rarity'] / 10, 1.0) +
            0.10 * min(cand['match_count'] / 3, 1.0)
        )
        
        # Boost by popularity slightly (breaks ties toward popular songs)
        final_score = final_score * (0.8 + 0.2 * pop_score)
        
        results.append(SearchResult(
            track_id=track_id,
            track_name=track,
            artist_name=artist,
            popularity=meta['popularity'],
            isrc=meta['isrc'],
            confidence=final_score,
            final_score=final_score
        ))
    
    # Sort by final score descending
    results.sort(key=lambda x: x.final_score, reverse=True)
    return results[:limit]


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
    """ Prefix match search using case-insensitive collation.
    Includes ISRC for deduplication. """
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
    # Check terminal size BEFORE creating UI
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
    """
    Fallback text search without prompt_toolkit.
    Used if the library is not installed or terminal is too small.
    """
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
