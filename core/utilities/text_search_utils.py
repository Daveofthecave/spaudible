# core/utilities/text_search_utils.py
"""
Text-based search utilities for Spaudible.
Provides fast semantic search using inverted index.
Creates optimized case-insensitive indexes for fast text search.
"""
import re
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
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

def parse_query_permutations(query: str) -> List[Dict[str, Any]]:
    """
    Generate ONLY the most likely 3 permutations to minimize database load.
    """
    if not query or not query.strip():
        return []
    
    tokens = query.strip().split()
    if len(tokens) == 1:
        return [{"artist": "", "track": tokens[0], "score": 1.0}]
    
    permutations = []
    
    # Most likely: "Artist Name" "Track Name" (2-token artist, rest track)
    if len(tokens) >= 3:
        permutations.append({
            "artist": " ".join(tokens[:2]),
            "track": " ".join(tokens[2:]),
            "score": 0.95
        })
    
    # Second most likely: "Artist" "Track Name" (1-token artist, rest track)
    permutations.append({
        "artist": tokens[0],
        "track": " ".join(tokens[1:]),
        "score": 0.85
    })
    
    # Fallback: whole string as track name
    permutations.append({
        "artist": "",
        "track": query,
        "score": 0.5
    })
    
    return permutations  # Max 3 permutations, not dozens

def search_tracks_by_permutations(
    permutations: List[Dict[str, Any]], 
    limit: int = 50
) -> List[SearchResult]:
    """
    Search using multiple permutations and merge results.
    Ranks by: match_quality × permutation_score × popularity
    Deduplicates by track signature (artist+track) and ISRC.
    
    FAST PATH: Uses inverted index if available.
    SLOW PATH: Falls back to SQLite permutation search.
    """
    # FAST PATH: Use inverted index if available
    if QueryIndexSearcher.is_available():
        try:
            # Use first permutation's track query as search string
            query_str = permutations[0]["track"] if permutations else ""
            if permutations and permutations[0]["artist"]:
                query_str = f"{permutations[0]['artist']} {query_str}".strip()
            
            searcher = QueryIndexSearcher()
            vector_indices = searcher.search(query_str, limit=limit)
            searcher.close()
            
            if not vector_indices:
                return []
            
            # Convert vector indices to SearchResults
            results = []
            main_db = PathConfig.get_main_db()
            conn = sqlite3.connect(main_db)
            conn.row_factory = sqlite3.Row
            
            try:
                for vector_idx in vector_indices:
                    # Get track metadata from vector index
                    track_id = _get_track_id_from_vector_idx(vector_idx)
                    if not track_id:
                        continue
                    
                    # Fetch metadata from SQLite
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT t.name, art.name as artist_name, t.popularity,
                                t.external_id_isrc as isrc
                        FROM tracks t
                        JOIN track_artists ta ON t.rowid = ta.track_rowid
                        JOIN artists art ON ta.artist_rowid = art.rowid
                        WHERE t.id = ?
                        LIMIT 1
                    """, (track_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        results.append(SearchResult(
                            track_id=track_id,
                            track_name=row["name"],
                            artist_name=row["artist_name"],
                            popularity=row["popularity"],
                            isrc=row["isrc"],
                            confidence=1.0
                        ))
            finally:
                conn.close()
            
            return results
            
        except Exception as e:
            print(f"  ⚠️  Query index search failed: {e}")
            print("  Falling back to slow SQL search...")
            # Continue to slow path below
    
    # Slow path: original SQLite permutation search (TODO: remove)
    if not permutations:
        return []
    
    all_results = []
    seen_signatures = set()
    seen_isrcs = set()
    
    # Use single database connection for all queries
    main_db_path = PathConfig.get_main_db()
    if not main_db_path.exists():
        raise FileNotFoundError(f"Main database not found: {main_db_path}")
    
    conn = sqlite3.connect(str(main_db_path))
    conn.row_factory = sqlite3.Row
    
    # Add optimizations
    conn.execute("PRAGMA cache_size = -200000")  # 200MB cache
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    
    try:
        for perm in permutations:
            # Stop if we have enough results
            if len(all_results) >= limit:
                break
            
            # Search exact match first
            results = _search_exact(conn, perm["track"], perm["artist"])
            for result in results:
                # Deduplicate by signature and ISRC
                if result.signature in seen_signatures:
                    continue
                if result.isrc and result.isrc in seen_isrcs:
                    continue
                
                result.confidence = perm["score"] * 1.0  # Exact match bonus
                all_results.append(result)
                seen_signatures.add(result.signature)
                if result.isrc:
                    seen_isrcs.add(result.isrc)
            
            # Search prefix match if needed
            if len(all_results) < limit:
                results = _search_prefix(conn, perm["track"], perm["artist"])
                for result in results:
                    if result.signature in seen_signatures:
                        continue
                    if result.isrc and result.isrc in seen_isrcs:
                        continue
                    
                    result.confidence = perm["score"] * 0.7  # Prefix penalty
                    all_results.append(result)
                    seen_signatures.add(result.signature)
                    if result.isrc:
                        seen_isrcs.add(result.isrc)
            
            # Track-only exact match
            if len(all_results) < limit and perm["artist"]:
                results = _search_exact(conn, perm["track"], "")
                for result in results:
                    if result.signature in seen_signatures:
                        continue
                    if result.isrc and result.isrc in seen_isrcs:
                        continue
                    
                    result.confidence = perm["score"] * 0.5
                    all_results.append(result)
                    seen_signatures.add(result.signature)
                    if result.isrc:
                        seen_isrcs.add(result.isrc)
        
        # Calculate final score and sort
        for result in all_results:
            # Normalize popularity to 0-1
            popularity_score = min(result.popularity / 100.0, 1.0)
            result.final_score = result.confidence * popularity_score
        
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        
    finally:
        conn.close()
    
    return all_results[:limit]

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
    conn: sqlite3.Connection, 
    track_name: str, 
    artist_name: str
    ) -> List[SearchResult]:
    """
    Exact match search using case-insensitive collation.
    Includes ISRC for deduplication.
    """
    cursor = conn.cursor()
    
    query = """
        SELECT DISTINCT t.id, t.name, art.name as artist_name, t.popularity, 
                t.external_id_isrc as isrc
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
    conn: sqlite3.Connection, 
    track_name: str, 
    artist_name: str
    ) -> List[SearchResult]:
    """
    Prefix match search using case-insensitive collation.
    Includes ISRC for deduplication.
    """
    cursor = conn.cursor()
    
    query = """
        SELECT DISTINCT t.id, t.name, art.name as artist_name, t.popularity,
                t.external_id_isrc as isrc
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
# Interactive UI with prompt_toolkit (Fixed & Optimized)
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
        print(f"\n  ⚠️  Terminal too small ({terminal_size.columns}x{terminal_size.lines})")
        print("  Minimum: 80x25. Using simple search...")
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
                # Parse permutations and search
                permutations = parse_query_permutations(query)
                results = search_tracks_by_permutations(permutations, limit=50)
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
                prefix = "→" if i == selected_idx else "  "
                result = results[i]
                lines.append(f"{prefix} {result.display_text}")
            
            if len(results) > 20:
                lines.append(f"  ... and {len(results) - 20} more")
            
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
        print(f"\n  ⚠️  Interactive UI failed: {e}")
        return simple_text_search_fallback(initial_query)

def simple_text_search_fallback(query: str) -> Optional[str]:
    """
    Fallback text search without prompt_toolkit.
    Used if the library is not installed or terminal is too small.
    """
    from ui.cli.console_utils import print_header
    
    try:
        permutations = parse_query_permutations(query)
        results = search_tracks_by_permutations(permutations, limit=20)
        
        if not results:
            print(f"\n  ❌ No results found for '{query}'")
            input("\n  Press Enter to continue...")
            return None
        
        print_header(f"Search Results for '{query}'")
        print()
        
        for idx, result in enumerate(results, 1):
            print(f"  {idx:2d}. {result.display_text}")
        
        print("\n  Options: [1-{}] select, [b]ack".format(len(results)))
        
        while True:
            choice = input("\n  > ").strip().lower()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    return results[idx].track_id
            elif choice == 'b':
                return None
            else:
                print("  ❌ Invalid choice. Try again.")
    
    except Exception as e:
        print(f"\n  ❌ Search error: {e}")
        input("\n  Press Enter to continue...")
        return None
