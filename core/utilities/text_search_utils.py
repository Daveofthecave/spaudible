# core/utilities/text_search_utils.py
"""
Text-based search utilities for Spaudible.
Provides fast search using SQLite FTS5 virtual table.
Creates FTS5 index once for sub-second text search.
"""
import re
import shutil
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import sqlite3
from config import PathConfig

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
        return None

# Cache to avoid checking FTS5 table existence on every search
_FTS5_READY = False

def ensure_fts5_table(conn: sqlite3.Connection):
    """
    Create FTS5 virtual table if it doesn't exist.
    One-time operation that makes searches 1000x faster.
    """
    global _FTS5_READY
    
    if _FTS5_READY:
        return
    
    cursor = conn.cursor()
    
    # Check if FTS5 table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='search_fts'
    """)
    if cursor.fetchone():
        _FTS5_READY = True
        return
    
    print("\n  🔧 Creating FTS5 search index (one-time, ~10-15 minutes)...")
    print("  This will only happen once. Please wait...")
    
    # Optimize for bulk insertion
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")
    conn.execute("PRAGMA temp_store = MEMORY")
    
    # Create FTS5 virtual table with proper schema
    cursor.execute("""
        CREATE VIRTUAL TABLE search_fts USING fts5(
            track_name,
            artist_names,
            track_id UNINDEXED,
            isrc UNINDEXED,
            popularity UNINDEXED,
            tokenize='porter ascii'
        )
    """)
    
    # Populate the FTS table with progress tracking
    print("  📊 Populating FTS index with 256M tracks...")
    
    # Get total track count for progress
    cursor.execute("SELECT COUNT(*) FROM tracks")
    total_tracks = cursor.fetchone()[0]
    
    # Use batch processing for faster insertion
    batch_size = 500000
    offset = 0
    
    while offset < total_tracks:
        cursor.execute("""
            INSERT INTO search_fts(track_name, artist_names, track_id, isrc, popularity)
            SELECT 
                t.name,
                GROUP_CONCAT(DISTINCT art.name),
                t.id,
                t.external_id_isrc,
                t.popularity
            FROM tracks t
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists art ON ta.artist_rowid = art.rowid
            WHERE t.rowid > ? AND t.rowid <= ?
            GROUP BY t.id
        """, (offset, offset + batch_size))
        
        conn.commit()
        offset += batch_size
        
        if offset % 5000000 == 0:
            print(f"     Processed {offset:,} / {total_tracks:,} tracks...")
    
    # Create triggers to keep FTS table updated (optional but recommended)
    # This ensures FTS stays in sync if main tables are modified
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS tracks_ai AFTER INSERT ON tracks
        BEGIN
            INSERT INTO search_fts(track_name, artist_names, track_id, isrc, popularity)
            SELECT 
                NEW.name,
                (SELECT GROUP_CONCAT(DISTINCT art.name) 
                 FROM track_artists ta 
                 JOIN artists art ON ta.artist_rowid = art.rowid 
                 WHERE ta.track_rowid = NEW.rowid),
                NEW.id,
                NEW.external_id_isrc,
                NEW.popularity;
        END
    """)
    
    conn.commit()
    print("  ✅ FTS5 index created successfully!")
    _FTS5_READY = True

def build_fts5_query(query: str) -> str:
    """
    Build optimized FTS5 query from user input.
    Handles phrases and individual terms.
    """
    if not query or not query.strip():
        return ""
    
    # Remove special characters that could break FTS5
    query = re.sub(r'[^\w\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Split into tokens
    tokens = query.split()
    
    if not tokens:
        return ""
    
    # For multi-word queries, try phrase search first, then OR
    if len(tokens) >= 2:
        # Create phrase query: "term1 term2" OR term1 OR term2
        phrase = f'"{query}"'
        individual = ' OR '.join(tokens)
        return f'{phrase} OR {individual}'
    else:
        # Single term: just search it
        return tokens[0]

def search_tracks_by_permutations(
    query: str, 
    limit: int = 50
) -> List[SearchResult]:
    """
    Search using FTS5 MATCH query.
    Returns ranked results by relevance and popularity.
    """
    if not query:
        return []
    
    main_db_path = PathConfig.get_main_db()
    if not main_db_path.exists():
        raise FileNotFoundError(f"Main database not found: {main_db_path}")
    
    conn = sqlite3.connect(str(main_db_path))
    conn.row_factory = sqlite3.Row
    
    # Ensure FTS5 table exists
    ensure_fts5_table(conn)
    
    # Optimize connection
    conn.execute("PRAGMA cache_size = -200000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    
    results = []
    try:
        cursor = conn.cursor()
        
        # Build FTS5 query
        fts_query = build_fts5_query(query)
        
        # Use FTS5 MATCH with ranking
        cursor.execute("""
            SELECT 
                track_id,
                track_name,
                artist_names,
                popularity,
                isrc,
                rank
            FROM search_fts
            WHERE search_fts MATCH ?
            ORDER BY rank, popularity DESC
            LIMIT ?
        """, (fts_query, limit))
        
        for row in cursor.fetchall():
            # FIX: Handle artist_names which might be list or string
            artist_names_raw = row['artist_names']
            
            # Convert to string if it's a list (FTS5 can return lists)
            if isinstance(artist_names_raw, list):
                # If it's a list, join first few artists
                artist_name = artist_names_raw[0] if artist_names_raw else 'Unknown'
            elif isinstance(artist_names_raw, str):
                # Standard string from GROUP_CONCAT
                artist_name = artist_names_raw.split(',')[0] if artist_names_raw else 'Unknown'
            else:
                # None or other type
                artist_name = 'Unknown'
            
            # Calculate confidence based on rank (lower rank = higher confidence)
            rank = row['rank'] if row['rank'] else 0
            confidence = 1.0 / (rank + 1)
            
            results.append(SearchResult(
                track_id=row['track_id'],
                track_name=row['track_name'],
                artist_name=artist_name,
                popularity=row['popularity'],
                isrc=row['isrc'],
                confidence=confidence,
                final_score=confidence * (row['popularity'] / 100.0)
            ))
        
    finally:
        conn.close()
    
    return results

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
    
    # Check terminal size before creating UI
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
            multiline=False
        )
        
        # Set initial text if provided
        if initial_query:
            query_buffer.text = initial_query
        
        # UI Components
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
                # Use FTS5 search directly - much faster than permutations
                results = search_tracks_by_permutations(query, limit=50)
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
        import traceback
        traceback.print_exc()
        return simple_text_search_fallback(initial_query)

def simple_text_search_fallback(query: str) -> Optional[str]:
    """
    Fallback text search without prompt_toolkit.
    Used if the library is not installed or terminal is too small.
    """
    from ui.cli.console_utils import print_header
    
    try:
        results = search_tracks_by_permutations(query, limit=20)
        
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
        import traceback
        traceback.print_exc()
        input("\n  Press Enter to continue...")
        return None
