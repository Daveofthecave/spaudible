# ui/cli/menu_system/main_menu_handlers/text_search.py
"""
Interactive text search interface for Spaudible.
Provides arrow-key navigation, query editing, and track selection.
"""
from typing import Optional, List
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.keys import Keys
from core.utilities.text_search_utils import search_tracks_by_permutations, SearchResult

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
            multiline=False
        )
        
        # Set initial text and cursor position
        if initial_query:
            query_buffer.text = initial_query
            query_buffer.cursor_position = len(initial_query)  # Cursor at end
        
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
        
        # FIX: Navigation in results list (global, auto-switches focus)
        @kb.add('up')
        def move_up(event):
            """Navigate up in results list (auto-switches from query field)"""
            nonlocal selected_idx
            if results:
                # If in query window, switch to results first
                if event.app.layout.current_window == query_window:
                    event.app.layout.focus(results_window)
                # Then navigate up
                selected_idx = max(0, selected_idx - 1)
                update_results_display()
        
        @kb.add('down')
        def move_down(event):
            """Navigate down in results list (auto-switches from query field)"""
            nonlocal selected_idx
            if results:
                # If in query window, switch to results first
                if event.app.layout.current_window == query_window:
                    event.app.layout.focus(results_window)
                # Then navigate down
                selected_idx = min(len(results) - 1, selected_idx + 1)
                update_results_display()
        
        # FIX: Cursor movement in query field (always works, switches focus)
        @kb.add('left')
        def move_left(event):
            """Move cursor left in query field (switches focus if needed)"""
            event.app.layout.focus(query_window)
            query_buffer.cursor_position = max(0, query_buffer.cursor_position - 1)
        
        @kb.add('right')
        def move_right(event):
            """Move cursor right in query field (switches focus if needed)"""
            event.app.layout.focus(query_window)
            query_buffer.cursor_position = min(len(query_buffer.text), query_buffer.cursor_position + 1)
        
        # Backspace handling
        @kb.add('backspace')
        def handle_backspace(event):
            """Handle backspace in query field"""
            if event.app.layout.current_window == query_window:
                if query_buffer.cursor_position > 0:
                    query_buffer.delete_before_cursor()
            else:
                # Switch focus to query field
                event.app.layout.focus(query_window)
        
        # Delete handling
        @kb.add('delete')
        def handle_delete(event):
            """Handle delete in query field"""
            if event.app.layout.current_window == query_window:
                query_buffer.delete()
        
        # Enter key handling
        @kb.add('enter')
        def handle_enter(event):
            """Handle Enter key - context-aware behavior"""
            current_query = query_buffer.text.strip()
            
            # If we're in the query field
            if event.app.layout.current_window == query_window:
                # Always perform search when in query field
                perform_search()
                # Move focus to results after search
                event.app.layout.focus(results_window)
            else:
                # We're in results field
                # Check if query was modified since last search
                if current_query != last_search_query:
                    # Query changed - re-run search instead of selecting
                    perform_search()
                    event.app.layout.focus(results_window)
                else:
                    # Query unchanged - select track
                    if results and selected_idx < len(results):
                        event.app.exit(result=results[selected_idx].track_id)
        
        # Cancel keys
        @kb.add('c-c')
        def handle_cancel(event):
            """Cancel search and return to main menu"""
            event.app.exit(result=None)
        
        @kb.add('escape')
        def handle_escape(event):
            """Escape key also cancels"""
            event.app.exit(result=None)
        
        # FIX: Catch-all for printable characters - route to query field
        @kb.add(Keys.Any)
        def handle_typing(event):
            """Any printable character switches to query field and inserts character"""
            # Switch focus to query field
            event.app.layout.focus(query_window)
            
            # Insert the character into the buffer at cursor position
            query_buffer.insert_text(event.data, overwrite=False)
        
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
        # CHANGED: Pass query string directly, not permutations
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
