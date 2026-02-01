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
from prompt_toolkit.widgets import SearchToolbar
from prompt_toolkit.formatted_text import HTML, merge_formatted_text
from prompt_toolkit.styles import Style
from core.utilities.text_search_utils import search_tracks_by_permutations, parse_query_permutations, SearchResult

def interactive_text_search(initial_query: str = "") -> Optional[str]:
    """
    Interactive text search with arrow-key navigation and query editing.
    Fixed for prompt_toolkit compatibility.
    """
    
    # State management
    results: List[SearchResult] = []
    selected_idx = 0
    query = initial_query
    
    # Create query buffer for editing
    query_buffer = Buffer(
        multiline=False,
        accept_handler=lambda buf: perform_search()
    )
    
    # Set initial text if provided
    if initial_query:
        query_buffer.text = initial_query
    
    # UI Components
    query_control = BufferControl(buffer=query_buffer)
    results_control = FormattedTextControl(text="")
    status_control = FormattedTextControl(text="")
    
    # Key bindings
    kb = KeyBindings()
    
    def perform_search():
        """Execute search and update results display"""
        nonlocal results, selected_idx, query
        
        query = query_buffer.text.strip()
        if not query:
            results_control.text = "Enter a search query above..."
            return
        
        try:
            # Parse permutations and search
            permutations = parse_query_permutations(query)
            results = search_tracks_by_permutations(permutations, limit=50)
            selected_idx = 0
            
            if not results:
                results_control.text = f"No results found for '{query}'"
            else:
                update_results_display()
        except Exception as e:
            results_control.text = f"Search error: {str(e)}"
            results = []
    
    def update_results_display():
        """Update the results list with current selection"""
        if not results:
            results_control.text = "No results"
            return
        
        lines = []
        display_count = min(len(results), 20)
        
        for i in range(display_count):
            prefix = "→" if i == selected_idx else "  "
            result = results[i]
            lines.append(f"{prefix} {result.display_text}")
        
        if len(results) > 20:
            lines.append(f"  ... and {len(results) - 20} more")
        
        results_control.text = "\n".join(lines)
    
    def update_status_bar():
        """Update status bar text"""
        if not query:
            status = "Enter a song/artist query"
        else:
            status = f"Query: {query} | {len(results)} results"
        
        status += " | ↑↓ Navigate | Enter=Select | Ctrl+C=Cancel | Backspace=Edit"
        status_control.text = status
    
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
        if event.app.layout.current_window.content == query_control:
            perform_search()
            event.app.layout.focus(results_control.window)
        else:
            # Results field is focused: select track
            if results and selected_idx < len(results):
                event.app.exit(result=results[selected_idx].track_id)
    
    @kb.add('backspace')
    def handle_backspace(event):
        """Switch focus to query field for editing"""
        event.app.layout.focus(query_control.window)
    
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
            # Query field
            Window(
                height=1,
                content=FormattedTextControl(text="Search query:"),
                style="class:query-label"
            ),
            Window(
                height=1,
                content=query_control,
                style="class:query-field",
                cursorline=True
            ),
            # Results list
            Window(
                height=1,
                content=FormattedTextControl(text="Results:"),
                style="class:results-label"
            ),
            Window(
                height=20,
                content=results_control,
                style="class:results-list",
                cursorline=False,
                always_hide_cursor=True
            ),
            # Status bar
            Window(
                height=1,
                content=status_control,
                style="class:status-bar"
            )
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
        app.layout.focus(results_control.window)
    else:
        app.layout.focus(query_control.window)
    
    # Run the event loop
    result = app.run()
    
    return result  # track_id or None

# Helper function for simple fallback (if prompt_toolkit is not available)
def simple_text_search_fallback(query: str) -> Optional[str]:
    """
    Fallback text search without prompt_toolkit.
    Used if the library is not installed.
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
