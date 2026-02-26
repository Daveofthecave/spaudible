# ui/gui/search_panel.py
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional, List
import dearpygui.dearpygui as dpg
from core.utilities.text_search_utils import search_tracks_flexible, SearchResult

class SearchMode(Enum):
    IDLE = auto()
    TEXT_SEARCHING = auto()
    SHOWING_SUGGESTIONS = auto()
    SIMILARITY_SEARCHING = auto()

class SearchPanel:
    def __init__(
        self,
        dpi_scale: float = 1.0,
        on_suggestion_selected: Optional[Callable[[SearchResult], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        self.dpi_scale = float(dpi_scale)
        self.tag = "search_panel"
        self.mode = SearchMode.IDLE
        self.on_suggestion_selected = on_suggestion_selected
        self.on_cancel = on_cancel
        
        self.input_tag = "search_input"
        self.button_container_tag = "search_button_container"
        self.suggestions_container_tag = "suggestions_container"
        self.status_tag = "search_status_text"
        
        self._search_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._current_results: List[SearchResult] = []

    def _s(self, value) -> int:
        """Scale value - ensure Python int."""
        return int(float(value) * self.dpi_scale)

    def build(self):
        """Build UI - NO parent parameter, relies on context manager."""
        # Header - NO parent=parent
        dpg.add_text("What would you like to find similar songs for?")
        dpg.add_spacer(height=self._s(10))
        
        # Input row
        dpg.add_text("Enter song, artist, track ID, ISRC, or drag audio file:")
        dpg.add_input_text(
            tag=self.input_tag,
            hint="Search query...",
            width=self._s(400),
        )
        dpg.add_spacer(height=self._s(10))
        
        # Button container
        with dpg.group(tag=self.button_container_tag, horizontal=True):
            dpg.add_button(
                label="Find Similar Songs",
                callback=self._handle_search,
                width=self._s(180),
                height=self._s(28),
            )
        
        # Status text
        dpg.add_spacer(height=self._s(5))
        dpg.add_text("", tag=self.status_tag, show=False, color=(150, 255, 150))
        
        # Suggestions container
        dpg.add_spacer(height=self._s(10))
        with dpg.child_window(
            tag=self.suggestions_container_tag,
            height=self._s(300),
            width=self._s(500),
            horizontal_scrollbar=False,
            show=False,
        ):
            pass

    def _handle_search(self):
        """Start text search."""
        query = self.get_query().strip()
        if not query:
            return
            
        self._set_mode(SearchMode.TEXT_SEARCHING)
        self._cancel_event.clear()
        self._search_thread = threading.Thread(
            target=self._text_search_worker,
            args=(query,),
            daemon=True,
        )
        self._search_thread.start()

    def _text_search_worker(self, query: str):
        """Background thread."""
        try:
            time.sleep(0.1)
            if self._cancel_event.is_set():
                return
                
            results = search_tracks_flexible(query, limit=50)
            if self._cancel_event.is_set():
                return
                
            self._current_results = results
            self._populate_suggestions(results)
            self._set_mode(SearchMode.SHOWING_SUGGESTIONS)
        except Exception as e:
            print(f"Search error: {e}")
            self._set_mode(SearchMode.IDLE)

    def _populate_suggestions(self, results: List[SearchResult]):
        """Fill suggestion list."""
        # Clear existing
        if dpg.does_item_exist(self.suggestions_container_tag):
            dpg.delete_item(self.suggestions_container_tag)
            
        # Recreate container with results
        with dpg.child_window(
            tag=self.suggestions_container_tag,
            height=self._s(300),
            width=self._s(500),
            horizontal_scrollbar=False,
        ):
            if not results:
                dpg.add_text("No results found.")
                return
                
            dpg.add_text("Select a song:")
            dpg.add_separator()
            
            for result in results:
                display = f"{result.track_name} - {result.artist_name}"
                dpg.add_selectable(
                    label=display,
                    callback=self._on_suggestion_clicked,
                    user_data=result,
                    height=self._s(24),
                )

    def _on_suggestion_clicked(self, sender, app_data, user_data):
        """Handle selection."""
        if not user_data or self.mode != SearchMode.SHOWING_SUGGESTIONS:
            return
        self._set_mode(SearchMode.SIMILARITY_SEARCHING)
        if self.on_suggestion_selected:
            self.on_suggestion_selected(user_data)

    def _set_mode(self, new_mode: SearchMode):
        """Update UI based on mode."""
        self.mode = new_mode
        
        if new_mode == SearchMode.IDLE:
            dpg.configure_item(self.suggestions_container_tag, show=False)
            dpg.configure_item(self.status_tag, show=False)
        elif new_mode == SearchMode.TEXT_SEARCHING:
            dpg.configure_item(self.suggestions_container_tag, show=False)
            dpg.set_value(self.status_tag, "Searching...")
            dpg.configure_item(self.status_tag, show=True)
        elif new_mode == SearchMode.SHOWING_SUGGESTIONS:
            dpg.configure_item(self.suggestions_container_tag, show=True)
            dpg.configure_item(self.status_tag, show=False)
        elif new_mode == SearchMode.SIMILARITY_SEARCHING:
            dpg.configure_item(self.suggestions_container_tag, show=False)
            dpg.set_value(self.status_tag, "Finding similar songs...")
            dpg.configure_item(self.status_tag, show=True)

    def _handle_cancel(self):
        """Cancel button."""
        self._cancel_event.set()
        self._set_mode(SearchMode.IDLE)
        self.set_query("")
        if self.on_cancel:
            self.on_cancel()

    def get_query(self) -> str:
        """Get input text."""
        if dpg.does_item_exist(self.input_tag):
            return dpg.get_value(self.input_tag)
        return ""

    def set_query(self, text: str):
        """Set input text."""
        if dpg.does_item_exist(self.input_tag):
            dpg.set_value(self.input_tag, text)

    def reset_to_idle(self):
        """Reset state."""
        self._set_mode(SearchMode.IDLE)

    def set_similarity_mode(self):
        """Set similarity mode."""
        self._set_mode(SearchMode.SIMILARITY_SEARCHING)
