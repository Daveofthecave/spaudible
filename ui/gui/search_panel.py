# ui/gui/search_panel.py
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional, List
import dearpygui.dearpygui as dpg
from ui.gui.theme import add_gradient_button
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
        
        # Thread-safe result marshalling
        self._text_search_complete = False
        self._text_search_error: Optional[str] = None

    def _s(self, value) -> int:
        """Scale value - ensure native Python int."""
        return int(float(value) * self.dpi_scale)

    def update(self):
        """Called from main thread to check for completed text searches."""
        if self._text_search_complete:
            self._text_search_complete = False
            if self._text_search_error:
                print(f"Text search error: {self._text_search_error}")
                self._populate_suggestions([])
                self._set_mode(SearchMode.SHOWING_SUGGESTIONS)
                self._text_search_error = None
            else:
                self._populate_suggestions(self._current_results)
                self._set_mode(SearchMode.SHOWING_SUGGESTIONS)

    def build(self):
        """Build with horizontal layout."""
        # Header
        dpg.add_text("What would you like to find similar songs for?")
        dpg.add_spacer(height=self._s(10))
        # Options list
        dpg.add_text(" Song, artist, or album (eg. Muse Knights of Cydonia)")
        dpg.add_text(" Spotify track URL https://open.spotify.com/track/...")
        dpg.add_text(" Spotify track ID (eg. 0eGsygTp906u18L0Oimnem)")
        dpg.add_text(" ISRC code (eg. GBARL9300135)")
        dpg.add_text(" Audio file (drag-and-drop /path/to/song.mp3)")
        dpg.add_spacer(height=self._s(20))
        # Centered Search Input
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=self._s(50))
            dpg.add_input_text(
                tag=self.input_tag,
                hint="Enter search query...",
                width=self._s(400),
            )
            dpg.add_spacer(width=self._s(50))
        dpg.add_spacer(height=self._s(10))
        # Button container
        with dpg.group(tag=self.button_container_tag, horizontal=True):
            self._build_idle_buttons()
        dpg.add_spacer(height=self._s(5))
        # Status text
        dpg.add_text("", tag=self.status_tag, show=False, color=(150, 255, 150))
        # Suggestions Container
        dpg.add_spacer(height=self._s(10))
        with dpg.child_window(
            tag=self.suggestions_container_tag,
            height=self._s(300),
            width=self._s(500),
            horizontal_scrollbar=False,
            show=False,
        ):
            pass
        dpg.add_spacer(height=self._s(20))

    def _set_mode(self, new_mode: SearchMode):
        """Transition UI to new mode."""
        self.mode = new_mode
        self._clear_container(self.button_container_tag)
        parent = self.button_container_tag
        if new_mode == SearchMode.IDLE:
            self._build_idle_buttons(parent=parent)
            dpg.configure_item(self.suggestions_container_tag, show=False)
            dpg.configure_item(self.status_tag, show=False)
            self._clear_container(self.suggestions_container_tag)
        elif new_mode == SearchMode.TEXT_SEARCHING:
            self._build_searching_buttons(parent=parent)
            dpg.configure_item(self.suggestions_container_tag, show=False)
            dpg.set_value(self.status_tag, "Searching database...")
            dpg.configure_item(self.status_tag, show=True)
        elif new_mode == SearchMode.SHOWING_SUGGESTIONS:
            self._build_refinement_buttons(parent=parent)
            dpg.configure_item(self.suggestions_container_tag, show=True)
            dpg.configure_item(self.status_tag, show=False)
        elif new_mode == SearchMode.SIMILARITY_SEARCHING:
            self._build_cancel_only_button(parent=parent)
            dpg.configure_item(self.suggestions_container_tag, show=False)
            dpg.set_value(self.status_tag, "Scanning 256M vectors for similar songs...")
            dpg.configure_item(self.status_tag, show=True)

    def _clear_container(self, tag: str):
        """Remove all children from a container."""
        if dpg.does_item_exist(tag):
            children = dpg.get_item_children(tag)
            if children:
                if isinstance(children, list):
                    for child in children:
                        if dpg.does_item_exist(child):
                            dpg.delete_item(child)
                elif isinstance(children, dict):
                    for slot in children.values():
                        for child in slot:
                            if dpg.does_item_exist(child):
                                dpg.delete_item(child)

    def _build_idle_buttons(self, parent=None):
        """Single 'Find Similar Songs' button."""
        add_gradient_button(
            label="Find Similar Songs",
            width=self._s(180),
            height=self._s(28),
            callback=self._handle_search,
            parent=parent,
        )

    def _build_searching_buttons(self, parent=None):
        """Cancel button during text search."""
        add_gradient_button(
            label="Cancel",
            width=self._s(100),
            height=self._s(28),
            callback=self._handle_cancel,
            parent=parent,
        )

    def _build_refinement_buttons(self, parent=None):
        """Refine Query and Cancel buttons."""
        with dpg.group(horizontal=True, parent=parent):
            add_gradient_button(
                label="Refine Query",
                width=self._s(140),
                height=self._s(28),
                callback=self._handle_refine,
            )
            dpg.add_spacer(width=self._s(20))
            add_gradient_button(
                label="Cancel",
                width=self._s(100),
                height=self._s(28),
                callback=self._handle_cancel,
            )

    def _build_cancel_only_button(self, parent=None):
        """Single cancel button during heavy similarity search."""
        add_gradient_button(
            label="Cancel Search",
            width=self._s(140),
            height=self._s(28),
            callback=self._handle_cancel_similarity,
            parent=parent,
        )

    def _on_search_enter(self, sender, app_data):
        """Handle Enter key in search box."""
        if app_data and self.mode in [SearchMode.IDLE, SearchMode.SHOWING_SUGGESTIONS]:
            self._handle_search()

    def _handle_search(self):
        """Start text search for suggestions."""
        query = self.get_query().strip()
        if not query:
            return
        if self.mode == SearchMode.TEXT_SEARCHING:
            return
        self._set_mode(SearchMode.TEXT_SEARCHING)
        self._cancel_event.clear()
        self._text_search_complete = False
        self._search_thread = threading.Thread(
            target=self._text_search_worker,
            args=(query,),
            daemon=True,
        )
        self._search_thread.start()

    def _text_search_worker(self, query: str):
        """Background thread for inverted index lookup."""
        try:
            time.sleep(0.1)
            if self._cancel_event.is_set():
                return
            results = search_tracks_flexible(query, limit=50)
            if self._cancel_event.is_set():
                return
            self._current_results = results
            # Signal completion to main thread
            self._text_search_complete = True
        except Exception as e:
            print(f"Text search error: {e}")
            if not self._cancel_event.is_set():
                self._text_search_error = str(e)
                self._text_search_complete = True

    def _populate_suggestions(self, results: List[SearchResult]):
        """Fill the scrollable list with selectable items."""
        # Clear existing content instead of deleting the container
        self._clear_container(self.suggestions_container_tag)
        
        # Show the container
        if dpg.does_item_exist(self.suggestions_container_tag):
            dpg.configure_item(self.suggestions_container_tag, show=True)
        
        # Add content to the existing container by passing parent explicitly
        if not results:
            dpg.add_text(
                "No results found. Try a different search term.",
                color=(200, 100, 100),
                parent=self.suggestions_container_tag,  # Explicit parent
            )
            return
        
        dpg.add_text(
            "Select a song to start the similarity search:",
            color=(200, 200, 200),
            parent=self.suggestions_container_tag,  # Explicit parent
        )
        dpg.add_separator(parent=self.suggestions_container_tag)
        
        for result in results:
            display_text = f"{result.track_name} - {result.artist_name}"
            if result.album_name:
                display_text += f" - {result.album_name}"
            if result.album_release_year:
                display_text += f" ({result.album_release_year})"
            
            dpg.add_selectable(
                label=display_text,
                callback=self._on_suggestion_clicked,
                user_data=result,
                height=self._s(24),
                parent=self.suggestions_container_tag,  # Explicit parent
            )

    def _on_suggestion_clicked(self, sender, app_data, user_data):
        """User clicked a suggestion."""
        if not user_data or self.mode != SearchMode.SHOWING_SUGGESTIONS:
            return
        result: SearchResult = user_data
        self._set_mode(SearchMode.SIMILARITY_SEARCHING)
        if self.on_suggestion_selected:
            self.on_suggestion_selected(result)

    def _handle_refine(self):
        """Refine Query button - go back to idle but keep text."""
        self._set_mode(SearchMode.IDLE)
        if dpg.does_item_exist(self.input_tag):
            dpg.focus_item(self.input_tag)

    def _handle_cancel(self):
        """Cancel button during suggestion phase."""
        self._cancel_event.set()
        self._set_mode(SearchMode.IDLE)
        self.set_query("")
        self._current_results = []
        if self.on_cancel:
            self.on_cancel()

    def _handle_cancel_similarity(self):
        """Cancel button during heavy similarity search."""
        self._set_mode(SearchMode.IDLE)
        if self.on_cancel:
            self.on_cancel()

    def get_query(self) -> str:
        """Get current search query text."""
        if dpg.does_item_exist(self.input_tag):
            return dpg.get_value(self.input_tag)
        return ""

    def set_query(self, text: str):
        """Set search query text."""
        if dpg.does_item_exist(self.input_tag):
            dpg.set_value(self.input_tag, text)

    def reset_to_idle(self):
        """Force reset to idle state."""
        self._set_mode(SearchMode.IDLE)

    def set_similarity_mode(self):
        """Call this when MainWindow starts the similarity search."""
        self._set_mode(SearchMode.SIMILARITY_SEARCHING)
