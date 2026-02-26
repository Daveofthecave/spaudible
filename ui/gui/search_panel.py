# ui/gui/search_panel.py
import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Callable, Optional
from ui.gui.theme import add_gradient_button

class SearchPanel:
    """Search input section with query box and action buttons."""
    
    def __init__(self, dpi_scale: float = 1.0, on_search: Optional[Callable] = None):
        self.dpi_scale = dpi_scale
        self.tag = "search_panel"
        self.on_search_callback = on_search
        self.input_tag = "search_input"
        
    def _s(self, value) -> int:
        """Scale a pixel value by DPI scale factor."""
        return int(value * self.dpi_scale)
    
    def build(self, parent: Optional[str] = None):
        """Build the search input section."""
        dpg.add_text("Search", color=(100, 200, 255))
        dpg.add_separator()
        
        dpg.add_input_text(
            tag=self.input_tag,
            hint="Enter song, artist, track ID, ISRC, or drag audio file...",
            width=-1,  # Fill width
            callback=self._on_search_enter,
            on_enter=True
        )
        
        with dpg.group(horizontal=True):
            add_gradient_button(
                tag="search_button",
                label="Find Similar Songs",
                width=self._s(150),
                height=self._s(24),
                callback=self._handle_search
            )
            add_gradient_button(
                label="Clear",
                width=self._s(80),
                height=self._s(24),
                callback=self._clear_search
            )
        
        dpg.add_spacer(height=self._s(10))
    
    def get_query(self) -> str:
        """Get current search query text."""
        return dpg.get_value(self.input_tag) if dpg.does_item_exist(self.input_tag) else ""
    
    def set_query(self, text: str):
        """Set search query text."""
        if dpg.does_item_exist(self.input_tag):
            dpg.set_value(self.input_tag, text)
    
    def _handle_search(self, sender=None, app_data=None):
        """Handle search button click."""
        query = self.get_query()
        if not query.strip():
            # Set placeholder text or show error
            if dpg.does_item_exist("results_placeholder"):
                dpg.set_value("results_placeholder", "Please enter a search query.")
            return
        
        # TODO: Integrate with actual search logic from core.similarity_engine
        if self.on_search_callback:
            self.on_search_callback(query)
        else:
            # Fallback: just update placeholder
            if dpg.does_item_exist("results_placeholder"):
                dpg.set_value("results_placeholder", f"Searching for: {query}...\n\n(Integration pending)")
    
    def _on_search_enter(self, sender, app_data):
        """Handle Enter key in search box."""
        if app_data:  # Only trigger if there's text
            self._handle_search()
    
    def _clear_search(self, sender=None, app_data=None):
        """Clear the search input."""
        self.set_query("")
        if dpg.does_item_exist(self.input_tag):
            dpg.focus_item(self.input_tag)
    
    def focus(self):
        """Focus the search input."""
        if dpg.does_item_exist(self.input_tag):
            dpg.focus_item(self.input_tag)
