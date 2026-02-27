# ui/gui/results_view.py
import dearpygui.dearpygui as dpg
from typing import List, Tuple, Optional, Dict, Any
from ui.gui.theme import add_gradient_button

class ResultsView:
    """Results display section showing similarity search results."""
    
    def __init__(self, dpi_scale: float = 1.0):
        self.dpi_scale = float(dpi_scale)
        self.tag = "results_panel"
        self.container_tag = "results_container"
        self.placeholder_tag = "results_placeholder"
        self.action_bar_tag = "results_action_bar"
        self.results = []
    
    def _s(self, value) -> int:
        """Scale a pixel value - ensure native Python int."""
        return int(float(value) * self.dpi_scale)
    
    def _clear_container(self, tag: str):
        """Remove all children from a container without deleting the container itself."""
        if dpg.does_item_exist(tag):
            children = dpg.get_item_children(tag)
            if children:
                # Handle both list and dict return types from DPG
                if isinstance(children, list):
                    for child in children:
                        if dpg.does_item_exist(child):
                            dpg.delete_item(child)
                elif isinstance(children, dict):
                    for slot in children.values():
                        for child in slot:
                            if dpg.does_item_exist(child):
                                dpg.delete_item(child)
    
    def build(self):
        """Build the results display structure once."""
        dpg.add_text("Results", color=(100, 200, 255))
        dpg.add_separator()
        
        # Action bar (hidden by default)
        with dpg.group(tag=self.action_bar_tag, horizontal=True, show=False):
            dpg.add_spacer(width=self._s(10))
            add_gradient_button(
                label="Save Playlist",
                width=self._s(120),
                height=self._s(24),
                callback=self._save_playlist
            )
            dpg.add_spacer(width=self._s(10))
        
        dpg.add_spacer(height=self._s(5))
        
        # Results container - created once and reused
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False,
            show=False
        ):
            # Initial placeholder
            dpg.add_text(
                tag=self.placeholder_tag,
                default_value="Enter a search query above to find similar songs.",
                color=(150, 150, 150)
            )
    
    def clear_results(self):
        """Clear all results and show placeholder."""
        self.results = []
        
        # Hide action bar
        if dpg.does_item_exist(self.action_bar_tag):
            dpg.configure_item(self.action_bar_tag, show=False)
        
        # Clear contents only, don't delete container
        self._clear_container(self.container_tag)
        
        # Show container and add placeholder with explicit parent
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
            dpg.add_text(
                "Enter a search query above to find similar songs.",
                tag=self.placeholder_tag,
                color=(150, 150, 150),
                parent=self.container_tag  # Explicit parent
            )
    
    def update_results(self, results: List[Tuple]):
        """Update results display with similarity search results."""
        self.results = results
        
        # Show action bar
        if dpg.does_item_exist(self.action_bar_tag):
            dpg.configure_item(self.action_bar_tag, show=True)
        
        # Clear and repopulate with explicit parenting
        self._clear_container(self.container_tag)
        
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
            
            if not results:
                dpg.add_text(
                    "No similar songs found.",
                    parent=self.container_tag  # Explicit parent
                )
                return
            
            # Header
            dpg.add_text(
                f"Found {len(results)} similar songs:",
                color=(200, 200, 200),
                parent=self.container_tag
            )
            dpg.add_separator(parent=self.container_tag)
            
            # Result rows
            for i, result in enumerate(results, 1):
                if len(result) == 3:
                    track_id, similarity, metadata = result
                    track_name = metadata.get('track_name', 'Unknown')
                    artist_name = metadata.get('artist_name', 'Unknown')
                    album = metadata.get('album_name', '')
                    year = metadata.get('album_release_year', '')
                else:
                    track_id, similarity = result
                    track_name = "Unknown"
                    artist_name = "Unknown"
                    album = ""
                    year = ""
                
                # Format text
                color_indicator = self._get_similarity_indicator(similarity)
                text = f"{i}. {color_indicator} {similarity:.4f} - {track_name} - {artist_name}"
                if album:
                    text += f" - {album}"
                if year:
                    text += f" ({year})"
                
                text_color = self._get_similarity_color(similarity)
                dpg.add_text(
                    text,
                    color=text_color,
                    parent=self.container_tag  # Explicit parent
                )
    
    def _get_similarity_indicator(self, similarity: float) -> str:
        """Get emoji indicator for similarity score."""
        if similarity >= 0.9997:
            return "🔵"
        elif similarity >= 0.85:
            return "🟢"
        elif similarity >= 0.7:
            return "🟡"
        elif similarity >= 0.65:
            return "🟠"
        elif similarity >= 0.5:
            return "🔴"
        else:
            return "🟣"
    
    def _get_similarity_color(self, similarity: float) -> tuple:
        """Get RGB color tuple for similarity score text."""
        if similarity >= 0.9997:
            return (100, 200, 255)
        elif similarity >= 0.85:
            return (100, 255, 100)
        elif similarity >= 0.7:
            return (255, 255, 100)
        elif similarity >= 0.65:
            return (255, 200, 100)
        elif similarity >= 0.5:
            return (255, 100, 100)
        else:
            return (200, 100, 255)
    
    def _save_playlist(self, sender=None, app_data=None):
        """Save current results as playlist."""
        if not self.results:
            return
        print(f"Saving playlist with {len(self.results)} tracks...")
    
    def show_loading(self, message: str = "Searching..."):
        """Show loading state."""
        if dpg.does_item_exist(self.action_bar_tag):
            dpg.configure_item(self.action_bar_tag, show=False)
        
        # Clear and show loading message with explicit parent
        self._clear_container(self.container_tag)
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
            dpg.add_text(
                message,
                color=(150, 255, 150),
                parent=self.container_tag  # Explicit parent
            )
    
    def hide_loading(self):
        """Hide loading state."""
        pass
