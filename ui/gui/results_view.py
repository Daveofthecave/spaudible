# ui/gui/results_view.py
import dearpygui.dearpygui as dpg
from typing import List, Tuple, Dict, Optional, Any
from ui.gui.theme import add_gradient_button, Colors

class ResultsView:
    """Results display section with collapsible rows and export functionality."""
    
    def __init__(self, dpi_scale: float = 1.0):
        self.dpi_scale = dpi_scale
        self.tag = "results_panel"
        self.container_tag = "results_container"
        self.placeholder_tag = "results_placeholder"
        self.results = []  # Store current results
        self.expanded_rows = set()  # Track which rows are expanded
        
    def _s(self, value) -> int:
        """Scale a pixel value by DPI scale factor."""
        return int(value * self.dpi_scale)
    
    def build(self, parent: Optional[str] = None):
        """Build the results display section."""
        dpg.add_text("Results", color=(100, 200, 255))
        dpg.add_separator()
        
        # Expand/Collapse all button
        with dpg.group(horizontal=True):
            add_gradient_button(
                label="Expand All",
                width=self._s(100),
                height=self._s(24),
                callback=self._expand_all_results
            )
            add_gradient_button(
                label="Collapse All",
                width=self._s(100),
                height=self._s(24),
                callback=self._collapse_all_results
            )
            add_gradient_button(
                label="Save Playlist",
                width=self._s(120),
                height=self._s(24),
                callback=self._save_playlist
            )
        
        dpg.add_spacer(height=self._s(5))
        
        # Results container
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False
        ):
            dpg.add_text(
                tag=self.placeholder_tag,
                default_value="Enter a search query above to find similar songs.",
                color=(150, 150, 150)
            )
    
    def clear_results(self):
        """Clear all results and show placeholder."""
        self.results = []
        self.expanded_rows.clear()
        # Clear container children except placeholder
        if dpg.does_item_exist(self.container_tag):
            children = dpg.get_item_children(self.container_tag)
            if children:
                for child in children:
                    if child != dpg.get_alias_id(self.placeholder_tag):
                        dpg.delete_item(child)
        
        if dpg.does_item_exist(self.placeholder_tag):
            dpg.set_value(self.placeholder_tag, "Enter a search query above to find similar songs.")
            dpg.configure_item(self.placeholder_tag, show=True)
    
    def update_results(self, results: List[Tuple], query_info: Optional[Dict] = None):
        """Update results display with new search results.
        
        Args:
            results: List of (track_id, similarity, metadata) tuples
            query_info: Optional dict with query track info
        """
        self.results = results
        self.clear_results()
        
        if not results:
            if dpg.does_item_exist(self.placeholder_tag):
                dpg.set_value(self.placeholder_tag, "No results found.")
            return
        
        # Hide placeholder
        if dpg.does_item_exist(self.placeholder_tag):
            dpg.configure_item(self.placeholder_tag, show=False)
        
        # Build result rows
        for i, result in enumerate(results):
            if len(result) == 3:
                track_id, similarity, metadata = result
                track_name = metadata.get('track_name', 'Unknown')
                artist_name = metadata.get('artist_name', 'Unknown')
                year = metadata.get('album_release_year', '')
            else:
                track_id, similarity = result
                track_name = "Unknown"
                artist_name = "Unknown"
                year = ""
            
            # Determine color based on similarity
            color = self._get_similarity_color(similarity)
            
            # Create collapsible row
            row_tag = f"result_row_{i}"
            with dpg.tree_node(
                label=f"{i+1}. {color} {similarity:.4f} - {track_name} - {artist_name}",
                parent=self.container_tag,
                tag=row_tag,
                default_open=False
            ):
                # Expanded content
                dpg.add_text(f"Track ID: {track_id}")
                dpg.add_text(f"Similarity: {similarity:.4f}")
                if year:
                    dpg.add_text(f"Year: {year}")
                # TODO: Add Spotify URL, genre, etc.
    
    def _get_similarity_color(self, similarity: float) -> str:
        """Get color indicator for similarity score."""
        if similarity >= 0.9997:
            return "🔵"  # Blue - identical match
        elif similarity >= 0.85:
            return "🟢"  # Green - excellent match
        elif similarity >= 0.7:
            return "🟡"  # Yellow - good match
        elif similarity >= 0.65:
            return "🟠"  # Orange - decent match
        elif similarity >= 0.5:
            return "🔴"  # Red - poor match
        else:
            return "🟣"  # Purple - terrible match
    
    def _expand_all_results(self, sender=None, app_data=None):
        """Expand all result rows."""
        # DPG doesn't have a direct way to expand all tree nodes
        # This is a stub for future implementation
        self.expanded_rows = set(range(len(self.results)))
        # TODO: Implement actual expansion logic
    
    def _collapse_all_results(self, sender=None, app_data=None):
        """Collapse all result rows."""
        self.expanded_rows.clear()
        # TODO: Implement actual collapse logic
    
    def _save_playlist(self, sender=None, app_data=None):
        """Save current results as playlist."""
        if not self.results:
            return
        
        # TODO: Implement playlist saving
        # This should call the existing save_playlist logic from CLI utils
        pass
    
    def show_loading(self, message: str = "Searching..."):
        """Show loading state."""
        if dpg.does_item_exist(self.placeholder_tag):
            dpg.set_value(self.placeholder_tag, message)
            dpg.configure_item(self.placeholder_tag, show=True)
    
    def hide_loading(self):
        """Hide loading state."""
        if dpg.does_item_exist(self.placeholder_tag) and not self.results:
            dpg.configure_item(self.placeholder_tag, show=True)
        elif dpg.does_item_exist(self.placeholder_tag):
            dpg.configure_item(self.placeholder_tag, show=False)
