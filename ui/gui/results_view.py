# ui/gui/results_view.py
import dearpygui.dearpygui as dpg
from typing import List, Tuple, Optional
from ui.gui.theme import add_gradient_button

class ResultsView:
    """Results display with table layout, expandable rows, and gradient color scoring."""
    
    def __init__(self, dpi_scale: float = 1.0):
        self.dpi_scale = float(dpi_scale)
        self.tag = "results_panel"
        self.container_tag = "results_container"
        self.action_bar_tag = "results_action_bar"
        self.results = []
        
        # Track expandable row state
        self._detail_groups: List[int] = []
        self._arrow_tags: List[int] = []
        self._all_expanded = False
        self._toggle_button_tag = "results_toggle_all_btn"
        
        # Column widths (scaled)
        self._col_rank = 70
        self._col_score = 110
        
    def _s(self, value) -> int:
        """Scale a pixel value - ensure native Python int."""
        return int(float(value) * self.dpi_scale)
        
    def _hsl_to_rgb(self, h: float, s: float, l: float) -> Tuple[int, int, int]:
        """Convert HSL (0-360, 0-1, 0-1) to RGB tuple."""
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        return (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255),
        )
        
    def _get_similarity_color(self, similarity: float) -> Tuple[int, int, int]:
        """Map similarity 0.0-1.0 to Purple (low) -> Cyan (high)."""
        s = max(0.0, min(1.0, similarity))
        hue = 270 - (s * 90)
        return self._hsl_to_rgb(hue, 1.0, 0.5)
        
    def build(self):
        """Build the results table structure."""
        # Header row with columns and toggle button
        with dpg.group(horizontal=True):
            dpg.add_text("Rank")
            dpg.add_spacer(width=self._s(self._col_rank - 40))
            dpg.add_text("Score")
            dpg.add_spacer(width=self._s(self._col_score - 50))
            dpg.add_text("Song")
            # Push button to right with spacer
            dpg.add_spacer(width=-1)
            # Toggle button
            add_gradient_button(
                label="Expand All",
                tag=self._toggle_button_tag,
                width=self._s(100),
                height=self._s(24),
                callback=self._toggle_all,
            )
        dpg.add_separator()
        
        # Results container - starts hidden
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False,
            show=False,
        ):
            pass
            
    def _clear_results_container(self):
        """Remove all children from results container."""
        if dpg.does_item_exist(self.container_tag):
            children = dpg.get_item_children(self.container_tag)
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
            self._detail_groups.clear()
            self._arrow_tags.clear()
            
    def update_results(self, results: List[Tuple]):
        """Populate table with results, color-coded by similarity."""
        self.results = results
        self._clear_results_container()
        
        # Show container
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
            
        # Update toggle button label
        if dpg.does_item_exist(self._toggle_button_tag):
            label = "Collapse All" if self._all_expanded else "Expand All"
            dpg.configure_item(self._toggle_button_tag, label=label)
            
        if not results:
            dpg.add_text(
                "No similar songs found.",
                parent=self.container_tag,
            )
            return
            
        # Result count
        dpg.add_text(
            f"Found {len(results)} similar songs:",
            color=(200, 200, 200),
            parent=self.container_tag,
        )
        dpg.add_separator(parent=self.container_tag)
        
        # Populate rows
        for i, result in enumerate(results, 1):
            if len(result) == 3:
                track_id, similarity, metadata = result
                track_name = metadata.get("track_name", "Unknown")
                artist_name = metadata.get("artist_name", "Unknown")
                album = metadata.get("album_name", "")
                year = metadata.get("album_release_year", "")
            else:
                track_id, similarity = result
                track_name = "Unknown"
                artist_name = "Unknown"
                album = ""
                year = ""
                
            # Row container
            row_container = dpg.add_group(
                horizontal=False,
                parent=self.container_tag,
            )
            detail_tag = dpg.generate_uuid()
            
            # Main row (horizontal)
            with dpg.group(horizontal=True, parent=row_container):
                # Rank
                dpg.add_text(str(i))
                dpg.add_spacer(width=self._s(20))
                
                # Score with color
                score_color = self._get_similarity_color(similarity)
                dpg.add_text(
                    f"{similarity:.4f}",
                    color=score_color,
                )
                dpg.add_spacer(width=self._s(20))
                
                # Song info
                song_text = f"{track_name} - {artist_name}"
                if album and year:
                    song_text += f" - {album} ({year})"
                elif album:
                    song_text += f" - {album}"
                elif year:
                    song_text += f" ({year})"
                max_len = 45
                if len(song_text) > max_len:
                    song_text = song_text[:max_len - 3] + "..."
                dpg.add_text(song_text)
                
                # Push arrow to right
                dpg.add_spacer(width=-1)
                
                # Arrow indicator
                arrow_lbl = "▼" if self._all_expanded else "▶"
                arrow_txt = dpg.add_text(
                    arrow_lbl,
                    color=(200, 200, 200),
                )
                self._arrow_tags.append(arrow_txt)
                
            # Clickable invisible button for the row
            dpg.add_button(
                label="",
                width=self._s(700),
                height=self._s(24),
                callback=self._toggle_row,
                user_data=(detail_tag, arrow_txt),
                parent=row_container,
            )
            
            # Detail row (expandable)
            with dpg.group(
                tag=detail_tag,
                show=self._all_expanded,
                parent=row_container,
                indent=self._s(20),
            ):
                dpg.add_text(
                    f"Spotify URL: https://open.spotify.com/track/{track_id}",
                    color=(150, 150, 150),
                )
                if album:
                    dpg.add_text(f"Album: {album}", color=(150, 150, 150))
                if year:
                    dpg.add_text(f"Released: {year}", color=(150, 150, 150))
                dpg.add_text("—" * 40, color=(80, 80, 80))
                
            self._detail_groups.append(detail_tag)
            
    def _toggle_row(self, sender, app_data, user_data):
        """Toggle individual row expansion."""
        detail_tag, arrow_tag = user_data
        if dpg.does_item_exist(detail_tag):
            current = dpg.get_item_configuration(detail_tag).get("show", False)
            new_state = not current
            dpg.configure_item(detail_tag, show=new_state)
            if dpg.does_item_exist(arrow_tag):
                dpg.set_value(arrow_tag, "▼" if new_state else "▶")
                
    def _toggle_all(self):
        """Expand or collapse all rows."""
        self._all_expanded = not self._all_expanded
        
        # Update button label
        if dpg.does_item_exist(self._toggle_button_tag):
            label = "Collapse All" if self._all_expanded else "Expand All"
            dpg.configure_item(self._toggle_button_tag, label=label)
            
        # Update all rows
        for detail_tag, arrow_tag in zip(self._detail_groups, self._arrow_tags):
            if dpg.does_item_exist(detail_tag):
                dpg.configure_item(detail_tag, show=self._all_expanded)
            if dpg.does_item_exist(arrow_tag):
                dpg.set_value(arrow_tag, "▼" if self._all_expanded else "▶")
                
    def clear_results(self):
        """Clear results and reset to empty state."""
        self.results = []
        self._all_expanded = False
        self._detail_groups.clear()
        self._arrow_tags.clear()
        
        if dpg.does_item_exist(self.action_bar_tag):
            dpg.configure_item(self.action_bar_tag, show=False)
        self._clear_results_container()
        
        # Reset button
        if dpg.does_item_exist(self._toggle_button_tag):
            dpg.configure_item(self._toggle_button_tag, label="Expand All")
            
        # Hide container initially
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=False)
            
    def show_loading(self, message: str = "Searching..."):
        """Show loading state."""
        if dpg.does_item_exist(self.action_bar_tag):
            dpg.configure_item(self.action_bar_tag, show=False)
        self._clear_results_container()
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
        dpg.add_text(message, color=(150, 255, 150), parent=self.container_tag)
        
    def hide_loading(self):
        """Hide loading state."""
        pass
