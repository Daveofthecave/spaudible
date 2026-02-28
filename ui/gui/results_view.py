# ui/gui/results_view.py
import dearpygui.dearpygui as dpg
from typing import List, Tuple, Optional, Any
from ui.gui.theme import add_gradient_button, Colors


class ResultsView:
    """Results display with clean table layout and expandable rows."""
    
    def __init__(self, dpi_scale: float = 1.0, header_font: Optional[Any] = None):
        self.dpi_scale = float(dpi_scale)
        self.header_font = header_font
        self.tag = "results_panel"
        self.container_tag = "results_container"
        self._toggle_button_tag = "results_toggle_all_btn"
        self.results = []
        self._detail_groups: List[int] = []
        self._arrow_tags: List[int] = []
        self._all_expanded = False
        
        # Column widths (logical pixels)
        self._w_rank = 80
        self._w_score = 100
        self._w_song = 480
        self._w_arrow = 40
        
        # Spacing between columns
        self._col_spacing = 20
    
    def _s(self, value) -> int:
        """Scale value."""
        return int(float(value) * self.dpi_scale)
    
    def _get_song_color(self, similarity: float) -> Tuple[int, int, int]:
        """Map similarity score to color gradient."""
        stops = [
            (1.0, (0, 252, 255)),   # Cyan
            (0.8, (0, 255, 0)),     # Green
            (0.6, (255, 255, 0)),   # Yellow
            (0.4, (255, 165, 0)),   # Orange
            (0.2, (255, 0, 0)),     # Red
            (0.0, (128, 0, 255)),   # Purple
        ]
        
        # Clamp similarity to valid range
        sim = max(0.0, min(1.0, similarity))
        
        # Find which segment we're in
        for i in range(len(stops) - 1):
            upper_sim, upper_color = stops[i]
            lower_sim, lower_color = stops[i + 1]
            
            if sim >= lower_sim:
                if upper_sim == lower_sim:
                    return upper_color
                # Interpolate between lower and upper
                t = (sim - lower_sim) / (upper_sim - lower_sim)
                r = int(lower_color[0] + t * (upper_color[0] - lower_color[0]))
                g = int(lower_color[1] + t * (upper_color[1] - lower_color[1]))
                b = int(lower_color[2] + t * (upper_color[2] - lower_color[2]))
                return (r, g, b)
        
        return stops[-1][1]
    
    def build(self):
        """Build results container."""
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False,
            show=False,
        ):
            pass
    
    def _clear_results_container(self):
        """Remove all children."""
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
        """Populate results table."""
        self.results = results
        self._clear_results_container()
        
        if not dpg.does_item_exist(self.container_tag):
            return
        
        dpg.configure_item(self.container_tag, show=True)
        
        if not results:
            dpg.add_text(
                "No similar songs found.",
                color=(200, 200, 200),
                parent=self.container_tag
            )
            return
        
        # Results count
        dpg.add_text(
            f"Found {len(results)} similar songs:",
            color=(200, 200, 200),
            parent=self.container_tag
        )
        dpg.add_separator(parent=self.container_tag)
        
        # Header row
        with dpg.group(horizontal=True, parent=self.container_tag):
            # Rank header
            dpg.add_text("Rank", color=Colors.TEXT_PRIMARY)
            if self.header_font:
                dpg.bind_item_font(dpg.last_item(), self.header_font)
            dpg.add_spacer(width=self._s(self._w_rank - 40))
            
            # Score header
            dpg.add_spacer(width=self._s(self._col_spacing))
            dpg.add_text("Score", color=Colors.TEXT_PRIMARY)
            if self.header_font:
                dpg.bind_item_font(dpg.last_item(), self.header_font)
            dpg.add_spacer(width=self._s(self._w_score - 50))
            
            # Song header
            dpg.add_spacer(width=self._s(self._col_spacing))
            dpg.add_text("Song", color=Colors.TEXT_PRIMARY)
            if self.header_font:
                dpg.bind_item_font(dpg.last_item(), self.header_font)
            dpg.add_spacer(width=self._s(self._w_song - 40))
            
            # Arrow spacer + Expand button
            dpg.add_spacer(width=self._s(self._col_spacing))
            dpg.add_spacer(width=self._s(self._w_arrow))
            dpg.add_spacer(width=self._s(10))
            
            add_gradient_button(
                label="Collapse All" if self._all_expanded else "Expand All",
                tag=self._toggle_button_tag,
                width=self._s(100),
                height=self._s(24),
                callback=self._toggle_all,
            )
        
        dpg.add_separator(parent=self.container_tag)
        
        # Data rows
        for i, result in enumerate(results, 1):
            if len(result) == 3:
                track_id, similarity, metadata = result
                track_name = metadata.get("track_name", "Unknown")
                artist_name = metadata.get("artist_name", "Unknown")
                album = metadata.get("album_name", "")
                year = metadata.get("album_release_year", "")
            else:
                track_id, similarity = result
                track_name, artist_name, album, year = "Unknown", "Unknown", "", ""
            
            # Build song text
            song_text = f"{track_name} - {artist_name}"
            if album:
                song_text += f" - {album}"
            if year:
                song_text += f" ({year})"
            
            # Truncate if needed
            max_chars = self._w_song // 7
            if len(song_text) > max_chars:
                song_text = song_text[:max_chars-1] + "..."
            
            song_color = self._get_song_color(similarity)
            detail_tag = dpg.generate_uuid()
            arrow_tag = dpg.generate_uuid()
            
            # Main row
            with dpg.group(horizontal=True, parent=self.container_tag):
                # Rank
                dpg.add_text(f"{i:>2d}", color=Colors.TEXT_PRIMARY)
                dpg.add_spacer(width=self._s(self._w_rank - 25))
                
                # Score
                dpg.add_spacer(width=self._s(self._col_spacing))
                dpg.add_text(f"{similarity:.4f}", color=Colors.TEXT_PRIMARY)
                dpg.add_spacer(width=self._s(self._w_score - 60))
                
                # Song (colored, clickable)
                dpg.add_spacer(width=self._s(self._col_spacing))
                song_btn = dpg.add_button(
                    label=song_text,
                    width=self._s(self._w_song),
                    height=self._s(24),
                    callback=lambda s, a, u: self._toggle_row(u[0], u[1]),
                    user_data=(detail_tag, arrow_tag)
                )
                
                with dpg.theme() as song_theme:
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                        dpg.add_theme_color(dpg.mvThemeCol_Text, song_color)
                        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                        dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.0, 0.5)
                dpg.bind_item_theme(song_btn, song_theme)
                
                # Arrow
                dpg.add_spacer(width=self._s(self._col_spacing))
                arrow_symbol = "v" if self._all_expanded else ">"
                arrow_btn = dpg.add_button(
                    label=arrow_symbol,
                    tag=arrow_tag,
                    width=self._s(self._w_arrow),
                    height=self._s(24),
                    callback=lambda s, a, u: self._toggle_row(u[0], u[1]),
                    user_data=(detail_tag, arrow_tag)
                )
                
                with dpg.theme() as arrow_theme:
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                        dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 180, 180))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.bind_item_theme(arrow_btn, arrow_theme)
                self._arrow_tags.append(arrow_tag)
            
            # Detail section - CRITICAL: Use add_group with explicit parent, not context manager with tag
            detail_group = dpg.add_group(
                horizontal=True,
                show=self._all_expanded,
                parent=self.container_tag
            )
            detail_tag = detail_group  # Use the returned tag
            
            # Indent using spacer
            spacer_width = self._s(self._w_rank + self._col_spacing + self._w_score + self._col_spacing)
            dpg.add_spacer(width=spacer_width, parent=detail_tag)
            
            # Content
            with dpg.group(parent=detail_tag):
                dpg.add_text(f"Spotify URL: https://open.spotify.com/track/{track_id}", color=(150, 150, 150))
                if album:
                    dpg.add_text(f"Album: {album}", color=(150, 150, 150))
                if year:
                    dpg.add_text(f"Released: {year}", color=(150, 150, 150))
                dpg.add_text("—" * 40, color=(80, 80, 80))
            
            self._detail_groups.append(detail_tag)
    
    def _toggle_row(self, detail_tag: int, arrow_tag: int):
        """Toggle individual row expansion."""
        if not dpg.does_item_exist(detail_tag):
            return
        
        current = dpg.get_item_configuration(detail_tag).get("show", False)
        new_state = not current
        dpg.configure_item(detail_tag, show=new_state)
        
        if dpg.does_item_exist(arrow_tag):
            dpg.configure_item(arrow_tag, label="v" if new_state else ">")
    
    def _toggle_all(self):
        """Toggle all rows."""
        self._all_expanded = not self._all_expanded
        
        if dpg.does_item_exist(self._toggle_button_tag):
            dpg.configure_item(
                self._toggle_button_tag,
                label="Collapse All" if self._all_expanded else "Expand All"
            )
        
        for arrow_tag in self._arrow_tags:
            if dpg.does_item_exist(arrow_tag):
                dpg.configure_item(arrow_tag, label="v" if self._all_expanded else ">")
        
        for detail_tag in self._detail_groups:
            if dpg.does_item_exist(detail_tag):
                dpg.configure_item(detail_tag, show=self._all_expanded)
    
    def clear_results(self):
        """Clear all results."""
        self.results = []
        self._all_expanded = False
        self._detail_groups.clear()
        self._arrow_tags.clear()
        self._clear_results_container()
        
        if dpg.does_item_exist(self._toggle_button_tag):
            dpg.configure_item(self._toggle_button_tag, label="Expand All")
        
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=False)
    
    def show_loading(self, message: str = "Searching..."):
        """Show loading message."""
        self._clear_results_container()
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
            dpg.add_text(message, color=(150, 255, 150), parent=self.container_tag)
    
    def hide_loading(self):
        """Hide loading."""
        pass
