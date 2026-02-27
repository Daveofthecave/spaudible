# ui/gui/results_view.py
import dearpygui.dearpygui as dpg
from typing import List, Tuple, Optional
from ui.gui.theme import add_gradient_button

class ResultsView:
    """Results display with clean table layout and expandable rows."""
    
    def __init__(self, dpi_scale: float = 1.0):
        self.dpi_scale = float(dpi_scale)
        self.tag = "results_panel"
        self.container_tag = "results_container"
        self._toggle_button_tag = "results_toggle_all_btn"
        self.results = []
        self._detail_groups: List[int] = []
        self._arrow_tags: List[int] = []
        self._all_expanded = False
        
        # Column widths (total row width ~650px scaled)
        self._w_rank = 60
        self._w_score = 80
        self._w_song = 480  # Remaining space for song
        self._w_arrow = 30
        
        # Predefined themes for consistent styling
        self._rank_theme = None
        self._score_theme = None
        self._arrow_theme = None
        self._header_text_theme = None

    def _s(self, value) -> int:
        """Scale value - ensure native Python int."""
        return int(float(value) * self.dpi_scale)

    def _get_song_color(self, similarity: float) -> Tuple[int, int, int]:
        """ Map similarity score to color gradient. 1.000 = Cyan (#00fcff), 0.000 = Purple """
        stops = [
            (1.0, (0, 252, 255)),    # Cyan
            (0.8, (0, 255, 0)),      # Green
            (0.6, (255, 255, 0)),    # Yellow
            (0.4, (255, 165, 0)),    # Orange
            (0.2, (255, 0, 0)),      # Red
            (0.0, (128, 0, 255)),    # Purple
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
        """Build results container (initially empty/hidden)."""
        # Container for results (initially hidden)
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False,
            show=False,
        ):
            pass
        
        # Initialize reusable themes for row elements
        # Rank: White text, right-aligned, transparent
        with dpg.theme() as self._rank_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 1.0, 0.5)  # Right align
                
        # Score: White text, right-aligned, transparent
        with dpg.theme() as self._score_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 1.0, 0.5)  # Right align
                
        # Arrow: Gray text, center-aligned, transparent
        with dpg.theme() as self._arrow_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (200, 200, 200))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5)  # Center
                
        # Header text: White, left-aligned for Song, transparent
        with dpg.theme() as self._header_text_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (255, 255, 255))  # For disabled state
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.0, 0.5)  # Left align

    def _clear_results_container(self):
        """Remove all children from the results container."""
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
        """Populate results with aligned columns and color-coded songs."""
        self.results = results
        self._clear_results_container()
        
        if not dpg.does_item_exist(self.container_tag):
            return
            
        dpg.configure_item(self.container_tag, show=True)
        
        if not results:
            dpg.add_text("No similar songs found.", color=(200, 200, 200), parent=self.container_tag)
            return

        # Found count
        dpg.add_text(f"Found {len(results)} similar songs:", color=(200, 200, 200), parent=self.container_tag)
        dpg.add_separator(parent=self.container_tag)
        
        # Header row with column labels and Expand/Collapse button
        with dpg.group(horizontal=True, parent=self.container_tag) as header_group:
            # Rank header (disabled button for alignment)
            rank_hdr = dpg.add_button(
                label="Rank", 
                width=self._s(self._w_rank), 
                height=self._s(24),
                enabled=False
            )
            # Apply theme manually since we need to bind it
            dpg.bind_item_theme(rank_hdr, self._rank_theme)
            
            # Score header
            score_hdr = dpg.add_button(
                label="Score", 
                width=self._s(self._w_score), 
                height=self._s(24),
                enabled=False
            )
            dpg.bind_item_theme(score_hdr, self._score_theme)
            
            # Song header
            song_hdr = dpg.add_button(
                label="Song", 
                width=self._s(self._w_song), 
                height=self._s(24),
                enabled=False
            )
            dpg.bind_item_theme(song_hdr, self._header_text_theme)
            
            # Spacer to push button to right
            dpg.add_spacer(width=self._s(20))
            
            # Expand/Collapse All button
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

            # Build song display text without premature truncation
            song_text = f"{track_name} - {artist_name}"
            if album:
                song_text += f" - {album}"
            if year:
                song_text += f" ({year})"
            
            # Get color for this song based on similarity
            song_color = self._get_song_color(similarity)
            
            # Row container (vertical stack of line + details)
            with dpg.group(horizontal=False, parent=self.container_tag) as row_container:
                # Generate unique tags for this row's expandable elements
                detail_tag = dpg.generate_uuid()
                arrow_tag = dpg.generate_uuid()
                
                # Horizontal line group containing the columns
                with dpg.group(horizontal=True):
                    # Rank column (right-aligned white text)
                    rank_btn = dpg.add_button(
                        label=f"{i:>2}",
                        width=self._s(self._w_rank),
                        height=self._s(24),
                        callback=self._toggle_row,
                        user_data=(detail_tag, arrow_tag)
                    )
                    dpg.bind_item_theme(rank_btn, self._rank_theme)
                    
                    # Score column (right-aligned white text)
                    score_btn = dpg.add_button(
                        label=f"{similarity:.4f}",
                        width=self._s(self._w_score),
                        height=self._s(24),
                        callback=self._toggle_row,
                        user_data=(detail_tag, arrow_tag)
                    )
                    dpg.bind_item_theme(score_btn, self._score_theme)
                    
                    # Song column (left-aligned colored text)
                    # Create per-row theme for the specific color
                    song_btn = dpg.add_button(
                        label=song_text,
                        width=self._s(self._w_song),
                        height=self._s(24),
                        callback=self._toggle_row,
                        user_data=(detail_tag, arrow_tag)
                    )
                    with dpg.theme() as song_row_theme:
                        with dpg.theme_component(dpg.mvButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                            dpg.add_theme_color(dpg.mvThemeCol_Text, song_color)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                            dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.0, 0.5)  # Left align
                    dpg.bind_item_theme(song_btn, song_row_theme)
                    
                    # Arrow indicator column (centered)
                    arrow_symbol = "▼" if self._all_expanded else "▶"
                    arrow_btn = dpg.add_button(
                        label=arrow_symbol,
                        tag=arrow_tag,
                        width=self._s(self._w_arrow),
                        height=self._s(24),
                        callback=self._toggle_row,
                        user_data=(detail_tag, arrow_tag)
                    )
                    dpg.bind_item_theme(arrow_btn, self._arrow_theme)
                
                # Store arrow tag for bulk updates
                self._arrow_tags.append(arrow_tag)
                
                # Detail section (expandable) - indented to align under Song column
                with dpg.group(
                    tag=detail_tag,
                    show=self._all_expanded,
                    indent=self._s(self._w_rank + self._w_score + 20)  # Indent past Rank+Score+spacing
                ):
                    dpg.add_text(f"Spotify URL: https://open.spotify.com/track/{track_id}", color=(150, 150, 150))
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
                # Update arrow symbol via label configuration
                dpg.configure_item(arrow_tag, label="▼" if new_state else "▶")

    def _toggle_all(self):
        """Toggle all rows expansion state."""
        self._all_expanded = not self._all_expanded
        
        if dpg.does_item_exist(self._toggle_button_tag):
            label = "Collapse All" if self._all_expanded else "Expand All"
            dpg.configure_item(self._toggle_button_tag, label=label)
        
        # Update all arrow indicators
        for arrow_tag in self._arrow_tags:
            if dpg.does_item_exist(arrow_tag):
                dpg.configure_item(arrow_tag, label="▼" if self._all_expanded else "▶")
        
        # Show/hide all detail groups
        for detail_tag in self._detail_groups:
            if dpg.does_item_exist(detail_tag):
                dpg.configure_item(detail_tag, show=self._all_expanded)

    def clear_results(self):
        """Clear all results and reset state."""
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
        """Show loading message in results area."""
        self._clear_results_container()
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
            dpg.add_text(message, color=(150, 255, 150), parent=self.container_tag)

    def hide_loading(self):
        """Hide loading indicator."""
        pass
