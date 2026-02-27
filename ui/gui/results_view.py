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
        # Column widths
        self._w_rank = 80
        self._w_score = 120

    def _s(self, value) -> int:
        """Scale value - ensure native Python int."""
        return int(float(value) * self.dpi_scale)

    def _get_song_color(self, similarity: float) -> Tuple[int, int, int]:
        """
        Map similarity score to color gradient.
        1.000 = Cyan (#00fcff) -> Green -> Yellow -> Orange -> Red -> Purple = 0.000
        """
        # Color stops: (similarity_threshold, (r, g, b))
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
        self.results = results
        self._clear_results_container()
        
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
        
        if not results:
            dpg.add_text("No similar songs found.", parent=self.container_tag)
            return
        
        # Header row with Expand/Collapse button
        with dpg.group(horizontal=True, parent=self.container_tag):
            # Rank column header (right-aligned)
            with dpg.group(width=self._s(self._w_rank)):
                dpg.add_spacer(width=-1)
                dpg.add_text("Rank")
            
            # Score column header (right-aligned)
            with dpg.group(width=self._s(self._w_score)):
                dpg.add_spacer(width=-1)
                dpg.add_text("Score")
            
            # Song column header (left-aligned, flexible)
            dpg.add_text("Song")
            
            # Push button to right
            dpg.add_spacer(width=-1)
            
            # Toggle button
            add_gradient_button(
                label="Collapse All" if self._all_expanded else "Expand All",
                tag=self._toggle_button_tag,
                width=self._s(100),
                height=self._s(24),
                callback=self._toggle_all,
            )
        
        dpg.add_separator(parent=self.container_tag)
        
        # Found count text
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
                track_name, artist_name, album, year = "Unknown", "Unknown", "", ""
            
            # Get color for song text based on similarity
            song_color = self._get_song_color(similarity)
            
            # Row container (vertical layout)
            row_container = dpg.add_group(
                horizontal=False,
                parent=self.container_tag,
            )
            
            detail_tag = dpg.generate_uuid()
            arrow_tag = dpg.generate_uuid()
            
            # Main row (selectable for clickability) - add_selectable returns a tag
            selectable_tag = dpg.add_selectable(
                parent=row_container,
                callback=self._toggle_row,
                user_data=(detail_tag, arrow_tag),
                width=self._s(2000),  # Wide enough to fill container
                height=self._s(24),
                label=""
            )
            
            # Content inside the selectable
            with dpg.group(horizontal=True, parent=selectable_tag):
                # Rank (right-aligned)
                with dpg.group(width=self._s(self._w_rank)):
                    dpg.add_spacer(width=-1)
                    dpg.add_text(str(i))
                
                # Score (right-aligned)
                with dpg.group(width=self._s(self._w_score)):
                    dpg.add_spacer(width=-1)
                    dpg.add_text(f"{similarity:.4f}")
                
                # Song (left-aligned, colored)
                song_text = f"{track_name} - {artist_name}"
                if album:
                    song_text += f" - {album}"
                if year:
                    song_text += f" ({year})"
                if len(song_text) > 50:
                    song_text = song_text[:47] + "..."
                dpg.add_text(song_text, color=song_color)
                
                # Push arrow to right
                dpg.add_spacer(width=-1)
                
                # Arrow indicator (text, not button)
                arrow_symbol = "▼" if self._all_expanded else "▶"
                dpg.add_text(arrow_symbol, tag=arrow_tag)
            
            # Detail row (expandable, outside selectable so clicks don't collapse)
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
            self._arrow_tags.append(arrow_tag)

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
        """Toggle all rows expansion state."""
        self._all_expanded = not self._all_expanded
        
        if dpg.does_item_exist(self._toggle_button_tag):
            label = "Collapse All" if self._all_expanded else "Expand All"
            dpg.configure_item(self._toggle_button_tag, label=label)
        
        for detail_tag, arrow_tag in zip(self._detail_groups, self._arrow_tags):
            if dpg.does_item_exist(detail_tag):
                dpg.configure_item(detail_tag, show=self._all_expanded)
            if dpg.does_item_exist(arrow_tag):
                dpg.set_value(arrow_tag, "▼" if self._all_expanded else "▶")

    def clear_results(self):
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
