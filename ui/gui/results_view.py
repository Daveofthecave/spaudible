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
        self.action_bar_tag = "results_action_bar"
        self.results = []
        self._detail_groups: List[int] = []
        self._arrow_tags: List[int] = []
        self._all_expanded = False
        self._toggle_button_tag = "results_toggle_all_btn"
        # Column widths
        self._w_rank = 80
        self._w_score = 120

    def _s(self, value) -> int:
        return int(float(value) * self.dpi_scale)

    def _get_similarity_color(self, similarity: float) -> Tuple[int, int, int]:
        """Purple (low) -> Cyan (high)."""
        s = max(0.0, min(1.0, similarity))
        hue = 270 - (s * 90)
        c = (1 - abs(2 * 0.5 - 1)) * 1.0
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = 0.5 - c / 2
        if hue < 180:
            r, g, b = 0, c, x
        elif hue < 240:
            r, g, b = 0, x, c
        else:
            r, g, b = x, 0, c
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    def build(self):
        """Build results table."""
        # Header
        with dpg.group(horizontal=True):
            # Rank column
            with dpg.group(width=self._s(self._w_rank)):
                dpg.add_text("Rank")
            # Score column
            with dpg.group(width=self._s(self._w_score)):
                dpg.add_text("Score")
            # Song column (flexible)
            dpg.add_text("Song")
            # Push button to right
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

        if dpg.does_item_exist(self._toggle_button_tag):
            label = "Collapse All" if self._all_expanded else "Expand All"
            dpg.configure_item(self._toggle_button_tag, label=label)

        if not results:
            dpg.add_text("No similar songs found.", parent=self.container_tag)
            return

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

            # Row container (vertical)
            row_container = dpg.add_group(
                horizontal=False,
                parent=self.container_tag,
            )
            detail_tag = dpg.generate_uuid()
            arrow_tag = dpg.generate_uuid()

            # Main row content (horizontal)
            with dpg.group(horizontal=True, parent=row_container):
                # Rank - fixed width
                with dpg.group(width=self._s(self._w_rank)):
                    dpg.add_text(str(i))

                # Score - fixed width with color
                score_color = self._get_similarity_color(similarity)
                with dpg.group(width=self._s(self._w_score)):
                    dpg.add_text(f"{similarity:.4f}", color=score_color)

                # Song - flexible
                song_text = f"{track_name} - {artist_name}"
                if album:
                    song_text += f" - {album}"
                if year:
                    song_text += f" ({year})"
                if len(song_text) > 50:
                    song_text = song_text[:47] + "..."
                dpg.add_text(song_text)

                # Push arrow to right
                dpg.add_spacer(width=-1)

                # Expand arrow button (click target)
                arrow_btn = dpg.add_button(
                    label="▼" if self._all_expanded else "▶",
                    tag=arrow_tag,
                    width=self._s(30),
                    height=self._s(24),
                    callback=self._toggle_row,
                    user_data=(detail_tag, arrow_tag),
                )
                self._arrow_tags.append(arrow_btn)

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
        detail_tag, arrow_tag = user_data
        if dpg.does_item_exist(detail_tag):
            current = dpg.get_item_configuration(detail_tag).get("show", False)
            new_state = not current
            dpg.configure_item(detail_tag, show=new_state)
            if dpg.does_item_exist(arrow_tag):
                dpg.configure_item(arrow_tag, label="▼" if new_state else "▶")

    def _toggle_all(self):
        self._all_expanded = not self._all_expanded
        if dpg.does_item_exist(self._toggle_button_tag):
            label = "Collapse All" if self._all_expanded else "Expand All"
            dpg.configure_item(self._toggle_button_tag, label=label)
        for detail_tag, arrow_tag in zip(self._detail_groups, self._arrow_tags):
            if dpg.does_item_exist(detail_tag):
                dpg.configure_item(detail_tag, show=self._all_expanded)
            if dpg.does_item_exist(arrow_tag):
                dpg.configure_item(arrow_tag, label="▼" if self._all_expanded else "▶")

    def clear_results(self):
        self.results = []
        self._all_expanded = False
        self._detail_groups.clear()
        self._arrow_tags.clear()
        if dpg.does_item_exist(self.action_bar_tag):
            dpg.configure_item(self.action_bar_tag, show=False)
        self._clear_results_container()
        if dpg.does_item_exist(self._toggle_button_tag):
            dpg.configure_item(self._toggle_button_tag, label="Expand All")
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=False)

    def show_loading(self, message: str = "Searching..."):
        if dpg.does_item_exist(self.action_bar_tag):
            dpg.configure_item(self.action_bar_tag, show=False)
        self._clear_results_container()
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=True)
        dpg.add_text(message, color=(150, 255, 150), parent=self.container_tag)

    def hide_loading(self):
        pass
