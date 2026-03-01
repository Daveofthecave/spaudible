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

        # Tags for margin spacers (outside the bordered frame)
        self._left_spacer_tag = None
        self._right_spacer_tag = None

        # Tag for the inner bordered frame
        self._inner_frame_tag = None

        # Tag for content group inside the frame
        self._content_group_tag = None

        # Store song column tags for dynamic resizing
        self._song_btn_tags: List[int] = []
        self._header_song_tag: Optional[int] = None

        self.results = []
        self._detail_groups: List[int] = []
        self._arrow_tags: List[int] = []
        self._all_expanded = False

        # Column widths (logical pixels) - Rank, Score, and Arrow are fixed
        self._w_rank = 35
        self._w_score = 50
        self._w_arrow = 100  # Expand All button / row arrows
        self._w_song = 480  # Base width, will be dynamically adjusted

        # Thin separator between columns
        self._sep_width = 2

        # Fixed margins on each side (logical pixels) - creates the "floating" effect
        self._fixed_margin = 40

        # Create alignment themes
        self._right_align_theme = None
        self._left_align_theme = None
        self._center_align_theme = None
        self._header_theme = None

    def _s(self, value) -> int:
        """Scale value."""
        return int(float(value) * self.dpi_scale)

    def _init_themes(self):
        """Initialize button alignment themes."""

        # Right-aligned text (for Rank, Score) - completely transparent
        with dpg.theme() as self._right_align_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 0, 0, 0))
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 1.0, 0.5)  # Right, Center
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 4)

        # Left-aligned text (for Song)
        with dpg.theme() as self._left_align_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.0, 0.5)  # Left, Center
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 4)

        # Center-aligned text (for arrows)
        with dpg.theme() as self._center_align_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5)  # Center, Center
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 4)

        # Header theme
        with dpg.theme() as self._header_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 0, 0, 0))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 4)

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
        """Build results container with floating bordered frame and margins."""

        self._init_themes()

        # Outer container - no border, acts as the margin container
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False,  # No border on outer - margins will be visible
            show=False,
        ):
            # Create horizontal layout: [margin] [bordered frame] [margin]
            self._left_spacer_tag = dpg.generate_uuid()
            self._inner_frame_tag = dpg.generate_uuid()
            self._right_spacer_tag = dpg.generate_uuid()

            with dpg.group(horizontal=True):
                # Left margin spacer
                dpg.add_spacer(width=0, tag=self._left_spacer_tag)

                # Inner bordered frame containing the results table
                # autosize_x=False to prevent it from pushing the right spacer off-screen
                with dpg.child_window(
                    tag=self._inner_frame_tag,
                    width=self._s(400),  # Initial width, will be adjusted dynamically
                    autosize_x=False,    # Explicitly sized to respect margins
                    autosize_y=True,
                    border=True,         # This is the visible border around results
                    horizontal_scrollbar=False,
                ):
                    # Actual content container inside the bordered frame
                    self._content_group_tag = dpg.generate_uuid()
                    with dpg.group(tag=self._content_group_tag):
                        pass

                # Right margin spacer
                dpg.add_spacer(width=0, tag=self._right_spacer_tag)

    def update_centering(self):
        """
        Update margins and column widths.
        The bordered frame floats with fixed margins on both sides.
        """
        if not dpg.does_item_exist(self.container_tag):
            return

        # Get outer container width (the actual available space)
        container_width = dpg.get_item_rect_size(self.container_tag)[0]
        if container_width == 0:
            return

        margin = self._s(self._fixed_margin)

        # Calculate inner frame width to leave room for both margins
        inner_width = max(self._s(400), container_width - (2 * margin))

        # Set inner frame width explicitly to create the floating effect
        if dpg.does_item_exist(self._inner_frame_tag):
            dpg.configure_item(self._inner_frame_tag, width=inner_width)

        # Set both margin spacer widths
        if dpg.does_item_exist(self._left_spacer_tag):
            dpg.configure_item(self._left_spacer_tag, width=margin)
        if dpg.does_item_exist(self._right_spacer_tag):
            dpg.configure_item(self._right_spacer_tag, width=margin)

        # Calculate column widths based on inner_width
        col_rank_w = self._s(self._w_rank)
        col_score_w = self._s(self._w_score)
        col_arrow_w = self._s(self._w_arrow)
        sep_w = self._s(self._sep_width)

        # Calculate available width for Song column
        # Layout: [rank][sep][score][sep][song][sep][arrow]
        fixed_content_width = (
            col_rank_w + sep_w + col_score_w + sep_w + sep_w + col_arrow_w
        )
        new_song_width = max(self._s(200), inner_width - fixed_content_width)

        # Update header song button width
        if self._header_song_tag and dpg.does_item_exist(self._header_song_tag):
            dpg.configure_item(self._header_song_tag, width=new_song_width)

        # Update all data row song buttons
        for tag in self._song_btn_tags:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=new_song_width)
    
    def _clear_results_container(self):
        """Remove all children from content group and clear stored tags."""

        if dpg.does_item_exist(self._content_group_tag):
            children = dpg.get_item_children(self._content_group_tag)
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
        self._song_btn_tags.clear()
        self._header_song_tag = None

    def update_results(self, results: List[Tuple]):
        """Populate results table with aligned columns and dynamic song width."""

        self.results = results
        self._clear_results_container()

        if not dpg.does_item_exist(self._content_group_tag):
            return

        dpg.configure_item(self.container_tag, show=True)

        if not results:
            dpg.add_text(
                "No similar songs found.",
                color=(200, 200, 200),
                parent=self._content_group_tag
            )
            self.update_centering()
            return

        # Calculate column widths based on current inner frame size
        inner_width = 0
        if dpg.does_item_exist(self._inner_frame_tag):
            inner_width = dpg.get_item_rect_size(self._inner_frame_tag)[0]

        col_rank_w = self._s(self._w_rank)
        col_score_w = self._s(self._w_score)
        col_arrow_w = self._s(self._w_arrow)
        sep_w = self._s(self._sep_width)

        if inner_width > 0:
            # Calculate available width for Song column inside the bordered frame
            fixed_total = (
                col_rank_w + sep_w + col_score_w + sep_w + sep_w + col_arrow_w
            )
            col_song_w = max(self._s(200), inner_width - fixed_total)
        else:
            col_song_w = self._s(self._w_song)

        parent = self._content_group_tag

        # Title
        dpg.add_text(
            f"Found {len(results)} similar songs:",
            color=(200, 200, 200),
            parent=parent
        )
        dpg.add_separator(parent=parent)
        dpg.add_spacer(height=self._s(5), parent=parent)

        # Header row with proper alignment
        with dpg.group(horizontal=True, parent=parent):
            # Rank header (right-aligned)
            rank_hdr = dpg.add_button(
                label="Rank",
                width=col_rank_w,
                height=self._s(22)
            )
            dpg.bind_item_theme(rank_hdr, self._right_align_theme)
            if self.header_font:
                dpg.bind_item_font(rank_hdr, self.header_font)

            dpg.add_spacer(width=sep_w)

            # Score header (right-aligned)
            score_hdr = dpg.add_button(
                label="Score",
                width=col_score_w,
                height=self._s(22)
            )
            dpg.bind_item_theme(score_hdr, self._center_align_theme)
            if self.header_font:
                dpg.bind_item_font(score_hdr, self.header_font)

            dpg.add_spacer(width=sep_w)

            # Song header (left-aligned)
            song_hdr = dpg.add_button(
                label="Song",
                width=col_song_w,
                height=self._s(22)
            )
            dpg.bind_item_theme(song_hdr, self._left_align_theme)
            if self.header_font:
                dpg.bind_item_font(song_hdr, self.header_font)
            self._header_song_tag = song_hdr

            dpg.add_spacer(width=sep_w)

            # Expand All / Collapse All button
            btn_label = "Collapse All ▲" if self._all_expanded else "Expand All ▼"
            add_gradient_button(
                label=btn_label,
                tag=self._toggle_button_tag,
                width=col_arrow_w,
                height=self._s(24),
                callback=self._toggle_all,
            )

        dpg.add_separator(parent=parent)

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

            # Get color based on similarity
            song_color = self._get_song_color(similarity)

            detail_tag = dpg.generate_uuid()
            arrow_tag = dpg.generate_uuid()

            # Create colored theme for this specific song
            with dpg.theme() as song_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 70, 60, 100))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 60, 50, 150))
                    dpg.add_theme_color(dpg.mvThemeCol_Text, song_color)
                    dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.0, 0.5)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 4)

            # Main row with aligned columns
            with dpg.group(horizontal=True, parent=parent):
                # Rank (right-aligned)
                rank_btn = dpg.add_button(
                    label=f"{i}",
                    width=col_rank_w,
                    height=self._s(22)
                )
                dpg.bind_item_theme(rank_btn, self._right_align_theme)

                dpg.add_spacer(width=sep_w)

                # Score (right-aligned)
                score_btn = dpg.add_button(
                    label=f"{similarity:.4f}",
                    width=col_score_w,
                    height=self._s(22)
                )
                dpg.bind_item_theme(score_btn, self._right_align_theme)

                dpg.add_spacer(width=sep_w)

                # Song (left-aligned, colored, clickable)
                song_btn = dpg.add_button(
                    label=song_text,
                    width=col_song_w,
                    height=self._s(22),
                    callback=lambda s, a, u: self._toggle_row(u[0], u[1]),
                    user_data=(detail_tag, arrow_tag)
                )
                dpg.bind_item_theme(song_btn, song_theme)
                self._song_btn_tags.append(song_btn)

                dpg.add_spacer(width=sep_w)

                # Arrow button (centered)
                arrow_symbol = "v" if self._all_expanded else "<"
                arrow_btn = dpg.add_button(
                    label=arrow_symbol,
                    tag=arrow_tag,
                    width=col_arrow_w,
                    height=self._s(22),
                    callback=lambda s, a, u: self._toggle_row(u[0], u[1]),
                    user_data=(detail_tag, arrow_tag)
                )
                dpg.bind_item_theme(arrow_btn, self._center_align_theme)
                self._arrow_tags.append(arrow_tag)

            # Detail section (expandable)
            detail_group = dpg.add_group(
                tag=detail_tag,
                horizontal=True,
                show=self._all_expanded,
                parent=parent
            )

            # Indent to align with Song column
            indent_width = col_rank_w + sep_w + col_score_w + sep_w
            dpg.add_spacer(width=indent_width, parent=detail_group)

            # Metadata content
            with dpg.group(parent=detail_group):
                dpg.add_text(
                    f"Spotify URL: https://open.spotify.com/track/{track_id}",
                    color=(150, 150, 150)
                )
                if album:
                    dpg.add_text(f"Album: {album}", color=(150, 150, 150))
                if year:
                    dpg.add_text(f"Released: {year}", color=(150, 150, 150))
                dpg.add_text("—" * 40, color=(80, 80, 80))

            self._detail_groups.append(detail_tag)

        # Apply margins and column widths
        self.update_centering()

    def _toggle_row(self, detail_tag: int, arrow_tag: int):
        """Toggle individual row expansion."""

        if not dpg.does_item_exist(detail_tag):
            return

        current = dpg.get_item_configuration(detail_tag).get("show", False)
        new_state = not current
        dpg.configure_item(detail_tag, show=new_state)

        if dpg.does_item_exist(arrow_tag):
            dpg.configure_item(arrow_tag, label="v" if new_state else "<")

    def _toggle_all(self):
        """Toggle all rows and update button text."""

        self._all_expanded = not self._all_expanded

        # Update button text
        if dpg.does_item_exist(self._toggle_button_tag):
            new_label = "Collapse All ▲" if self._all_expanded else "Expand All ▼"
            dpg.configure_item(self._toggle_button_tag, label=new_label)

        # Update all arrows
        for arrow_tag in self._arrow_tags:
            if dpg.does_item_exist(arrow_tag):
                dpg.configure_item(
                    arrow_tag, label="v" if self._all_expanded else "<"
                )

        # Show/hide all detail groups
        for detail_tag in self._detail_groups:
            if dpg.does_item_exist(detail_tag):
                dpg.configure_item(detail_tag, show=self._all_expanded)

    def clear_results(self):
        """Clear all results and reset expansion state."""

        self.results = []
        self._all_expanded = False
        self._detail_groups.clear()
        self._arrow_tags.clear()
        self._song_btn_tags.clear()
        self._header_song_tag = None
        self._clear_results_container()

        # Reset button text
        if dpg.does_item_exist(self._toggle_button_tag):
            dpg.configure_item(self._toggle_button_tag, label="Expand All ▼")

        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=False)

    def show_loading(self, message: str = "Searching..."):
        """Show loading message inside the bordered frame."""

        self._clear_results_container()

        if dpg.does_item_exist(self._content_group_tag):
            dpg.configure_item(self.container_tag, show=True)
            dpg.add_text(
                message,
                color=(150, 255, 150),
                parent=self._content_group_tag
            )
            # Ensure margins are set even for loading state
            self.update_centering()

    def hide_loading(self):
        """Hide loading."""
        pass
