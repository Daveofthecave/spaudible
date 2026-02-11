# ui/cli/menu_system/audio_file_confirmation.py
""" Audio File Confirmation UI
============================
Single-file confirmation dialog with table layout.
Displays filename alongside matched track, allowing confirm/refine/cancel.
"""
import logging
from pathlib import Path
from typing import Optional

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from core.utilities.audio_file_input_processor import (
    AudioFileInputResolver,
    ResolvedAudioFile,
)
from core.utilities.text_search_utils import search_tracks_flexible
from ui.cli.console_utils import print_header, clear_screen
from config import FRAME_WIDTH

logger = logging.getLogger(__name__)

def _search_adapter(query: str, limit: int = 3) -> list:
    """Adapt search_tracks_flexible to return dicts for AudioFileInputResolver."""
    results = search_tracks_flexible(query, limit=limit)
    return [
        {
            'track_id': r.track_id,
            'track_name': r.track_name,
            'artist_name': r.artist_name,
            'album_name': r.album_name,
            'album_release_year': r.album_release_year,
            'popularity': r.popularity,
            'isrc': r.isrc,
        }
        for r in results
    ]

class AudioFileConfirmationDialog:
    """Table-based confirmation dialog for audio file matching.
    
    Layout:
    [Audio File         | Matched Song                        ]
    [─────────────────────────────────────────────────────────]
    [audio_filename.mp3 | → Track Title - Artist - Album (Yr) ]
    [                                                         ]

    [Enter] Confirm  [Tab] Refine match  [Esc] Cancel
    """

    def __init__(self):
        self.resolved: Optional[ResolvedAudioFile] = None
        self.result: Optional[str] = None
        self.should_refine = False

    def show(self, file_path: Path) -> Optional[str]:
        """Show confirmation dialog with table layout."""
        clear_screen()
        print_header("Audio File Match")
        
        # Resolve the file
        print(f"\n   Analyzing: {file_path.name}")
        print("   Searching database for matching track...")
        
        resolver = AudioFileInputResolver(search_func=_search_adapter)
        self.resolved = resolver.resolve(file_path)
        
        # Build UI
        self._build_ui()
        
        # Run
        app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=False,
            mouse_support=False,
        )
        self.result = app.run()
        
        if self.should_refine:
            return "__REFINE__"
        return self.result

    def _build_ui(self):
        """Build the table-style UI layout."""
        # Header row - aligned with content below
        header_text = " " + "Audio File".ljust(26) + " │ " + "Matched Song"
        separator_text = "─" * FRAME_WIDTH
        
        # File row - truncate filename from beginning if too long
        filename = self.resolved.audio_file_input.filename
        if len(filename) > 26:
            filename = "…" + filename[-25:]
        
        # Match row
        if self.resolved.is_resolved and self.resolved.matched_variation:
            match = self.resolved.matched_variation.results[0] if \
                self.resolved.matched_variation.results else None
            if match:
                artist = match.get('artist_name', 'Unknown')
                title = match.get('track_name', 'Unknown')
                album = match.get('album_name', '')
                year = match.get('album_release_year', '')
                
                # Build match text
                match_text = f"{title} - {artist}"
                if album:
                    match_text += f" - {album}"
                if year:
                    match_text += f" ({year})"
                
                # Truncate if too long
                if len(match_text) > 200:
                    match_text = match_text[:199] + "…"
                file_row = f" {filename:<26} │ → {match_text}"
                row_style = "class:matched"
            else:
                file_row = f" {filename:<26} │ (Error loading match)"
                row_style = "class:no-match"
        else:
            file_row = f" {filename:<26} │ ❌ No match found"
            row_style = "class:no-match"
        
        # Create windows
        self.header_window = Window(
            height=1,
            content=FormattedTextControl(text=header_text),
            style="class:header"
        )
        self.separator_window = Window(
            height=1,
            content=FormattedTextControl(text=separator_text),
            style="class:separator"
        )
        self.file_window = Window(
            height=1,
            content=FormattedTextControl(text=file_row),
            style=row_style,
            always_hide_cursor=True  # Hide the blinking cursor
        )
        
        # Help text
        if self.resolved.is_resolved:
            help_text = " [Enter] Confirm  [Tab] Refine match  [Esc] Cancel"
        else:
            help_text = " [Tab] Try other searches  [Esc] Cancel"
        
        self.help_window = Window(
            height=1,
            content=FormattedTextControl(text=help_text),
            style="class:help"
        )
        
        # Layout container
        container = HSplit([
            Window(height=1),  # Top spacer
            self.header_window,
            self.separator_window,
            self.file_window,
            Window(height=1),  # Bottom spacer
            self.help_window,
        ])
        
        # Set layout with focus on the file entry (no cursor visible)
        self.layout = Layout(container, focused_element=self.file_window)
        
        # Key bindings
        self.kb = KeyBindings()
        
        @self.kb.add('enter')
        def confirm(event):
            """Confirm the match and return track_id."""
            if self.resolved.is_resolved:
                self.resolved.is_confirmed = True
                event.app.exit(result=self.resolved.track_id)
        
        @self.kb.add('tab')
        @self.kb.add('s-tab')
        def refine(event):
            """Signal that user wants to refine the match."""
            self.should_refine = True
            event.app.exit(result=None)
        
        @self.kb.add('escape')
        @self.kb.add('c-c')
        def cancel(event):
            """Cancel the operation."""
            event.app.exit(result=None)
        
        # Style
        self.style = Style.from_dict({
            'header': 'bold ansiblue',
            'separator': 'ansiblue',
            'matched': 'ansigreen',
            'no-match': 'ansired',
            'help': 'ansiwhite',
        })

def confirm_audio_file(file_path: Path) -> Optional[str]:
    """Convenience function to show confirmation dialog.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        track_id if confirmed, None if cancelled, "__REFINE__" if user wants to refine
    """
    dialog = AudioFileConfirmationDialog()
    return dialog.show(file_path)


def confirm_audio_file_with_fallback(file_path: Path) -> Optional[str]:
    """Show confirmation, and if user wants to refine, open refinement dialog.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Final track_id selected by user, or None
    """
    from ui.cli.menu_system.audio_file_refinement import refine_audio_file_match
    
    result = confirm_audio_file(file_path)
    if result == "__REFINE__":
        # Re-resolve to get all variations
        resolver = AudioFileInputResolver(search_func=_search_adapter)
        resolved = resolver.resolve(file_path)
        return refine_audio_file_match(resolved)
    return result
