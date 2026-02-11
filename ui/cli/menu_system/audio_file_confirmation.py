# ui/cli/menu_system/audio_file_confirmation.py
"""
Audio File Confirmation UI
============================
Single-file confirmation dialog for audio file matching.
Displays filename alongside matched track, allowing confirm/refine/cancel.
"""
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from core.utilities.audio_file_input_processor import (
    AudioFileInput,
    ResolvedAudioFile,
    AudioFileInputResolver,
)
from core.utilities.text_search_utils import search_tracks_flexible
from ui.cli.console_utils import print_header

logger = logging.getLogger(__name__)

def _search_adapter(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Adapt search_tracks_flexible to return dicts for AudioFileInputResolver.
    """
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
            'confidence': r.confidence,
        }
        for r in results
    ]

class AudioFileConfirmationDialog:
    """
    Prompt-toolkit based confirmation dialog for audio file matching.
    
    Layout:
    [Filename: ...]
    [Matched: ...] (or "Searching..." / "No match found")
    
    [Confirm (Enter)] [Refine (Tab)] [Cancel (Esc)]
    """
    
    def __init__(self):
        self.resolved: Optional[ResolvedAudioFile] = None
        self.result: Optional[str] = None  # track_id or None
        self.should_refine = False
    
    def show(self, file_path: Path, resolver: Optional[AudioFileInputResolver] = None) -> Optional[str]:
        """
        Show confirmation dialog for an audio file.
        
        Args:
            file_path: Path to the audio file
            resolver: Optional resolver instance (creates new if None)
            
        Returns:
            track_id if confirmed, None if cancelled
        """
        print_header("Audio File Match")
        
        # Resolve the file (this may take a moment)
        print(f"\n Analyzing: {file_path.name}")
        print(" Searching database for matching track...")
        
        if resolver is None:
            resolver = AudioFileInputResolver(search_func=_search_adapter)
        
        self.resolved = resolver.resolve(file_path)
        
        # Build UI
        self._build_ui()
        
        # Run app
        app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=False,
            mouse_support=False,
        )
        
        self.result = app.run()
        
        if self.should_refine:
            # Return special signal to caller to open refinement
            return "__REFINE__"
        
        return self.result
    
    def _build_ui(self):
        """Build the prompt_toolkit UI layout."""
        
        # Status line at top
        filename_line = f"File: {self.resolved.audio_file_input.filename}"
        
        # Match line
        if not self.resolved.is_resolved:
            match_line = "❌ No match found in database"
            match_style = "class:no-match"
        else:
            match_text = self.resolved.get_display_text()
            match_line = f"Match: {match_text}"
            match_style = "class:matched"
        
        # Create windows
        self.filename_window = Window(
            height=1,
            content=FormattedTextControl(text=filename_line),
            style="class:filename"
        )
        self.match_window = Window(
            height=1,
            content=FormattedTextControl(text=match_line),
            style=match_style
        )
        
        # Help text
        if self.resolved.is_resolved:
            help_text = "[Enter] Confirm  [Tab] Refine match  [Esc] Cancel"
        else:
            help_text = "[Tab] Try other searches  [Esc] Cancel"
        
        self.help_window = Window(
            height=1,
            content=FormattedTextControl(text=help_text),
            style="class:help"
        )
        
        # Layout
        container = HSplit([
            self.filename_window,
            self.match_window,
            Window(height=1),  # Spacer
            self.help_window,
        ])
        self.layout = Layout(container)
        
        # Key bindings
        self.kb = KeyBindings()
        
        @self.kb.add('enter')
        def confirm(event):
            """Confirm the match and return track_id."""
            if self.resolved.is_resolved:
                self.resolved.is_confirmed = True
                event.app.exit(result=self.resolved.track_id)
            else:
                # Enter with no match does nothing (or could trigger refine)
                pass
        
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
            'filename': 'bold ansiblue',
            'matched': 'bold ansigreen',
            'no-match': 'bold ansired',
            'help': 'ansiwhite',
        })

def confirm_audio_file(file_path: Path) -> Optional[str]:
    """
    Convenience function to show confirmation dialog.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        track_id if confirmed, None if cancelled or failed,
        "__REFINE__" if user wants to refine (caller should open refinement UI)
    """
    dialog = AudioFileConfirmationDialog()
    return dialog.show(file_path)

def confirm_audio_file_with_fallback(file_path: Path) -> Optional[str]:
    """
    Show confirmation, and if user wants to refine, open refinement dialog.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Final track_id selected by user, or None
    """
    from .audio_file_refinement import refine_audio_file_match
    
    result = confirm_audio_file(file_path)
    
    if result == "__REFINE__":
        # Open refinement dialog
        resolver = AudioFileInputResolver(search_func=_search_adapter)
        resolved = resolver.resolve(file_path)
        return refine_audio_file_match(resolved)
    
    return result
