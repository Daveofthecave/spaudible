# ui/cli/menu_system/audio_file_refinement.py
"""
Audio File Refinement UI
========================
Allows user to select from alternative matches when automatic resolution fails or is incorrect.
Shows candidates from multiple variations plus a manual search option.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from core.utilities.audio_file_input_processor import ResolvedAudioFile, SearchVariation
from core.utilities.text_search_utils import search_tracks_flexible, SearchResult
from ui.cli.console_utils import print_header

logger = logging.getLogger(__name__)

@dataclass
class RefinementCandidate:
    """Single candidate for display in refinement UI."""
    track_id: str
    display_text: str
    variation_source: str  # e.g., "Title + Artist", "Filename", "Manual"
    isrc: Optional[str] = None
    popularity: int = 0

class AudioFileRefinementDialog:
    """Refinement dialog showing candidates from multiple variations."""

    def __init__(self):
        self.resolved: Optional[ResolvedAudioFile] = None
        self.candidates: List[RefinementCandidate] = []
        self.manual_results: List[RefinementCandidate] = []
        self.selected_idx: int = 0
        self.query_buffer: Optional[Buffer] = None
        self.layout: Optional[Layout] = None
        self.kb: Optional[KeyBindings] = None
        self.result: Optional[str] = None
        self.main_window: Optional[Window] = None
        self.query_window: Optional[Window] = None

    def show(self, resolved: ResolvedAudioFile) -> Optional[str]:
        """Show refinement dialog for a resolved audio file."""
        self.resolved = resolved
        self._build_candidates()
        self._build_ui()
        
        app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=False,
            mouse_support=False,
        )
        self.result = app.run()
        return self.result

    def _build_candidates(self):
        """Build list of candidates from all variations."""
        seen_track_ids: Set[str] = set()
        
        # Add candidates from ALL variations that have results
        for variation in self.resolved.audio_file_input.variations:
            if not variation.results:
                continue
            for result in variation.results:
                track_id = result.get('track_id')
                if not track_id or track_id in seen_track_ids:
                    continue
                seen_track_ids.add(track_id)
                candidate = RefinementCandidate(
                    track_id=track_id,
                    display_text=self._format_result(result),
                    variation_source=variation.description,
                    isrc=result.get('isrc'),
                    popularity=result.get('popularity', 0)
                )
                self.candidates.append(candidate)
        
        # Sort by popularity descending
        self.candidates.sort(key=lambda x: x.popularity, reverse=True)

    def _format_result(self, result: Dict) -> str:
        """Format a search result for display."""
        artist = result.get('artist_name', 'Unknown')
        title = result.get('track_name', 'Unknown')
        album = result.get('album_name', '')
        year = result.get('album_release_year', '')
        parts = [f"{title} - {artist}"]
        if album:
            parts.append(f" - {album}")
        if year:
            parts.append(f" ({year})")
        return "".join(parts)

    def _build_ui(self):
        """Build the prompt_toolkit UI."""
        # Query buffer for manual search
        self.query_buffer = Buffer(multiline=False)
        self.query_buffer.text = ""

        # Create query window (at top)
        self.query_window = Window(
            height=1,
            content=BufferControl(buffer=self.query_buffer),
            style="class:query"
        )

        # Create main display window
        self.main_window = Window(
            height=20,
            content=FormattedTextControl(text=""),
            style="class:candidate"
        )

        # Key bindings
        self.kb = KeyBindings()

        @self.kb.add('up')
        def move_up(event):
            """Move selection up."""
            if self.selected_idx > 0:
                self.selected_idx -= 1
                self._update_display()

        @self.kb.add('down')
        def move_down(event):
            """Move selection down."""
            max_idx = len(self.candidates) + len(self.manual_results) - 1
            if self.selected_idx < max_idx:
                self.selected_idx += 1
                self._update_display()

        @self.kb.add('enter')
        def select(event):
            """Select current candidate OR perform search if query entered."""
            # If user typed something in the search box, perform search instead of selecting
            if self.query_buffer.text.strip():
                self._do_manual_search()
                self._update_display()
                return  # Stay in dialog to let user see results
            
            # Otherwise, select the highlighted candidate
            all_candidates = self.candidates + self.manual_results
            if 0 <= self.selected_idx < len(all_candidates):
                self.result = all_candidates[self.selected_idx].track_id
                event.app.exit(result=self.result)

        @self.kb.add('c-r')
        def search_manual(event):
            """Trigger manual search."""
            if self.query_buffer.text.strip():
                self._do_manual_search()
                self._update_display()

        @self.kb.add('escape')
        @self.kb.add('c-c')
        def cancel(event):
            """Cancel and return None."""
            event.app.exit(result=None)

        # Build layout: Query label, query input, then main content
        container = HSplit([
            Window(height=1, content=FormattedTextControl(text="Refine query (type to search):"), style="class:help"),
            self.query_window,
            Window(height=1),  # Spacer
            self.main_window,
        ])
        
        self.layout = Layout(container, focused_element=self.query_window)

        # Build initial display
        self._update_display()

        # Style
        self.style = Style.from_dict({
            'query': 'bold ansigreen',
            'candidate': '',
            'selected': 'reverse ansiwhite',
            'variation': 'ansicyan',
            'help': 'ansiwhite',
        })

    def _update_display(self):
        """Update the main display content (everything except the query input)."""
        lines = []
        
        # Header
        lines.append(f"Refining match for: {self.resolved.audio_file_input.filename}")
        lines.append("")
        
        # Candidates from variations
        if self.candidates:
            lines.append("─" * 60)
            lines.append("Suggested matches:")
            for i, candidate in enumerate(self.candidates):
                prefix = "→ " if i == self.selected_idx else "  "
                lines.append(f"{prefix}[{candidate.variation_source}] {candidate.display_text}")
            lines.append("")
        
        # Manual results
        if self.manual_results:
            lines.append("─" * 60)
            lines.append("Manual search results:")
            offset = len(self.candidates)
            for i, candidate in enumerate(self.manual_results):
                idx = offset + i
                prefix = "→ " if idx == self.selected_idx else "  "
                lines.append(f"{prefix}{candidate.display_text}")
            lines.append("")
        
        # Help
        lines.append("─" * 60)
        lines.append("↑↓ Navigate | Enter: Select | Ctrl+R: Search | Esc: Cancel")
        
        self.main_window.content.text = "\n".join(lines)

    def _do_manual_search(self):
        """Perform manual text search."""
        query = self.query_buffer.text.strip()
        if not query:
            return
        
        try:
            # Clear previous manual results
            self.manual_results = []
            
            results = search_tracks_flexible(query, limit=10)
            
            # Convert SearchResult to RefinementCandidate
            seen_track_ids = {c.track_id for c in self.candidates}
            
            for result in results:
                if result.track_id in seen_track_ids:
                    continue
                
                candidate = RefinementCandidate(
                    track_id=result.track_id,
                    display_text=result.display_text,
                    variation_source="Manual",
                    isrc=result.isrc,
                    popularity=result.popularity
                )
                self.manual_results.append(candidate)
            
            # Select first manual result if any
            if self.manual_results:
                self.selected_idx = len(self.candidates)
            
        except Exception as e:
            logger.error(f"Manual search failed: {e}")
            self.manual_results = []

def refine_audio_file_match(resolved: ResolvedAudioFile) -> Optional[str]:
    """Convenience function to show refinement dialog.
    
    Args:
        resolved: ResolvedAudioFile from initial resolution
        
    Returns:
        Selected track_id or None if cancelled
    """
    dialog = AudioFileRefinementDialog()
    return dialog.show(resolved)
