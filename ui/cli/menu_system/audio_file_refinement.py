# ui/cli/menu_system/audio_file_refinement.py
""" 
Audio File Refinement UI 
========================
Refinement dialog that mirrors the text search UI.
Uses interactive_text_search with pre-populated results from variations.
"""

import logging
from typing import Optional, List

from core.utilities.audio_file_input_processor import ResolvedAudioFile
from core.utilities.text_search_utils import SearchResult
from ui.cli.menu_system.main_menu_handlers.text_search import interactive_text_search

logger = logging.getLogger(__name__)


def refine_audio_file_match(resolved: ResolvedAudioFile) -> Optional[str]:
    """Show refinement dialog using text search UI.
    
    Displays all variation results in the standard search interface, 
    allowing user to select or type a new query.
    
    Args:
        resolved: Resolved audio file with variations
        
    Returns:
        Selected track_id or None if cancelled
    """
    # Convert all variation results to SearchResult objects
    initial_results: List[SearchResult] = []
    seen_track_ids: set = set()
    
    # Add results from ALL variations (in priority order)
    for variation in resolved.audio_file_input.variations:
        if not variation.results:
            continue
        for result in variation.results:
            track_id = result.get('track_id')
            if not track_id or track_id in seen_track_ids:
                continue
            seen_track_ids.add(track_id)
            
            # Create SearchResult
            sr = SearchResult(
                track_id=track_id,
                track_name=result.get('track_name', 'Unknown'),
                artist_name=result.get('artist_name', 'Unknown'),
                album_name=result.get('album_name', ''),
                album_release_year=result.get('album_release_year'),
                popularity=result.get('popularity', 0),
                isrc=result.get('isrc'),
                confidence=0.0,
                matched_tokens={}
            )
            initial_results.append(sr)
    
    # Sort by popularity (most likely correct first)
    initial_results.sort(key=lambda x: x.popularity, reverse=True)
    
    # Call the standard text search interface with our pre-populated results
    subtitle = f"Refining match for: {resolved.audio_file_input.filename}"
    
    return interactive_text_search(
        initial_query="",  # Start with empty query
        initial_results=initial_results,
        header_title="Refine Match",
        subtitle=subtitle,
        show_progress=False  # Don't show "Searching..." since we have results already
    )
