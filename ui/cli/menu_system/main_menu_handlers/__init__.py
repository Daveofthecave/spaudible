# ui/cli/menu_system/main_menu_handlers/__init__.py
"""
Main Menu Handlers Package
"""
from .core_search import handle_core_search
from .settings_manager import handle_settings
from .input_router import detect_input_type, InputType
from .text_search import interactive_text_search, simple_text_search_fallback
from .utils import (
    extract_track_id,
    extract_playlist_id,
    save_playlist,
    check_preprocessed_files,
    format_track_display,
    get_metadata_db_path,
    validate_vector,
    get_similarity_color,
    format_elapsed_time
)

__all__ = [
    'handle_core_search',
    'handle_settings',
    'detect_input_type',
    'InputType',
    'interactive_text_search',
    'simple_text_search_fallback',
    'extract_track_id',
    'extract_playlist_id',
    'save_playlist',
    'check_preprocessed_files',
    'format_track_display',
    'get_metadata_db_path',
    'validate_vector',
    'get_similarity_color',
    'format_elapsed_time'
]
