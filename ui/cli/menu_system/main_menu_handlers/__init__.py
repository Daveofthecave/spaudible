# ui/cli/menu_system/main_menu_handlers/__init__.py
"""
Main Menu Handlers Package
"""
from .core_search import handle_core_search
from .settings_manager import handle_settings
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
