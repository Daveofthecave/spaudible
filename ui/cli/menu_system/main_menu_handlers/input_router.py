# ui/cli/menu_system/main_menu_handlers/input_router.py
import os
import re
from enum import Enum
from pathlib import Path
from typing import Tuple, Any, Optional

class InputType(Enum):
    """Enum for input type classification."""
    SPOTIFY_TRACK = "Spotify Track URL"
    SPOTIFY_PLAYLIST = "Spotify Playlist URL"
    TRACK_ID = "Spotify Track ID"
    FILE_PATH = "Audio File"
    ISRC_CODE = "ISRC Code"
    TEXT_QUERY = "Text Search Query"
    UNKNOWN = "Unknown Input"

def detect_input_type(user_input: str) -> Tuple[str, Any]:
    """
    Detect input type and extract relevant data.
    
    Detection hierarchy (first match wins):
    1. ISRC codes (12 characters)
    2. Spotify track URLs
    3. Spotify playlist URLs
    4. Spotify track IDs (22 characters)
    5. Local audio file paths
    6. Text queries (fallback for any other string)
    
    Args:
        user_input: Raw user input string
        
    Returns:
        Tuple of (input_type_key, extracted_data)
    """
    if not user_input:
        return "unknown", None
    
    user_input = user_input.strip()
    detected_type = InputType.UNKNOWN
    processed_data = None

    # 1. ISRC codes (12 characters: CC-XXX-YY-NNNNN or CCXXXYYNNNNN format)
    # Accept both formatted and compact forms
    compact_isrc = user_input.replace('-', '').upper()
    if len(compact_isrc) == 12 and compact_isrc[:2].isalpha() and compact_isrc[2:].isalnum():
        detected_type = InputType.ISRC_CODE
        processed_data = compact_isrc
    
    # 2. Spotify track URLs
    elif "open.spotify.com/track/" in user_input.lower():
        track_id = extract_spotify_track_id(user_input)
        if track_id:
            detected_type = InputType.SPOTIFY_TRACK
            processed_data = track_id
    
    # 3. Spotify playlist URLs
    elif "open.spotify.com/playlist/" in user_input.lower():
        playlist_id = extract_spotify_playlist_id(user_input)
        if playlist_id:
            detected_type = InputType.SPOTIFY_PLAYLIST
            processed_data = playlist_id
    
    # 4. Spotify track IDs (22 chars)
    elif len(user_input) == 22 and re.match(r'^[A-Za-z0-9_-]+$', user_input):
        detected_type = InputType.TRACK_ID
        processed_data = user_input
    
    # 5. Local audio file paths
    elif os.path.sep in user_input or user_input.startswith('.') or user_input.startswith('~'):
        expanded_path = os.path.expanduser(user_input)
        if os.path.exists(expanded_path):
            audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.oga', '.m4b'}
            if Path(expanded_path).suffix.lower() in audio_extensions:
                detected_type = InputType.FILE_PATH
                processed_data = expanded_path
    
    # 6. Text queries (fallback)
    elif len(user_input) >= 2:
        detected_type = InputType.TEXT_QUERY
        processed_data = user_input

    # Map enum to string key for backward compatibility
    type_to_key = {
        InputType.SPOTIFY_TRACK: "spotify_track_url",
        InputType.SPOTIFY_PLAYLIST: "spotify_playlist_url",
        InputType.TRACK_ID: "track_id",
        InputType.FILE_PATH: "audio_file",
        InputType.ISRC_CODE: "isrc_code",
        InputType.TEXT_QUERY: "text_query",
        InputType.UNKNOWN: "unknown"
    }

    return type_to_key[detected_type], processed_data

def extract_spotify_track_id(url: str) -> Optional[str]:
    """
    Extract track ID from various Spotify URL formats.
    Supports:
    - https://open.spotify.com/track/ID
    - spotify:track:ID
    - track/ID
    """
    patterns = [
        r'open\.spotify\.com/track/([A-Za-z0-9_-]+)',
        r'spotify:track:([A-Za-z0-9_-]+)',
        r'track/([A-Za-z0-9_-]{22})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            track_id = match.group(1)
            if len(track_id) == 22:
                return track_id
    
    return None

def extract_spotify_playlist_id(url: str) -> Optional[str]:
    """
    Extract playlist ID from various Spotify URL formats.
    Supports:
    - https://open.spotify.com/playlist/ID
    - spotify:playlist:ID
    - playlist/ID
    """
    patterns = [
        r'open\.spotify\.com/playlist/([A-Za-z0-9_-]+)',
        r'spotify:playlist:([A-Za-z0-9_-]+)',
        r'playlist/([A-Za-z0-9_-]{22})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            playlist_id = match.group(1)
            if len(playlist_id) == 22:
                return playlist_id
    
    return None

def is_valid_isrc(isrc: str) -> bool:
    """
    Validate ISRC format:
    - Exactly 12 characters
    - First 2 characters: letters (country code)
    - Remaining 10 characters: alphanumeric
    """
    if len(isrc) != 12:
        return False
    
    # Country code must be 2 letters
    if not isrc[:2].isalpha():
        return False
    
    # Remainder must be alphanumeric
    if not isrc[2:].isalnum():
        return False
    
    return True

def route_input(user_input: str) -> Tuple[str, Any]:
    """
    Backward-compatible alias for detect_input_type.
    Maintains original function name for any code still using it.
    """
    return detect_input_type(user_input)
