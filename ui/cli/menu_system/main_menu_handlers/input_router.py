# ui/cli/menu_system/main_menu_handlers/input_router.py
import os
import re
from enum import Enum
from pathlib import Path

class InputType(Enum):
    SPOTIFY_TRACK = "Spotify Track URL"
    SPOTIFY_PLAYLIST = "Spotify Playlist URL"
    TRACK_ID = "Spotify Track ID"
    FILE_PATH = "Audio File"
    ISRC_CODE = "ISRC Code"
    TEXT_QUERY = "Text Search Query"
    UNKNOWN = "Unknown Input"

def route_input(user_input: str):
    """Detect input type and extract relevant data."""
    if not user_input:
        return "unknown", None

    user_input = user_input.strip()
    detected_type = InputType.UNKNOWN
    processed_data = None

    # Spotify track URLs
    if "open.spotify.com/track/" in user_input.lower():
        track_id = extract_spotify_track_id(user_input)
        if track_id:
            detected_type = InputType.SPOTIFY_TRACK
            processed_data = track_id
    
    # Spotify playlist URLs
    elif "open.spotify.com/playlist/" in user_input.lower():
        playlist_id = extract_spotify_playlist_id(user_input)
        if playlist_id:
            detected_type = InputType.SPOTIFY_PLAYLIST
            processed_data = playlist_id
    
    # Spotify track IDs
    elif len(user_input) == 22 and re.match(r'^[A-Za-z0-9_-]+$', user_input):
        detected_type = InputType.TRACK_ID
        processed_data = user_input
    
    # Local file paths
    elif os.path.sep in user_input or user_input.startswith('.') or user_input.startswith('~'):
        expanded_path = os.path.expanduser(user_input)
        if os.path.exists(expanded_path):
            audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.oga', '.m4b'}
            if Path(expanded_path).suffix.lower() in audio_extensions:
                detected_type = InputType.FILE_PATH
                processed_data = expanded_path
    
    # ISRC codes
    elif len(user_input) == 12 and user_input[:2].isalpha() and user_input[2:].isalnum():
        detected_type = InputType.ISRC_CODE
        processed_data = user_input
    
    # Text query
    elif len(user_input) >= 2:
        detected_type = InputType.TEXT_QUERY
        processed_data = user_input

    # Map to string key
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

def extract_spotify_track_id(url: str) -> str:
    """Extract track ID from Spotify URL."""
    patterns = [
        r'open\.spotify\.com/track/([A-Za-z0-9_-]+)',
        r'spotify:track:([A-Za-z0-9_-]+)',
        r'track/([A-Za-z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            track_id = match.group(1)
            if len(track_id) == 22:
                return track_id
    return None

def extract_spotify_playlist_id(url: str) -> str:
    """Extract playlist ID from Spotify URL."""
    patterns = [
        r'open\.spotify\.com/playlist/([A-Za-z0-9_-]+)',
        r'spotify:playlist:([A-Za-z0-9_-]+)',
        r'playlist/([A-Za-z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            playlist_id = match.group(1)
            if len(playlist_id) == 22:
                return playlist_id
    return None
