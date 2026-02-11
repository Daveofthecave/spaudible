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

def is_valid_isrc(isrc: str) -> bool:
    """
    Strict ISRC validation according to ISO 3901 standard.
    Format: CC-XXX-YY-NNNNN or CCXXXYYNNNNNN where:
    - CC: 2 letters (country code, e.g., US, GB, FR)
    - XXX: 3 alphanumeric (registrant code)
    - YY: 2 digits (year, 00-99)
    - NNNNN: 5 digits (recording designation)
    """
    if not isrc:
        return False
    
    # Remove hyphens and normalize
    compact = isrc.replace('-', '').upper()
    
    if len(compact) != 12:
        return False
    
    # CC: Country code must be 2 letters
    if not compact[0:2].isalpha():
        return False
    
    # XXX: Registrant code must be 3 alphanumeric characters
    if not compact[2:5].isalnum():
        return False
    
    # YY: Year must be 2 digits (00-99)
    if not compact[5:7].isdigit():
        return False
    
    # NNNNN: Designation must be 5 digits
    if not compact[7:12].isdigit():
        return False
    
    return True

def detect_input_type(user_input: str) -> Tuple[str, Any]:
    """
    Detect input type with strict validation to prevent false positives.
    
    Detection hierarchy (first match wins):
    1. Spotify track URLs (eg. https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8)
    2. Spotify playlist URLs (eg. https://open.spotify.com/playlist/37i9dQZF1EIec0dMqGbsyB)
    3. ISRC codes (eg. USIR20400274)
    4. Spotify track IDs (eg. 0eGsygTp906u18L0Oimnem)
    5. Local audio file paths (must look like path or have audio extension)
    6. Text queries (fallback for any remaining input including "AC/DC", "song name", etc.)
    
    Args:
        user_input: Raw user input string
    
    Returns:
        Tuple of (input_type_key, extracted_data)
    """
    if not user_input:
        return "unknown", None
    
    user_input = user_input.strip()
    processed_data = None
    
    # 1. Spotify track URLs - check before anything else since they're very specific
    if "open.spotify.com/track/" in user_input.lower():
        track_id = extract_spotify_track_id(user_input)
        if track_id:
            return "spotify_track_url", track_id
    
    # 2. Spotify playlist URLs
    elif "open.spotify.com/playlist/" in user_input.lower():
        playlist_id = extract_spotify_playlist_id(user_input)
        if playlist_id:
            return "spotify_playlist_url", playlist_id
    
    # 3. ISRC codes - strict validation to avoid matching words like "illumination"
    #    ISRCs never contain spaces, so skip if space present
    if ' ' not in user_input and len(user_input.replace('-', '')) == 12:
        if is_valid_isrc(user_input):
            # Return non-hyphenated for consistency
            return "isrc_code", user_input.replace('-', '').upper()
    
    # 4. Spotify track IDs (22 characters, base62)
    #    Must be exactly 22 chars, no spaces, and valid base62 characters
    if (len(user_input) == 22 and 
        ' ' not in user_input and 
        re.match(r'^[A-Za-z0-9_-]+$', user_input) and
        # Must also contain at least one letter AND at least one digit,
        # since there are no purely alphabetical or purely numerical track IDs
        # in the Spotify database
        re.search(r'[A-Za-z]', user_input) and 
        re.search(r'[0-9]', user_input)):
        return "track_id", user_input
    
    # 5. Audio file paths - tightened to avoid "AC/DC" false positives
    #    After quote removal, must either:
    #      - Start with explicit path indicators (/, ./, ~/, C:\, etc.)
    #      - OR contain path separators AND have audio file extension
    user_input_quoteless = user_input.strip('"\'')
    has_audio_ext = os.path.splitext(user_input_quoteless)[1].lower() in {
        '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', 
        '.oga', '.m4b', '.wma', '.aiff', '.opus'
    }
    
    is_audio_file_path = (
        # Unix: absolute or relative paths
        user_input_quoteless.startswith(('./', '../', '~/', '/')) or
        # Windows: drive letters or UNC paths  
        re.match(r'^[A-Za-z]:[/\\]', user_input_quoteless) or
        re.match(r'^\\\\', user_input_quoteless) or  # UNC paths \\server\share
        # Has separator and extension (e.g., "music/song.mp3" or "music\song.mp3")
        (('/' in user_input_quoteless or '\\' in user_input_quoteless) and has_audio_ext)
    )
    
    if is_audio_file_path:
        expanded_path = os.path.expanduser(user_input_quoteless)
        # Only accept if file exists OR has audio extension (avoid "AC/DC" which has no ext)
        if os.path.exists(expanded_path) or has_audio_ext:
            # Verify it's not a URL that happens to have a dot
            if not user_input_quoteless.startswith('http'):
                return "audio_file", expanded_path
    
    # Check for bare filename with audio extension in current directory (terminal drag-drop)
    if has_audio_ext and '/' not in user_input_quoteless and '\\' not in user_input_quoteless:
        cwd_path = Path.cwd() / user_input_quoteless
        if cwd_path.exists():
            return "audio_file", str(cwd_path)
    
    # 6. Text queries
    #    This catches:
    #      - "AC/DC" (has slash but not a file path)
    #      - "illumination" (12 chars but not ISRC format)
    #      - "Keane Perfect Symmetry"
    if len(user_input) >= 1:
        return "text_query", user_input
    
    return "unknown", None

def extract_spotify_track_id(url: str) -> Optional[str]:
    """ Extract track ID from various Spotify URL formats. """
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
    """ Extract playlist ID from various Spotify URL formats. """
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

def route_input(user_input: str) -> Tuple[str, Any]:
    """
    Backwards-compatible alias for detect_input_type.
    """
    return detect_input_type(user_input)
