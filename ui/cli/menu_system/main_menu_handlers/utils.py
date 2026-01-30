# ui/cli/menu_system/main_menu_handlers/utils.py
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Union
from config import PathConfig

def extract_track_id(input_str: str) -> Optional[str]:
    """Extract Spotify track ID from input string."""
    if not input_str:
        return None
    input_str = input_str.strip()
    
    if "open.spotify.com/track/" in input_str:
        parts = input_str.split("track/")
        if len(parts) > 1:
            track_id = parts[1].split("?")[0].split("&")[0].split("#")[0]
            return track_id.strip()
    
    if len(input_str) == 22 and re.match(r'^[A-Za-z0-9_-]+$', input_str):
        return input_str
    
    match = re.search(r'track/([A-Za-z0-9_-]{22})', input_str)
    if match:
        return match.group(1)
    
    match = re.search(r'([A-Za-z0-9_-]{22})', input_str)
    if match:
        return match.group(1)
    
    return None

def extract_playlist_id(input_str: str) -> Optional[str]:
    """Extract Spotify playlist ID from input string."""
    if not input_str:
        return None
    input_str = input_str.strip()
    
    if "open.spotify.com/playlist/" in input_str:
        parts = input_str.split("playlist/")
        if len(parts) > 1:
            playlist_id = parts[1].split("?")[0].split("&")[0].split("#")[0]
            return playlist_id.strip()
    
    if len(input_str) == 22 and re.match(r'^[A-Za-z0-9_-]+$', input_str):
        return input_str
    
    match = re.search(r'playlist/([A-Za-z0-9_-]{22})', input_str)
    if match:
        return match.group(1)
    
    return None

def save_playlist(results: List[Tuple], playlist_name: str) -> str:
    """Save search results as a playlist JSON file with proper serialization."""
    os.makedirs("playlists", exist_ok=True)
    playlist_data = {
        "name": playlist_name,
        "created": datetime.now().isoformat(),
        "tracks": []
    }
    
    for result in results:
        if len(result) == 3:
            track_id, similarity, metadata = result
            # Convert numpy float32 to native Python float
            similarity = float(similarity)
            
            track_info = {
                "track_id": track_id,
                "similarity": similarity,
                "track_name": metadata.get('track_name', 'Unknown'),
                "artist_name": metadata.get('artist_name', 'Unknown'),
                "album_name": metadata.get('album_name', 'Unknown'),
                "year": metadata.get('album_release_year')
            }
        else:
            track_id, similarity = result
            # Convert numpy float32 to native Python float
            similarity = float(similarity)
            
            track_info = {
                "track_id": track_id,
                "similarity": similarity,
                "track_name": "Unknown",
                "artist_name": "Unknown",
                "album_name": "Unknown",
                "year": None
            }
        playlist_data["tracks"].append(track_info)
    
    safe_name = re.sub(r'[^\w\s-]', '', playlist_name).strip()
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"playlists/{safe_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(playlist_data, f, indent=2)
    
    return filename

def check_preprocessed_files() -> Tuple[bool, Optional[str]]:
    """
    Check if preprocessed vector files exist and have correct sizes for new 104-byte format.
    """
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    
    if not vectors_path.exists():
        return False, "Vector file not found. Run preprocessing first."
    if not index_path.exists():
        return False, "Index file not found. Run preprocessing first."
    
    # NEW: 104 bytes per vector record (was 128)
    vectors_size = vectors_path.stat().st_size
    expected_min_size = 256_000_000 * 104  # 104 bytes per vector
    
    # Allow 10% margin for processing variations
    if vectors_size < expected_min_size * 0.9:
        return False, (
            f"Vector file size too small. Expected at least {expected_min_size:,} bytes, "
            f"got {vectors_size:,}. File may be incomplete or corrupted."
        )
    
    # NEW: 26 bytes per index entry (was 42: 22B ID + 8B offset + 12B ISRC)
    index_size = index_path.stat().st_size
    expected_index_size = 256_000_000 * 26  # 26 bytes per index entry
    
    if index_size < expected_index_size * 0.9:
        return False, (
            f"Index file size too small. Expected at least {expected_index_size:,} bytes, "
            f"got {index_size:,}. File may be incomplete or corrupted."
        )
    
    # Validate header info if metadata.json exists
    metadata_path = PathConfig.VECTORS / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            actual_tracks = metadata.get('total_tracks', 0)
            if actual_tracks < 256_000_000 * 0.99:
                return False, f"Index incomplete: {actual_tracks:,} tracks processed"
        except Exception:
            pass  # Metadata optional for validation
    
    return True, None

def format_track_display(track_id: str, similarity: float, metadata: Optional[Dict] = None) -> str:
    """Format a track for display."""
    if metadata:
        track_name = metadata.get('track_name', 'Unknown Track')
        artist_name = metadata.get('artist_name', 'Unknown Artist')
        year = metadata.get('album_release_year')
        year_str = f" ({year})" if year else ""
        return f"{similarity:.4f} - {track_name} - {artist_name}{year_str}"
    else:
        return f"{similarity:.4f} - {track_id}"

def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms      "
    elif seconds < 60:
        return f"{seconds:.1f}s      "
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.0f}s      "
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m      "

def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def get_metadata_db_path() -> Optional[str]:
    """Get path to metadata database."""
    possible_dbs = [
        PathConfig.get_main_db(),
        PathConfig.get_audio_db()
    ]
    for db_path in possible_dbs:
        if db_path.exists():
            return str(db_path)
    return None

def validate_vector(vector: List[float]) -> Tuple[bool, Optional[str]]:
    """Validate a vector for search."""
    if not vector:
        return False, "Vector is empty"
    if len(vector) != 32:
        return False, f"Vector must be 32-dimensional, got {len(vector)} dimensions"
    if not all(isinstance(v, (int, float)) for v in vector):
        return False, "Vector must contain only numbers"
    extreme_count = sum(1 for v in vector if abs(v) > 10)
    if extreme_count > 5:
        return False, f"Vector has {extreme_count} extreme values (>10)"
    return True, None

def get_similarity_color(similarity: float) -> str:
    """Get a color indicator for similarity score."""
    if similarity >= 0.9997:
        return "🔵"  # Blue - identical match
    if similarity >= 0.85:
        return "🟢"  # Green - excellent match
    elif similarity >= 0.7:
        return "🟡"  # Yellow - good match
    elif similarity >= 0.65:
        return "🟠"  # Orange - decent match
    elif similarity >= 0.5:
        return "🔴"  # Red - poor match
    else:
        return "🟣"  # Purple - terrible match
