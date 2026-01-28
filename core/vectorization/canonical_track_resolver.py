# core/vectorization/canonical_track_resolver.py
import os
import sqlite3
from config import PathConfig
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from .track_vectorizer import build_track_vector

class TrackResolver:
    """Resolves tracks to their canonical versions with audio features."""
    
    def __init__(self, 
                 main_db_path: Optional[str] = None,
                 audio_db_path: Optional[str] = None):
        """
        Initialize track resolver with database paths.
        
        Args:
            main_db_path: Path to main Spotify database
            audio_db_path: Path to audio features database
        """
        # Use PathConfig defaults if not provided
        self.main_db_path = main_db_path or str(PathConfig.get_main_db())
        self.audio_db_path = audio_db_path or str(PathConfig.get_audio_db())
        
        # Verify paths exist
        if not Path(self.main_db_path).exists():
            raise FileNotFoundError(f"Main database not found: {self.main_db_path}")
        if not Path(self.audio_db_path).exists():
            raise FileNotFoundError(f"Audio database not found: {self.audio_db_path}")
    
    def resolve_track_id(self, track_id: str, use_canonical: bool = True) -> Tuple[str, bool, Dict]:
        """
        Resolve a track ID to its best available version.
        
        Args:
            track_id: Spotify track ID
            use_canonical: Whether to find canonical version
            
        Returns:
            Tuple: (resolved_id, was_resolved, resolution_info)
        """
        if not use_canonical:
            return track_id, False, {"strategy": "exact_match"}
        
        # Strategy 1: Check if this track has audio features
        if self._has_audio_features(track_id):
            return track_id, False, {"strategy": "exact_match", "reason": "Track has audio features"}
        
        # Strategy 2: Find by ISRC
        canonical_id, isrc = self._resolve_by_isrc(track_id)
        if canonical_id and canonical_id != track_id:
            return canonical_id, True, {"strategy": "isrc_resolution", "original_isrc": isrc, "reason": "Found canonical version"}
        
        # Strategy 3: Find by metadata
        canonical_id, metadata = self._resolve_by_metadata(track_id)
        if canonical_id and canonical_id != track_id:
            return canonical_id, True, {"strategy": "metadata_resolution", "original_metadata": metadata, "reason": "Found similar track by metadata"}
        
        # Strategy 4: Return original
        return track_id, False, {"strategy": "exact_match", "reason": "No canonical version found"}

    def _has_audio_features(self, track_id: str) -> bool:
        """Check if a track has valid audio features."""
        conn = sqlite3.connect(self.audio_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM track_audio_features WHERE track_id = ? AND null_response = 0 LIMIT 1", (track_id,))
        result = cursor.fetchone() is not None
        conn.close()
        return result
    
    def _resolve_by_isrc(self, track_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve track by ISRC."""
        conn = sqlite3.connect(self.main_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT external_id_isrc FROM tracks WHERE id = ? LIMIT 1", (track_id,))
        result = cursor.fetchone()
        if not result or not result[0]:
            conn.close()
            return None, None
        isrc = result[0]
        
        cursor.execute("""
            SELECT t.id, t.popularity
            FROM tracks t
            WHERE t.external_id_isrc = ?
            ORDER BY t.popularity DESC
        """, (isrc,))
        versions = cursor.fetchall()
        conn.close()
        
        if not versions:
            return None, isrc
        
        # Find first version with audio features
        for version_id, popularity in versions:
            if self._has_audio_features(version_id):
                return version_id, isrc
        
        # If no version has audio features, return the most popular one
        # This ensures we always return a valid track ID instead of None
        return versions[0][0], isrc
    
    def _resolve_by_metadata(self, track_id: str) -> Tuple[Optional[str], Dict]:
        """Resolve track by metadata."""
        conn = sqlite3.connect(self.main_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.name, GROUP_CONCAT(DISTINCT art.name) as artists
            FROM tracks t
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists art ON ta.artist_rowid = art.rowid
            WHERE t.id = ?
            GROUP BY t.id
        """, (track_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None, {}
        name, artists = result
        artist_list = artists.split(',') if artists else []
        
        # Build query to find similar tracks
        query = """
            SELECT t.id
            FROM tracks t
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists art ON ta.artist_rowid = art.rowid
            WHERE t.name = ? AND art.name IN ({})
            GROUP BY t.id
            ORDER BY t.popularity DESC
            LIMIT 10
        """.format(','.join(['?'] * len(artist_list)))
        
        cursor.execute(query, [name] + artist_list)
        similar_tracks = cursor.fetchall()
        conn.close()
        
        # Find first similar track with audio features
        for (version_id,) in similar_tracks:
            if version_id == track_id:
                continue
            if self._has_audio_features(version_id):
                return version_id, {
                    "original_name": name,
                    "original_artists": artists,
                }
        
        return None, {"original_name": name, "original_artists": artists}
    
    def get_track_info(self, track_id: str) -> Dict:
        """Get comprehensive track info."""
        conn = sqlite3.connect(self.main_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.name, t.popularity, t.duration_ms, t.explicit,
                   a.name, a.release_date,
                   GROUP_CONCAT(DISTINCT art.name),
                   GROUP_CONCAT(DISTINCT ag.genre),
                   MAX(art.followers_total)
            FROM tracks t
            JOIN albums a ON t.album_rowid = a.rowid
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists art ON ta.artist_rowid = art.rowid
            LEFT JOIN artist_genres ag ON art.rowid = ag.artist_rowid
            WHERE t.id = ?
            GROUP BY t.id
        """, (track_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {}
        
        return {
            "track_id": track_id,
            "track_name": result[0],
            "popularity": result[1],
            "duration_ms": result[2],
            "explicit": bool(result[3]),
            "album": result[4],
            "release_date": result[5],
            "artists": result[6].split(',') if result[6] else [],
            "genres": result[7].split(',') if result[7] else [],
            "max_followers": result[8]
        }

class CanonicalVectorBuilder:
    """Builds track vectors with canonical resolution."""
    
    def __init__(self, 
                 main_db_path: Optional[str] = None,
                 audio_db_path: Optional[str] = None):
        """
        Initialize vector builder.
        
        Args:
            main_db_path: Path to main Spotify database
            audio_db_path: Path to audio features database
        """
        # Use PathConfig defaults if not provided
        self.main_db_path = main_db_path or str(PathConfig.get_main_db())
        self.audio_db_path = audio_db_path or str(PathConfig.get_audio_db())
        
        # Verify paths exist
        if not Path(self.main_db_path).exists():
            raise FileNotFoundError(f"Main database not found: {self.main_db_path}")
        if not Path(self.audio_db_path).exists():
            raise FileNotFoundError(f"Audio database not found: {self.audio_db_path}")

    def get_track_data(self, track_id: str) -> Optional[Dict]:
        """Get comprehensive track data for vector building."""
        resolver = get_resolver()
        resolved_id, was_resolved, _ = resolver.resolve_track_id(track_id, use_canonical=True)
        
        # Get metadata from main database
        conn = sqlite3.connect(self.main_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.rowid, t.id, t.name, t.duration_ms, t.popularity,
                   a.name AS album_name, a.release_date,
                   GROUP_CONCAT(DISTINCT art.name) AS artist_names,
                   MAX(art.followers_total) AS max_followers,
                   GROUP_CONCAT(DISTINCT ag.genre) AS genres
            FROM tracks t
            JOIN albums a ON t.album_rowid = a.rowid
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists art ON ta.artist_rowid = art.rowid
            LEFT JOIN artist_genres ag ON art.rowid = ag.artist_rowid
            WHERE t.id = ?
            GROUP BY t.rowid
        """, (resolved_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        track_data = dict(row)
        track_data['artist_names'] = track_data['artist_names'].split(',') if track_data['artist_names'] else []
        track_data['genres'] = track_data['genres'].split(',') if track_data['genres'] else []
        
        # Get audio features
        audio_conn = sqlite3.connect(self.audio_db_path)
        audio_conn.row_factory = sqlite3.Row
        audio_cursor = audio_conn.cursor()
        audio_cursor.execute("""
            SELECT *
            FROM track_audio_features 
            WHERE track_id = ? AND null_response = 0
        """, (resolved_id,))
        audio_row = audio_cursor.fetchone()
        audio_conn.close()
        
        if audio_row:
            track_data.update(dict(audio_row))
        
        # Add resolution info
        track_data['original_track_id'] = track_id
        track_data['resolved_track_id'] = resolved_id
        track_data['was_resolved'] = was_resolved
        
        return track_data
    
    def build_vector(self, track_id: str) -> Tuple[Optional[list], Optional[Dict]]:
        """Build vector for a track with canonical resolution."""
        track_data = self.get_track_data(track_id)
        if not track_data:
            return None, None
        
        vector = build_track_vector(track_data)
        return vector, track_data

# Singleton instances
_default_resolver = None
_default_builder = None

def get_resolver(main_db_path: Optional[str] = None, 
                 audio_db_path: Optional[str] = None) -> TrackResolver:
    """Get singleton TrackResolver instance."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = TrackResolver(
            main_db_path=main_db_path,
            audio_db_path=audio_db_path
        )
    return _default_resolver

def get_builder(main_db_path: Optional[str] = None, 
                audio_db_path: Optional[str] = None) -> CanonicalVectorBuilder:
    """Get singleton CanonicalVectorBuilder instance."""
    global _default_builder
    if _default_builder is None:
        _default_builder = CanonicalVectorBuilder(
            main_db_path=main_db_path,
            audio_db_path=audio_db_path
        )
    return _default_builder

def resolve_track(track_id: str, use_canonical: bool = True) -> Tuple[str, Dict]:
    """Resolve track to canonical version with resolution info."""
    resolver = get_resolver()
    resolved_id, was_resolved, info = resolver.resolve_track_id(track_id, use_canonical)
    return resolved_id, {
        "original_track_id": track_id,
        "resolved_track_id": resolved_id,
        "was_resolved": was_resolved,
        "resolution_info": info
    }

def build_canonical_vector(track_id: str) -> Tuple[Optional[list], Optional[Dict]]:
    """Build canonical vector for a track with resolution info."""
    builder = get_builder()
    return builder.build_vector(track_id)
