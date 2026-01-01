# core/similarity_engine/metadata_service.py
"""
Enrich track IDs with human-readable metadata from the Spotify databases
"""
import sqlite3
from typing import Dict, List, Optional
from pathlib import Path

class MetadataManager:
    """Manages metadata fetched from the Spotify databases."""
    
    def __init__(self, metadata_db: Optional[str] = None):
        """
        Initialize metadata manager.
        
        Args:
            metadata_db: Path to SQLite database (spotify_clean.sqlite3)
        """
        self.metadata_db = Path(metadata_db) if metadata_db else None
        self.conn = None
    
    def connect(self) -> bool:
        """Connect to metadata database."""
        if not self.metadata_db or not self.metadata_db.exists():
            return False
        
        try:
            self.conn = sqlite3.connect(str(self.metadata_db))
            # Use default row factory (tuples)
            # self.conn.row_factory = None  # This is the default
            return True
        except Exception as e:
            print(f"⚠️  Could not connect to metadata database: {e}")
            return False

    def get_track_metadata(self, track_id: str) -> Dict[str, Optional[str]]:
        """
        Get metadata for a single track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dictionary with track_name, artist_name, album_name, album_release_year
        """
        if not self.conn and not self.connect():
            return self._default_metadata()
        
        try:
            cursor = self.conn.cursor()
            
            # Detect schema and query appropriately
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            rows = cursor.fetchall()
            # Each row is a tuple with one element (the table name)
            tables = [row[0] for row in rows]
            
            if 'tracks' in tables and 'albums' in tables and 'artists' in tables:
                # spotify_clean.sqlite3 schema
                cursor.execute("""
                    SELECT 
                        t.name AS track_name,
                        alb.name AS album_name,
                        alb.release_date,
                        GROUP_CONCAT(art.name, ', ') AS artist_name
                    FROM tracks t
                    JOIN albums alb ON t.album_rowid = alb.rowid
                    JOIN track_artists ta ON t.rowid = ta.track_rowid
                    JOIN artists art ON ta.artist_rowid = art.rowid
                    WHERE t.id = ?
                    GROUP BY t.rowid
                    LIMIT 1
                """, (track_id,))
            else:
                # Simple schema (spotify_tracks.db)
                cursor.execute("""
                    SELECT track_name, artist_name, album_name, album_release_year
                    FROM tracks 
                    WHERE track_id = ?
                    LIMIT 1
                """, (track_id,))
            
            row = cursor.fetchone()
            
            if row:
                # Convert tuple to dictionary
                if len(row) == 4:  # Complex query result
                    track_name, album_name, release_date, artist_name = row
                    metadata = {
                        'track_name': track_name or 'Unknown',
                        'artist_name': artist_name or 'Unknown',
                        'album_name': album_name or 'Unknown',
                    }
                    if release_date and len(str(release_date)) >= 4:
                        metadata['album_release_year'] = str(release_date)[:4]
                    else:
                        metadata['album_release_year'] = None
                    return metadata
                elif len(row) == 4:  # Simple query result (also 4 columns)
                    track_name, artist_name, album_name, year = row
                    return {
                        'track_name': track_name or 'Unknown',
                        'artist_name': artist_name or 'Unknown',
                        'album_name': album_name or 'Unknown',
                        'album_release_year': year
                    }
            
        except Exception as e:
            print(f"⚠️  Error fetching metadata for {track_id}: {e}")
        
        return self._default_metadata()
    
    def get_track_metadata_batch(self, track_ids: List[str]) -> List[Dict[str, Optional[str]]]:
        """
        Get metadata for multiple tracks efficiently.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            List of metadata dictionaries in same order as input
        """
        if not self.conn and not self.connect():
            return [self._default_metadata() for _ in track_ids]
        
        try:
            placeholders = ','.join(['?'] * len(track_ids))
            cursor = self.conn.cursor()
            
            # Check schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            rows = cursor.fetchall()
            tables = [row[0] for row in rows]
            
            if 'tracks' in tables and 'albums' in tables and 'artists' in tables:
                cursor.execute(f"""
                    SELECT 
                        t.id AS track_id,
                        t.name AS track_name,
                        alb.name AS album_name,
                        alb.release_date,
                        GROUP_CONCAT(art.name, ', ') AS artist_name
                    FROM tracks t
                    JOIN albums alb ON t.album_rowid = alb.rowid
                    JOIN track_artists ta ON t.rowid = ta.track_rowid
                    JOIN artists art ON ta.artist_rowid = art.rowid
                    WHERE t.id IN ({placeholders})
                    GROUP BY t.rowid
                """, track_ids)
                
                rows = cursor.fetchall()
                
                # Create lookup dictionary
                metadata_dict = {}
                for row in rows:
                    # row is a tuple: (track_id, track_name, album_name, release_date, artist_name)
                    track_id_val = row[0]
                    track_name = row[1] or 'Unknown'
                    album_name = row[2] or 'Unknown'
                    release_date = row[3]
                    artist_name = row[4] or 'Unknown'
                    
                    year = None
                    if release_date and len(str(release_date)) >= 4:
                        year = str(release_date)[:4]
                    
                    metadata_dict[track_id_val] = {
                        'track_name': track_name,
                        'artist_name': artist_name,
                        'album_name': album_name,
                        'album_release_year': year
                    }
                
                # Return in same order as input
                return [metadata_dict.get(tid, self._default_metadata()) for tid in track_ids]
                
            else:
                cursor.execute(f"""
                    SELECT track_id, track_name, artist_name, album_name, album_release_year
                    FROM tracks 
                    WHERE track_id IN ({placeholders})
                """, track_ids)
                
                rows = cursor.fetchall()
                
                # Create lookup dictionary
                metadata_dict = {}
                for row in rows:
                    # row is a tuple: (track_id, track_name, artist_name, album_name, album_release_year)
                    track_id_val = row[0]
                    metadata_dict[track_id_val] = {
                        'track_name': row[1] or 'Unknown',
                        'artist_name': row[2] or 'Unknown',
                        'album_name': row[3] or 'Unknown',
                        'album_release_year': row[4]
                    }
                
                # Return in same order as input
                return [metadata_dict.get(tid, self._default_metadata()) for tid in track_ids]
            
        except Exception as e:
            print(f"⚠️  Error fetching metadata batch: {e}")
            import traceback
            traceback.print_exc()
            return [self._default_metadata() for _ in track_ids]
    
    def _default_metadata(self) -> Dict[str, Optional[str]]:
        """Return default metadata when track not found."""
        return {
            'track_name': 'Unknown',
            'artist_name': 'Unknown',
            'album_name': 'Unknown',
            'album_release_year': None
        }
    
    def format_track_display(self, track_id: str, similarity: Optional[float] = None) -> str:
        """
        Format track for display.
        
        Args:
            track_id: Spotify track ID
            similarity: Optional similarity score
            
        Returns:
            Formatted string
        """
        metadata = self.get_track_metadata(track_id)
        year_str = f" ({metadata['album_release_year']})" if metadata['album_release_year'] else ""
        
        if similarity is not None:
            return f"{metadata['track_name']} - {metadata['artist_name']}{year_str} - {similarity:.4f}"
        else:
            return f"{metadata['track_name']} - {metadata['artist_name']}{year_str}"
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        """Cleanup."""
        self.close()
