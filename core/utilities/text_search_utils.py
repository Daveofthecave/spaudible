# core/utilities/text_search_utils.py
"""
Text-based search utilities for Spaudible.
Provides permutation-aware search using existing SQLite indexes.
No new disk indexes required.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import sqlite3
from config import PathConfig

@dataclass
class SearchResult:
    """Represents a single search result."""
    track_id: str
    track_name: str
    artist_name: str
    popularity: int
    confidence: float = 1.0
    final_score: float = 0.0
    
    @property
    def display_text(self) -> str:
        """Formatted display string for CLI."""
        year = self.extract_year()
        year_str = f" ({year})" if year else ""
        return f"{self.track_name} - {self.artist_name}{year_str}"
    
    def extract_year(self) -> Optional[str]:
        """Extract year from release_date if available."""
        # This would be fetched from the database if needed
        return None

def parse_query_permutations(query: str) -> List[Dict[str, Any]]:
    """
    Parse query into multiple permutations of artist/track/album.
    
    For "Keane Perfect Symmetry", generates:
    [
        {"artist": "Keane", "track": "Perfect Symmetry", "score": 1.0},
        {"artist": "", "track": "Keane Perfect Symmetry", "score": 0.5},
        {"artist": "Keane Perfect", "track": "Symmetry", "score": 0.7},
        # ... etc.
    ]
    
    Scoring is based on token distribution balance.
    """
    if not query or not query.strip():
        return []
    
    tokens = query.strip().split()
    if len(tokens) == 1:
        # Single token: treat as track name only
        return [{"artist": "", "track": tokens[0], "score": 1.0}]
    
    permutations = []
    
    # Generate splits at every position
    for i in range(1, len(tokens)):
        artist_part = " ".join(tokens[:i])
        track_part = " ".join(tokens[i:])
        
        # Score based on balance (prefer 2-3 token artists, 2-5 token tracks)
        artist_tokens = len(tokens[:i])
        track_tokens = len(tokens[i:])
        balance_penalty = abs(artist_tokens - track_tokens) * 0.1
        
        # Bonus for reasonable token counts
        artist_bonus = 0.2 if 1 <= artist_tokens <= 3 else 0
        track_bonus = 0.2 if 2 <= track_tokens <= 5 else 0
        
        score = 1.0 - balance_penalty + artist_bonus + track_bonus
        score = max(0.1, min(1.0, score))
        
        permutations.append({
            "artist": artist_part,
            "track": track_part,
            "score": score
        })
    
    # Add whole-string as track-only (lowest priority)
    permutations.append({
        "artist": "",
        "track": query,
        "score": 0.3
    })
    
    # Sort by score descending
    return sorted(permutations, key=lambda x: x["score"], reverse=True)

def search_tracks_by_permutations(
    permutations: List[Dict[str, Any]], 
    limit: int = 50
) -> List[SearchResult]:
    """
    Search using multiple permutations and merge results.
    Ranks by: match_quality × permutation_score × popularity
    """
    if not permutations:
        return []
    
    all_results = []
    seen_track_ids = set()
    
    # Use single database connection for all queries
    main_db_path = PathConfig.get_main_db()
    if not main_db_path.exists():
        raise FileNotFoundError(f"Main database not found: {main_db_path}")
    
    conn = sqlite3.connect(str(main_db_path))
    conn.row_factory = sqlite3.Row
    
    try:
        for perm in permutations:
            # Stop if we have enough results
            if len(all_results) >= limit:
                break
            
            # Search exact match first
            results = _search_exact(conn, perm["track"], perm["artist"])
            for result in results:
                if result.track_id not in seen_track_ids:
                    result.confidence = perm["score"] * 1.0  # Exact match bonus
                    all_results.append(result)
                    seen_track_ids.add(result.track_id)
            
            # Search prefix match if needed
            if len(all_results) < limit:
                results = _search_prefix(conn, perm["track"], perm["artist"])
                for result in results:
                    if result.track_id not in seen_track_ids:
                        result.confidence = perm["score"] * 0.7  # Prefix penalty
                        all_results.append(result)
                        seen_track_ids.add(result.track_id)
            
            # Track-only exact match
            if len(all_results) < limit and perm["artist"]:
                results = _search_exact(conn, perm["track"], "")
                for result in results:
                    if result.track_id not in seen_track_ids:
                        result.confidence = perm["score"] * 0.5
                        all_results.append(result)
                        seen_track_ids.add(result.track_id)
        
        # Calculate final score and sort
        for result in all_results:
            # Normalize popularity to 0-1
            popularity_score = min(result.popularity / 100.0, 1.0)
            result.final_score = result.confidence * popularity_score
        
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        
    finally:
        conn.close()
    
    return all_results[:limit]

def _search_exact(
    conn: sqlite3.Connection, 
    track_name: str, 
    artist_name: str
) -> List[SearchResult]:
    """
    Exact match search using existing indexes.
    """
    query = """
        SELECT DISTINCT t.id, t.name, art.name as artist_name, t.popularity 
        FROM tracks t 
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        JOIN artists art ON ta.artist_rowid = art.rowid
        WHERE LOWER(t.name) = LOWER(?)
    """
    params = [track_name]
    
    if artist_name:
        query += " AND LOWER(art.name) = LOWER(?)"
        params.append(artist_name)
    
    query += " ORDER BY t.popularity DESC LIMIT 10"
    
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    return [
        SearchResult(
            track_id=row["id"],
            track_name=row["name"],
            artist_name=row["artist_name"],
            popularity=row["popularity"],
            confidence=1.0
        )
        for row in cursor.fetchall()
    ]

def _search_prefix(
    conn: sqlite3.Connection, 
    track_name: str, 
    artist_name: str
) -> List[SearchResult]:
    """
    Prefix match search using existing indexes.
    """
    query = """
        SELECT DISTINCT t.id, t.name, art.name as artist_name, t.popularity 
        FROM tracks t 
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        JOIN artists art ON ta.artist_rowid = art.rowid
        WHERE LOWER(t.name) LIKE LOWER(?)
    """
    params = [f"{track_name}%"]
    
    if artist_name:
        query += " AND LOWER(art.name) LIKE LOWER(?)"
        params.append(f"{artist_name}%")
    
    query += " ORDER BY t.popularity DESC LIMIT 10"
    
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    return [
        SearchResult(
            track_id=row["id"],
            track_name=row["name"],
            artist_name=row["artist_name"],
            popularity=row["popularity"],
            confidence=0.7
        )
        for row in cursor.fetchall()
    ]

def extract_track_id_from_result(result: SearchResult) -> str:
    """Extract track_id for similarity search."""
    return result.track_id

def format_search_results(results: List[SearchResult]) -> str:
    """Format results for display in CLI."""
    if not results:
        return "No results found."
    
    lines = []
    for idx, result in enumerate(results[:20], 1):
        lines.append(f"{idx:2d}. {result.display_text}")
    
    return "\n".join(lines)
