# core/preprocessing/querying/query_tokenizer.py
"""
Query Tokenizer
===============
Simple, deterministic tokenization for semantic search.
"""
import re
from typing import List

def normalize_token(text: str) -> str:
    """
    Normalize text for tokenization.
    - Lowercase
    - Keep apostrophes within words (rock'n'roll)
    - Replace other punctuation with spaces
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace non-apostrophe punctuation with spaces
    # Keep apostrophes that are inside words
    text = re.sub(r"[^a-z0-9']+", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def tokenize(text: str, field_prefix: str = "") -> List[str]:
    """
    Tokenize text into UNIGRAMS ONLY. No bigrams.
    
    Args:
        text: Input text (track name, artist name, etc.)
        field_prefix: "" for track, "a_" for artist, "al_" for album
    
    Returns:
        List of prefixed unigram tokens
    """
    normalized = normalize_token(text)
    if not normalized:
        return []
    
    # Split into unigrams only
    unigrams = normalized.split()
    
    # Apply field prefix to all tokens
    return [f"{field_prefix}{token}" for token in unigrams]

def tokenize_track_name(name: str) -> List[str]:
    """Tokenize track name (no prefix)"""
    return tokenize(name, field_prefix="")

def tokenize_artist_name(name: str) -> List[str]:
    """
    Tokenize artist name with 'a_' prefix for weighting.
    Only tokenizes the first artist to prevent explosion on multi-artist tracks.
    """
    if not name:
        return []
    
    # Take first artist only (e.g., "Keane, The Fray" → "Keane")
    first_artist = name.split(',')[0].strip()
    return tokenize(first_artist, field_prefix="a_")

def tokenize_album_name(name: str) -> List[str]:
    """Tokenize album name with 'al_' prefix for weighting"""
    return tokenize(name, field_prefix="al_")
