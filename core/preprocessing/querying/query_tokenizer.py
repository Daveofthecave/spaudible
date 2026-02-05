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
    Normalize text for tokenization with full Unicode support.
    
    This function prepares text for tokenization by:
    - Converting to lowercase (including Unicode characters like "Å" → "å")
    - Preserving apostrophes within words (e.g., "rock'n'roll")
    - Preserving Unicode letters from all languages (e.g., "Håkan", "Björk", "Édith", "周杰伦")
    - Replacing all punctuation/special characters with spaces
    - Collapsing multiple spaces into single spaces
    
    Args:
        text: Input text string to normalize
        
    Returns:
        Normalized text string ready for tokenization
        
    Examples:
        normalize_token("Håkan Hellström") ->
        'håkan hellström'

        normalize_token("Rock'n'Roll") ->
        "rock'n'roll"

        normalize_token("Oops!...I Did It Again") ->
        'oops i did it again'

        normalize_token("Café 1989") ->
        'café 1989'
    """
    if not text:
        return ""
    
    text = text.lower()
    # re.UNICODE flag ensures Unicode-aware matching (default in Python 3 but explicit is safer)
    text = re.sub(r"[^\w']+", " ", text, flags=re.UNICODE)
    
    # Collapse multiple consecutive spaces into single spaces
    # This handles cases where multiple punctuation marks were removed
    # Example: "Track  -  Remix" → "track - remix" → "track remix"
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
