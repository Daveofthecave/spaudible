# core/preprocessing/querying/query_tokenizer.py
"""
Query Tokenizer
===============
Deterministic tokenization for semantic search.
Used by build_query_index.py to build inverted_index.bin
and marisa_trie.bin. Also used by text_search_utils.py
to generate tokens for querying the inverted index.
"""
import re
import unidecode
from typing import List

def normalize_token(text: str) -> str:
    """
    Normalize text for tokenization with ASCII folding, acronym handling, etc.
    
    This function prepares text for tokenization by:
    - Converting to lowercase
    - Folding Unicode to ASCII (eg. "Café" → "cafe")
    - Replacing $ with s (eg. "Ke$ha" → "kesha")
    - Replacing '_' with ' ' (eg. "Song_Name" → "song name")
    - Removing dots from acronyms (eg. "R.E.M." → "REM")
    - Stripping decorative symbols (eg. "P!nk" → "pnk")
    - Preserving apostrophes within words (eg. "rock'n'roll")
    - Replacing most punctuation/special characters with spaces
    - Collapsing multiple spaces into single spaces
    
    Args:
        text: Input text string to normalize
    
    Returns:
        Normalized text string ready for tokenization
    """
    if not text:
        return ""
    
    # Convert string to lowercase
    text = text.lower()
    # Fold Unicode to ASCII
    text = unidecode.unidecode(text)
    # Replace $ with s
    text = text.replace('$', 's')
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Remove dots from acronyms
    text = re.sub(r'(?<=\w)\.(?=\w)', '', text)
    # Remove decorative symbols
    text = re.sub(r'[!@#%?*\^]', '', text)
    # Replace remaining punctuation with spaces
    text = re.sub(r"[^\w']+", " ", text, flags=re.UNICODE)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def tokenize(text: str, field_prefix: str = "") -> List[str]:
    """
    Tokenize text into unigrams only.
    
    Args:
        text: Input text (may include track, artist, and/or album)
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
    Only tokenizes the first artist to prevent explosion on multi-artist tracks;
    eg. "Coldplay, Rihanna" → "Coldplay"
    """
    if not name:
        return []
    
    # Take first artist only
    first_artist = name.split(',')[0].strip()
    return tokenize(first_artist, field_prefix="a_")

def tokenize_album_name(name: str) -> List[str]:
    """Tokenize album name with 'al_' prefix for weighting"""
    return tokenize(name, field_prefix="al_")
