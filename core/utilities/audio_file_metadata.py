# core/utilities/audio_file_metadata.py
"""
Audio Metadata Extraction
=========================
Extract metadata from audio files using mutagen, with fallback to filename parsing.
Supports: MP3, FLAC, M4A/MP4, OGG, WMA, AIFF, WAV, OPUS
"""
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from mutagen import File as MutagenFile
from core.preprocessing.querying.query_tokenizer import normalize_token

logger = logging.getLogger(__name__)

@dataclass
class AudioFileMetadata:
    """Structured metadata extracted from an audio file."""
    file_path: Path
    isrc: Optional[str] = None
    title: Optional[str] = None
    artists: List[str] = field(default_factory=list)
    album: Optional[str] = None
    album_artist: Optional[str] = None
    
    # Source tracking
    metadata_source: str = "unknown"  # "embedded", "filename", "mixed"
    
    @property
    def primary_artist(self) -> Optional[str]:
        """Return the first artist or album artist."""
        if self.artists:
            return self.artists[0]
        return self.album_artist
    
    @property
    def has_embedded_metadata(self) -> bool:
        """Check if any embedded metadata was found."""
        return any([self.title, self.artists, self.album])

class FilenameParser:
    """
    Parse filenames to extract track information.
    Handles common naming conventions:
    - "Artist - Title.ext"
    - "01 - Artist - Title.ext"
    - "Artist_Title.ext"
    """
    
    # Patterns to remove from filenames
    CLEANUP_PATTERNS = [
        r'^\d+\s*[-.]\s*',  # Leading track numbers: "01 - ", "02."
        r'^\d+\s+',          # Leading numbers with space: "01 "
        r'\s*\([^)]*version[^)]*\)$',  # "(Radio Version)", etc.
        r'\s*\[[^\]]*]$',     # "[Remastered]", etc.
    ]
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.stem = file_path.stem
    
    def parse(self) -> Dict[str, Any]:
        """
        Attempt to parse artist and title from filename.
        Returns dict with 'title', 'artists' (list), and confidence.
        """
        cleaned = self._clean_filename(self.stem)
        
        # Try "Artist - Title" patterns with various separators
        separators = [' - ', ' – ', '_', ' -', '- ', '–']
        
        for sep in separators:
            if sep in cleaned:
                parts = cleaned.split(sep, 1)
                if len(parts) == 2:
                    artist_part = parts[0].strip()
                    title_part = parts[1].strip()
                    
                    return {
                        'title': title_part,
                        'artists': [artist_part] if artist_part else [],
                        'confidence': 0.7 if sep in [' - ', ' – '] else 0.5
                    }
        
        # No separator found, treat whole thing as title
        return {
            'title': cleaned,
            'artists': [],
            'confidence': 0.3
        }
    
    def _clean_filename(self, filename: str) -> str:
        """Apply cleanup patterns to filename."""
        cleaned = filename
        
        for pattern in self.CLEANUP_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Replace underscores with spaces
        cleaned = cleaned.replace('_', ' ')
        
        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def get_tokenized_query(self) -> str:
        """Return tokenized version of filename for text search fallback."""
        cleaned = self._clean_filename(self.stem)
        return normalize_token(cleaned)

class AudioFileMetadataExtractor:
    """
    Extract metadata from audio files using mutagen.
    Falls back to filename parsing if no embedded metadata found.
    """
    
    # ISRC tag locations vary by format
    ISRC_TAGS = {
        'MP3': ['TSRC', 'ISRC'],
        'FLAC': ['ISRC'],
        'MP4': ['isrc'],
        'OGG': ['ISRC'],
        'ASF': ['WM/ISRC'],
    }
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.filename_parser = FilenameParser(self.file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")
    
    def extract(self) -> AudioFileMetadata:
        """ Extract metadata from the audio file. 
        Returns AudioFileMetadata with embedded data and filename fallback. """
        metadata = AudioFileMetadata(file_path=self.file_path)
        
        try:
            audio = MutagenFile(self.file_path)
            if audio is None:
                return self._fallback_to_filename(metadata)
            
            # Extract ISRC
            metadata.isrc = self._extract_isrc(audio)
            
            # Extract title
            metadata.title = self._extract_tag(audio, ['TIT2', 'TITLE', '©nam', 'Title'])
            
            # Extract artists
            artists = self._extract_tag(audio, ['TPE1', 'ARTIST', '©ART', 'Author'], allow_list=True)
            if artists:
                metadata.artists = artists if isinstance(artists, list) else [artists]
            
            # Extract album
            metadata.album = self._extract_tag(audio, ['TALB', 'ALBUM', '©alb', 'Album'])
            
            # Extract album artist
            metadata.album_artist = self._extract_tag(audio, ['TPE2', 'ALBUMARTIST', 'aART'])
            
            # Determine source
            if metadata.has_embedded_metadata:
                metadata.metadata_source = "embedded"
            else:
                metadata = self._fallback_to_filename(metadata)
                
            return metadata
            
        except Exception as e:
            # Use debug level to avoid console spam - failures are expected for some formats
            logger.debug(f"Could not read embedded metadata from {self.file_path.name}: {e}")
            return self._fallback_to_filename(metadata)

    def _extract_isrc(self, audio) -> Optional[str]:
        """Extract ISRC code from audio metadata."""
        file_type = type(audio).__name__
        tags_to_check = self.ISRC_TAGS.get(file_type, ['ISRC', 'TSRC'])
        
        for tag in tags_to_check:
            if tag in audio:
                value = audio[tag]
                if isinstance(value, list):
                    value = value[0]
                if hasattr(value, 'text'):  # ID3 tags
                    value = value.text[0] if value.text else None
                
                if value:
                    isrc = str(value).strip().upper().replace('-', '')
                    # Basic validation: 12 chars, starts with 2 letters
                    if len(isrc) == 12 and isrc[:2].isalpha():
                        return isrc
        return None
    
    def _extract_tag(self, audio, possible_tags: List[str], allow_list: bool = False):
        """Extract a tag value from audio metadata."""
        for tag in possible_tags:
            if tag in audio:
                value = audio[tag]
                
                if isinstance(value, list):
                    if not value:
                        continue
                    value = value[0]
                
                if hasattr(value, 'text'):
                    text_values = [str(v) for v in value.text if v]
                    if allow_list:
                        return text_values
                    return text_values[0] if text_values else None
                
                str_value = str(value)
                if allow_list:
                    return [str_value]
                return str_value
        
        return None
    
    def _fallback_to_filename(self, metadata: AudioFileMetadata) -> AudioFileMetadata:
        """Populate metadata from filename parsing."""
        parsed = self.filename_parser.parse()
        
        metadata.title = metadata.title or parsed.get('title')
        metadata.artists = metadata.artists or parsed.get('artists', [])
        metadata.metadata_source = "filename"
        
        return metadata
    
    def get_search_variations(self) -> List[Dict[str, Any]]:
        """
        Generate search query variations based on extracted metadata.
        Returns list of dicts with 'query', 'type', and 'priority' (1-7).
        """
        variations = []
        metadata = self.extract()
        from_filename = metadata.metadata_source == "filename"
        
        # Variation 1: ISRC from metadata (rarely available)
        if metadata.isrc:
            variations.append({
                'query': metadata.isrc,
                'type': 'isrc',
                'priority': 1,
                'description': 'ISRC code'
            })
        
        # Variation 2: song + artist + album from metadata (only if embedded metadata exists)
        if not from_filename and metadata.title and (metadata.artists or metadata.album_artist) and metadata.album:
            artist = metadata.primary_artist
            query = f"{metadata.title} {artist} {metadata.album}"
            variations.append({
                'query': normalize_token(query),
                'type': 'title_artist_album',
                'priority': 2,
                'description': 'Title + Artist + Album'
            })
        
        # Variation 3: song + artist from metadata (only if embedded metadata exists)
        if not from_filename and metadata.title and (metadata.artists or metadata.album_artist):
            artist = metadata.primary_artist
            query = f"{metadata.title} {artist}"
            variations.append({
                'query': normalize_token(query),
                'type': 'title_artist',
                'priority': 3,
                'description': 'Title + Artist'
            })
        
        # Variation 4: song + album from metadata (only if embedded metadata exists)
        if not from_filename and metadata.title and metadata.album:
            query = f"{metadata.title} {metadata.album}"
            variations.append({
                'query': normalize_token(query),
                'type': 'title_album',
                'priority': 4,
                'description': 'Title + Album'
            })
        
        # Variation 5: extensionless tokenized filename stripped of leading number
        tokenized = self.filename_parser.get_tokenized_query()
        if tokenized:
            variations.append({
                'query': tokenized,
                'type': 'filename',
                'priority': 5,
                'description': 'Filename'
            })
        
        # Variation 6: song from metadata (only if embedded metadata exists)
        if not from_filename and metadata.title:
            variations.append({
                'query': normalize_token(metadata.title),
                'type': 'title_only',
                'priority': 6,
                'description': 'Title only'
            })
        
        # Variation 7: first half slice of the filename
        tokenized_filename = self.filename_parser.get_tokenized_query()
        if tokenized_filename:
            words = tokenized_filename.split()
            if len(words) > 2:
                half_point = len(words) // 2
                first_half = ' '.join(words[:half_point])
                variations.append({
                    'query': first_half,
                    'type': 'filename_half',
                    'priority': 7,
                    'description': 'First half of filename'
                })
        
        return variations

def extract_audio_metadata(file_path: Union[str, Path]) -> AudioFileMetadata:
    """
    Convenience function to extract metadata from an audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        AudioFileMetadata object with extracted information
    """
    extractor = AudioFileMetadataExtractor(file_path)
    return extractor.extract()
