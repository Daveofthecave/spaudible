# core/utilities/audio_file_input_processor.py
"""
Audio File Input Processing
============================
Processes audio file inputs for Spaudible, generating search variations 
and managing resolution state.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Union
from core.utilities.audio_file_metadata import (
    AudioFileMetadata,
    AudioFileMetadataExtractor,
    extract_audio_metadata,
)
from core.preprocessing.querying.query_tokenizer import normalize_token

logger = logging.getLogger(__name__)

@dataclass
class SearchVariation:
    """Represents a single search strategy for finding a track."""
    query: str
    variation_type: str  # 'isrc', 'title_artist_album', 'filename', etc.
    priority: int  # 1-7, lower is tried first
    description: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False  # Did this variation yield results?

    @property
    def has_results(self) -> bool:
        """Check if this variation produced any results."""
        return len(self.results) > 0

@dataclass
class AudioFileInput:
    """
    Represents an audio file input with extracted metadata and search variations.
    This is the primary data structure for audio file processing.
    """
    file_path: Path
    metadata: Optional[AudioFileMetadata] = None
    variations: List[SearchVariation] = field(default_factory=list)
    _current_variation_idx: int = field(default=0, repr=False)

    def __post_init__(self):
        """Extract metadata and generate variations if not provided."""
        if self.metadata is None:
            self.metadata = extract_audio_metadata(self.file_path)
        if not self.variations:
            self._generate_variations()

    def _generate_variations(self):
        """Generate all search variations from metadata."""
        extractor = AudioFileMetadataExtractor(self.file_path)
        raw_variations = extractor.get_search_variations()
        
        # Convert to SearchVariation objects
        self.variations = [
            SearchVariation(
                query=v['query'],
                variation_type=v['type'],
                priority=v['priority'],
                description=v['description']
            )
            for v in sorted(raw_variations, key=lambda x: x['priority'])
        ]
        logger.debug(f"Generated {len(self.variations)} variations for {self.file_path.name}")

    @property
    def filename(self) -> str:
        """Get the filename for display."""
        return self.file_path.name

    @property
    def stem(self) -> str:
        """Get the filename without extension."""
        return self.file_path.stem

    @property
    def has_isrc(self) -> bool:
        """Check if ISRC is available."""
        return self.metadata is not None and self.metadata.isrc is not None

    @property
    def is_resolved(self) -> bool:
        """Check if any variation has succeeded."""
        return any(v.success for v in self.variations)

    @property
    def best_match(self) -> Optional[SearchVariation]:
        """Get the first successful variation."""
        for v in self.variations:
            if v.success:
                return v
        return None

    def get_next_variation(self) -> Optional[SearchVariation]:
        """
        Get the next variation to try (lazy evaluation).
        Returns None if all variations exhausted.
        """
        while self._current_variation_idx < len(self.variations):
            variation = self.variations[self._current_variation_idx]
            self._current_variation_idx += 1
            return variation
        return None

    def reset_variation_iterator(self):
        """Reset to try variations from the beginning."""
        self._current_variation_idx = 0

    def mark_variation_success(self, variation_type: str, results: List[Dict[str, Any]]):
        """Mark a variation as successful with results."""
        for v in self.variations:
            if v.variation_type == variation_type:
                v.results = results
                v.success = True
                logger.debug(f"Variation '{variation_type}' succeeded with {len(results)} results")
                return

    def get_variation_by_type(self, variation_type: str) -> Optional[SearchVariation]:
        """Get a specific variation by type."""
        for v in self.variations:
            if v.variation_type == variation_type:
                return v
        return None

    def get_all_attempted_variations(self) -> List[SearchVariation]:
        """Get list of variations that have been attempted (up to current index)."""
        return self.variations[:self._current_variation_idx]

    def get_display_summary(self) -> str:
        """Get a short summary for UI display."""
        if self.is_resolved and self.best_match:
            top_result = self.best_match.results[0] if self.best_match.results else None
            if top_result:
                artist = top_result.get('artist_name', 'Unknown')
                title = top_result.get('track_name', 'Unknown')
                return f"{title} - {artist}"
        
        # Fallback to metadata
        if self.metadata and self.metadata.title:
            artist = self.metadata.primary_artist or "Unknown"
            return f"{self.metadata.title} - {artist} (unmatched)"
        
        return f"{self.stem} (unmatched)"

@dataclass
class ResolvedAudioFile:
    """
    Result of resolving an audio file to a database track.
    Contains the resolved track ID and resolution metadata.
    """
    audio_file_input: AudioFileInput
    track_id: Optional[str] = None
    matched_variation: Optional[SearchVariation] = None
    confidence: float = 0.0
    is_confirmed: bool = False  # User confirmed this match
    user_override: bool = False  # User manually selected different track
    
    # Store alternative matches for refinement UI
    alternative_matches: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        """Check if successfully resolved to a track."""
        return self.track_id is not None

    @property
    def resolution_method(self) -> str:
        """How this track was resolved."""
        if self.user_override:
            return "manual"
        if self.matched_variation:
            return self.matched_variation.variation_type
        return "unresolved"

    def get_display_text(self) -> str:
        """Get formatted display text for the match."""
        if not self.is_resolved:
            return f"❌ {self.audio_file_input.filename} (no match)"
        
        if self.matched_variation and self.matched_variation.results:
            top = self.matched_variation.results[0]
            artist = top.get('artist_name', 'Unknown')
            title = top.get('track_name', 'Unknown')
            album = top.get('album_name', '')
            year = top.get('album_release_year', '')
            
            parts = [f"{title} - {artist}"]
            if album:
                parts.append(f" - {album}")
            if year:
                parts.append(f" ({year})")
            
            status = "✓" if self.is_confirmed else "→"
            return f"{status} {''.join(parts)}"
        
        return f"? {self.track_id}"

    @property
    def best_match(self) -> Optional[SearchVariation]:
        """Get the matched variation (convenience alias)."""
        return self.matched_variation
    

class VariationGenerator:
    """
    Generates and manages search variations for audio files.
    Handles deduplication and normalization.
    """
    
    def __init__(self):
        self.seen_queries: Set[str] = set()

    def generate_for_file(self, file_path: Union[str, Path]) -> AudioFileInput:
        """
        Generate AudioFileInput with variations for a file.
        Automatically deduplicates variations.
        """
        audio_file_input = AudioFileInput(file_path=Path(file_path))
        
        # Remove duplicate queries (keep first occurrence by priority)
        unique_variations = []
        for v in audio_file_input.variations:
            normalized = self._normalize_for_dedup(v.query)
            if normalized not in self.seen_queries:
                self.seen_queries.add(normalized)
                unique_variations.append(v)
            else:
                logger.debug(f"Deduplicated variation: {v.description}")
        
        audio_file_input.variations = unique_variations
        return audio_file_input

    def _normalize_for_dedup(self, query: str) -> str:
        """Normalize query string for deduplication comparison."""
        return normalize_token(query).strip().lower()

    def reset(self):
        """Clear deduplication cache."""
        self.seen_queries.clear()

class AudioFileInputResolver:
    """
    Resolves audio files to track IDs using lazy variation evaluation.
    Stops at the first variation that yields results.
    """
    
    def __init__(self, search_func=None):
        """
        Initialize resolver.
        
        Args:
            search_func: Function to call for text search. Should accept 
                        (query, limit) and return list of results.
                        If None, variations are generated but not searched.
        """
        self.search_func = search_func
        self._cache: Dict[str, ResolvedAudioFile] = {}

    def resolve(self, file_path: Union[str, Path], max_results_per_variation: int = 3) -> ResolvedAudioFile:
        """
        Resolve an audio file to a track ID using lazy variation evaluation.
        
        Args:
            file_path: Path to audio file
            max_results_per_variation: How many results to fetch per variation
            
        Returns:
            ResolvedAudioFile with track_id (if found) and resolution details
        """
        path = Path(file_path)
        cache_key = str(path.resolve())
        
        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {path.name}")
            return self._cache[cache_key]

        # Generate variations
        audio_file_input = AudioFileInput(file_path=path)
        resolved = ResolvedAudioFile(audio_file_input=audio_file_input)
        
        if not self.search_func:
            logger.warning("No search function provided, returning unresolved")
            return resolved

        # Lazy evaluation: try variations in order until one succeeds
        logger.info(f"Resolving {path.name}...")
        for variation in audio_file_input.variations:
            logger.debug(f"Trying variation {variation.priority}: {variation.description}")
            
            try:
                results = self.search_func(variation.query, limit=max_results_per_variation)
                
                if results:
                    # Success!
                    audio_file_input.mark_variation_success(variation.variation_type, results)
                    resolved.track_id = results[0].get('track_id') if results else None
                    resolved.matched_variation = variation
                    resolved.confidence = self._calculate_confidence(variation, results)
                    
                    # Store alternatives (remaining results)
                    resolved.alternative_matches = results[1:] if len(results) > 1 else []
                    
                    logger.info(f"Resolved {path.name} using {variation.description}: {resolved.track_id}")
                    break
                    
            except Exception as e:
                logger.error(f"Error searching with variation {variation.description}: {e}")
                continue

        # Cache result
        self._cache[cache_key] = resolved
        return resolved

    def resolve_batch(
        self, 
        file_paths: List[Union[str, Path]], 
        progress_callback=None
    ) -> List[ResolvedAudioFile]:
        """
        Resolve multiple files with optional progress callback.
        
        Args:
            file_paths: List of paths to process
            progress_callback: Callable(current, total, resolved_count) -> bool
                             Return False to cancel processing.
                             
        Returns:
            List of ResolvedAudioFile in same order as input
        """
        results = []
        total = len(file_paths)
        resolved_count = 0
        
        for i, path in enumerate(file_paths):
            resolved = self.resolve(path)
            results.append(resolved)
            
            if resolved.is_resolved:
                resolved_count += 1
            
            if progress_callback:
                should_continue = progress_callback(i + 1, total, resolved_count)
                if not should_continue:
                    logger.info("Batch processing cancelled by user")
                    break
        
        return results

    def _calculate_confidence(self, variation: SearchVariation, results: List[Dict]) -> float:
        """
        Calculate confidence score for a match.
        Higher priority variations (lower number) get higher confidence.
        """
        base_confidence = 1.0 / variation.priority  # 1.0, 0.5, 0.33, etc.
        
        # Boost if top result has high popularity
        if results and 'popularity' in results[0]:
            pop = results[0]['popularity']
            if pop > 80:
                base_confidence *= 1.2
        
        # Boost for exact title match
        if results and 'track_name' in results[0]:
            # Could implement fuzzy matching here
            pass
        
        return min(base_confidence, 1.0)

    def clear_cache(self):
        """Clear resolution cache."""
        self._cache.clear()

# Convenience functions

def create_audio_file_input(file_path: Union[str, Path]) -> AudioFileInput:
    """Create AudioFileInput for a file with all variations generated."""
    return AudioFileInput(file_path=Path(file_path))

def get_variations_for_file(file_path: Union[str, Path]) -> List[SearchVariation]:
    """Get all search variations for an audio file."""
    audio_file_input = create_audio_file_input(file_path)
    return audio_file_input.variations
