# core/vectorization/__init__.py
"""
Vectorization Package
"""
from .track_vectorizer import build_track_vector
from .genre_mapper import load_genre_mapping, compute_genre_intensities
from .canonical_track_resolver import (
    TrackResolver,
    CanonicalVectorBuilder,
    get_resolver,
    get_builder,
    resolve_track,
    build_canonical_vector
)

__all__ = [
    'build_track_vector',
    'load_genre_mapping',
    'compute_genre_intensities',
    'TrackResolver',
    'CanonicalVectorBuilder',
    'get_resolver',
    'get_builder',
    'resolve_track',
    'build_canonical_vector'
]
