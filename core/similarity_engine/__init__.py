# core/similarity_engine/__init__.py
"""
Similarity Engine Package
"""
from .vector_comparer import ChunkedSearch
from .vector_io import VectorReader
from .vector_math import VectorOps
from .metadata_service import MetadataManager
from .index_manager import IndexManager
from .weight_layers import WeightLayers
from .chunk_size_optimizer import ChunkSizeOptimizer

__all__ = [
    'ChunkedSearch',
    'VectorReader',
    'VectorOps',
    'MetadataManager',
    'IndexManager',
    'WeightLayers',
    'ChunkSizeOptimizer'
]
