# core/similarity_engine/orchestrator.py
import time
import numpy as np
from config import PathConfig
from core.vectorization.canonical_track_resolver import build_canonical_vector
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable
from .vector_io import VectorReader
from .vector_math import VectorOps
from .vector_comparer import ChunkedSearch
from .metadata_service import MetadataManager
from .index_manager import IndexManager

class SearchOrchestrator:
    """High-level coordinator for similarity search operations."""
    
    def __init__(self,
                 vectors_path: Optional[str] = None,
                 index_path: Optional[str] = None,
                 metadata_db: Optional[str] = None,
                 chunk_size: int = 20_000):
        """
        Initialize the search orchestrator.
        
        Args:
            vectors_path: Path to track_vectors.bin (default: from PathConfig)
            index_path: Path to track_index.bin (default: from PathConfig)
            metadata_db: Path to metadata database
            chunk_size: Chunk size for vector processing
        """
        # Set default paths if not provided
        vectors_path = vectors_path or str(PathConfig.get_vector_file())
        index_path = index_path or str(PathConfig.get_index_file())
        
        # Initialize components
        self.vector_reader = VectorReader(vectors_path)
        self.vector_ops = VectorOps()
        self.index_manager = IndexManager(index_path)
        self.metadata_manager = MetadataManager(metadata_db)
        self.chunked_search = ChunkedSearch(chunk_size)
        
        self.total_vectors = self.vector_reader.get_total_vectors()
    
    def _vector_source(self, start_idx: int, num_vectors: int) -> np.ndarray:
        """Wrapper for vector reader."""
        return self.vector_reader.read_chunk(start_idx, num_vectors)
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        search_mode: str = "sequential",
        top_k: int = 10,
        with_metadata: bool = True,
        **kwargs
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Find similar tracks to a query vector.
        
        Args:
            query_vector: 32D vector to search for
            search_mode: One of "sequential", "random", "progressive"
            top_k: Number of results to return
            with_metadata: Whether to include metadata
            **kwargs: Additional search parameters
            
        Returns:
            List of (track_id, similarity, metadata) tuples
        """
        # Convert query to numpy array
        if isinstance(query_vector, list):
            query_np = self.vector_ops.to_numpy_array(query_vector)
        else:
            query_np = query_vector
        
        # Execute search algorithm
        if search_mode == "sequential":
            indices, similarities = self.chunked_search.sequential_scan(
                query_np,
                self._vector_source,
                self.total_vectors,
                self.vector_ops,
                top_k=top_k,
                **kwargs
            )
        elif search_mode == "random":
            indices, similarities = self.chunked_search.random_chunk_search(
                query_np,
                self._vector_source,
                self.total_vectors,
                self.vector_ops,
                top_k=top_k,
                **kwargs
            )
        elif search_mode == "progressive":
            indices, similarities = self.chunked_search.progressive_search(
                query_np,
                self._vector_source,
                self.total_vectors,
                self.vector_ops,
                top_k=top_k,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
        
        # Convert indices to track IDs
        track_ids = self.index_manager.get_track_ids_batch(indices)
        
        # Get metadata if requested
        if with_metadata:
            metadata_list = self.metadata_manager.get_track_metadata_batch(track_ids)
            results = list(zip(track_ids, similarities, metadata_list))
        else:
            results = list(zip(track_ids, similarities))
        
        return results
    
    def find_similar_to_track(
        self,
        track_id: str,
        top_k: int = 10,
        search_mode: str = "sequential",
        with_metadata: bool = True,
        **kwargs
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Find tracks similar to a given track ID.
        
        Args:
            track_id: Spotify track ID
            top_k: Number of results to return
            search_mode: Search algorithm to use
            with_metadata: Whether to include metadata
            **kwargs: Additional search parameters
            
        Returns:
            List of (track_id, similarity, metadata) tuples
        """
        # Build vector for the track
        vector, _ = build_canonical_vector(track_id)
        
        if vector is None:
            raise ValueError(f"Could not build vector for track: {track_id}")
        
        return self.search(
            vector,
            search_mode=search_mode,
            top_k=top_k,
            with_metadata=with_metadata,
            **kwargs
        )
    
    def close(self):
        """Clean up resources."""
        self.metadata_manager.close()

def find_similar_tracks(
    track_id: str,
    top_k: int = 10,
    search_mode: str = "sequential",
    **kwargs
) -> List:
    """
    Simplified API for finding similar tracks.
    
    Args:
        track_id: Spotify track ID
        top_k: Number of results
        search_mode: Search algorithm
        **kwargs: Additional parameters
        
    Returns:
        List of results
    """
    orchestrator = SearchOrchestrator()
    results = orchestrator.find_similar_to_track(
        track_id,
        top_k=top_k,
        search_mode=search_mode,
        **kwargs
    )
    orchestrator.close()
    return results
