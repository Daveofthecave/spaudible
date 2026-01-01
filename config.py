# config.py
VERSION = "0.1.2"
AUTO_OPTIMIZE_CHUNK_SIZE = True

import os
from pathlib import Path

class PathConfig:
    BASE_DIR = Path(__file__).parent
    DATABASES = BASE_DIR / "data/databases"
    VECTORS = BASE_DIR / "data/vectors"
    GENRE_MAPPING = BASE_DIR / "data/genre_intensity_mapping.csv"
    
    @classmethod
    def get_vector_file(cls):
        return cls.VECTORS / "track_vectors.bin"
    
    @classmethod
    def get_index_file(cls):
        return cls.VECTORS / "track_index.bin"

    @classmethod
    def get_metadata_file(cls):
        return cls.VECTORS / "metadata.json"        
    
    @classmethod
    def get_main_db(cls):
        return cls.DATABASES / "spotify_clean.sqlite3"
    
    @classmethod
    def get_audio_db(cls):
        return cls.DATABASES / "spotify_clean_audio_features.sqlite3"

    @classmethod
    def get_genre_mapping(cls):
        return cls.GENRE_MAPPING

    @classmethod
    def all_required_files(cls):
        """Return all required files for setup completion"""
        return [
            cls.get_main_db(),
            cls.get_audio_db(),
            cls.get_vector_file(),
            cls.get_index_file(),
            cls.get_metadata_file()
        ]

    def get_gpu_config():
        return {
            "enabled": True,
            "half_precision": False,
            "max_batch_size": 5_000_000  # ~2GB for FP32
        }       
