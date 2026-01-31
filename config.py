# config.py
VERSION = "0.2.3"
AUTO_OPTIMIZE_CHUNK_SIZE = True
VRAM_SAFETY_FACTOR = 0.85 # What percentage of available VRAM to use
VRAM_SCALING_FACTOR_MB = 2**8
FORCE_CPU_MODE = False
REGION_FILTER_STRENGTH = 1.0 # 1 = stick to the same region; 0 = any region is fine
EXPECTED_VECTORS = 256_039_007 # How many tracks are in the database

# Constants pertaining to the file structure of data/vectors/track_vectors.bin
VECTOR_RECORD_SIZE = 104        # Total bytes per vector record
VECTOR_HEADER_SIZE = 16         # Header size at start of file
ISRC_OFFSET_IN_RECORD = 70      # ISRC starts at byte 70
TRACK_ID_OFFSET_IN_RECORD = 82  # Track ID starts at byte 82

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

    @classmethod
    def get_config_path(cls):
        return cls.BASE_DIR / "config.json"
