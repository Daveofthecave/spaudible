# config.py
import os
import tomllib
from pathlib import Path

def _get_version():
    """Read Spaudible's version from pyproject.toml"""
    try:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        return "unknown"  # Fallback if pyproject.toml is missing

VERSION = _get_version()
AUTO_OPTIMIZE_CHUNK_SIZE = True
VRAM_SAFETY_FACTOR = 0.85 # What percentage of available VRAM to use
VRAM_SCALING_FACTOR_MB = 2**8
FORCE_CPU_MODE = False
REGION_FILTER_STRENGTH = 1.0 # 1 = stick to the same region; 0 = any region is fine
EXPECTED_VECTORS = 256_039_007 # How many tracks are in the database
FRAME_WIDTH = 70 # For CLI UI headings

# Constants pertaining to the file structure of data/vectors/track_vectors.bin
VECTOR_RECORD_SIZE = 104        # Total bytes per vector record
VECTOR_HEADER_SIZE = 16         # Header size at start of file
ISRC_OFFSET_IN_RECORD = 70      # ISRC starts at byte 70
TRACK_ID_OFFSET_IN_RECORD = 82  # Track ID starts at byte 82

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
    def get_query_index_dir(cls):
        """Directory for query index files"""
        return cls.VECTORS / "query_index"

    @classmethod
    def get_query_marisa_file(cls):
        """MARISA trie file for token lookup"""
        return cls.get_query_index_dir() / "marisa_trie.bin"

    @classmethod
    def get_query_postings_file(cls):
        """Inverted index postings file"""
        return cls.get_query_index_dir() / "inverted_index.bin"

    @classmethod
    def get_all_required_files(cls):
        """Return all required files for setup completion"""
        return [
            cls.get_main_db(),
            cls.get_audio_db(),
            cls.get_vector_file(),
            cls.get_index_file(),
            cls.get_query_marisa_file(),
            cls.get_query_postings_file()
        ]

    @classmethod
    def get_config_path(cls):
        return cls.BASE_DIR / "config.json"

class DownloadConfig:
    """Configuration for HuggingFace downloads and file specifications."""
    
    # Repository IDs
    REPO_DB = "Daveofthecave/spaudible_db"
    REPO_VECTORS = "Daveofthecave/spaudible_vectors"
    
    # Database files: (filename, compressed_size_gb, extracted_size_gb_approx)
    DATABASE_FILES = [
        ("spotify_clean.sqlite3.zst", 36.7, 125.4),
        ("spotify_clean_audio_features.sqlite3.zst", 17.7, 41.5)
    ]
    
    # Vector files: (filename, subdir, size_gb)
    VECTOR_FILES = [
        ("track_vectors.bin", None, 26.6),
        ("track_index.bin", None, 6.7),
        ("inverted_index.bin", "query_index", 4.7),
        ("marisa_trie.bin", "query_index", 0.148)
    ]
    
    @classmethod
    def get_download_state_file(cls):
        """Path to download state JSON for resume tracking."""
        return PathConfig.BASE_DIR / "data" / "download_state.json"
    
    @classmethod
    def get_required_space_gb(cls, include_databases: bool = True, include_vectors: bool = True) -> float:
        """Calculate total required download space in GB."""
        total = 0.0
        if include_databases:
            total += sum(compressed for _, compressed, _ in cls.DATABASE_FILES)
        if include_vectors:
            total += sum(size for _, _, size in cls.VECTOR_FILES)
        return total
    
    @classmethod
    def get_total_extracted_space_gb(cls) -> float:
        """Calculate max disk space needed at peak (compressed + extracted)."""
        db_compressed = sum(compressed for _, compressed, _ in cls.DATABASE_FILES)
        db_extracted = sum(extracted for _, _, extracted in cls.DATABASE_FILES)
        vectors = sum(size for _, _, size in cls.VECTOR_FILES)
        # Peak usage: compressed DBs + extracted DBs + vectors
        return db_compressed + db_extracted + vectors
