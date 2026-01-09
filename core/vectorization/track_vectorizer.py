# core/vectorization/track_vectorizer.py
import math
import numpy as np
from .genre_mapper import compute_genre_intensities_batch
from typing import List, Tuple

# Constants for vectorization
MIN_TEMPO = 40.0
MAX_TEMPO = 250.0
MIN_YEAR = 1900
MAX_YEAR = 2025
MAX_FOLLOWERS = 141_174_367

def safe_float(value, default=-1.0):
    """Safely convert a value to float, returning default if conversion fails."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def normalize_loudness(loudness_db):
    """Linearly normalize loudness from [-60, 0] dB to [0, 1]."""
    if isinstance(loudness_db, np.ndarray):
        clipped = np.clip(loudness_db, -60.0, 0.0)
        return (clipped + 60.0) / 60.0
    if loudness_db is None:
        return -1.0
    return (max(min(loudness_db, 0.0), -60.0) + 60.0) / 60.0

def normalize_key(key_number, mode):
    """Linearly normalize key to [0, 1] based on the number of accidentals (0-6)."""
    if isinstance(key_number, np.ndarray) and isinstance(mode, np.ndarray):
        # Create mask for valid keys (0-11)
        valid_mask = (key_number >= 0) & (key_number <= 11)
        
        # Initialize result with -1.0 (invalid)
        result = np.full(key_number.shape, -1.0, dtype=np.float32)
        
        # Process valid keys
        if np.any(valid_mask):
            # If mode is minor, relative major is 3 semitones higher
            adjusted_key = np.where(mode[valid_mask] == 0, 
                                  (key_number[valid_mask] + 3) % 12, 
                                  key_number[valid_mask])
            
            # Map to accidentals count
            accidentals_map = np.array([0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5], dtype=np.float32)
            result[valid_mask] = accidentals_map[adjusted_key.astype(int)] / 6.0
        
        return result
    
    # Scalar version
    if key_number is None or mode is None:
        return -1.0
    if key_number < 0 or key_number > 11:
        return -1.0
    
    adjusted_key = (key_number + (3 if mode == 0 else 0)) % 12
    accidentals_map = [0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5]
    return accidentals_map[adjusted_key] / 6.0

def normalize_tempo(tempo_bpm):
    """Normalize tempo from [40, 250] to [0, 1] using log2 scaling."""
    if isinstance(tempo_bpm, np.ndarray):
        clipped = np.clip(tempo_bpm, MIN_TEMPO, MAX_TEMPO)
        log_min = np.log2(MIN_TEMPO)
        log_max = np.log2(MAX_TEMPO)
        return (np.log2(clipped) - log_min) / (log_max - log_min)
    if tempo_bpm is None:
        return -1.0
    tempo = max(min(tempo_bpm, MAX_TEMPO), MIN_TEMPO)
    return (np.log2(tempo) - np.log2(MIN_TEMPO)) / (np.log2(MAX_TEMPO) - np.log2(MIN_TEMPO))

def normalize_time_signature(time_sig):
    """One-hot encode time signature into a 4D binary vector."""
    if isinstance(time_sig, np.ndarray):
        result = np.zeros((len(time_sig), 4), dtype=np.float32)
        result[time_sig == 4, 0] = 1.0
        result[time_sig == 3, 1] = 1.0
        result[time_sig == 5, 2] = 1.0
        result[~np.isin(time_sig, [3, 4, 5]), 3] = 1.0
        return result
    if time_sig is None:
        return [0.0, 0.0, 0.0, 0.0]
    if time_sig == 4:
        return [1.0, 0.0, 0.0, 0.0]
    elif time_sig == 3:
        return [0.0, 1.0, 0.0, 0.0]
    elif time_sig == 5:
        return [0.0, 0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 0.0, 1.0]

def normalize_duration(duration_ms, c=5.0):
    """Normalize duration to [0, 1) using a rational function."""
    if isinstance(duration_ms, np.ndarray):
        minutes = np.abs(duration_ms) / 60000.0
        return minutes / (minutes + np.abs(c))
    if duration_ms is None:
        return -1.0
    minutes = abs(duration_ms) / 60000.0
    return minutes / (minutes + abs(c))

def normalize_release_date(release_date_str):
    """Linearly normalize release year to [0, 1]."""
    if isinstance(release_date_str, np.ndarray):
        years = np.full(len(release_date_str), -1.0, dtype=np.float32)
        
        # Create mask for valid release dates
        valid_mask = np.array([s is not None and len(s) >= 4 for s in release_date_str], dtype=bool)
        
        # Extract year from string for valid entries
        year_strs = np.array([s[:4] for s in release_date_str[valid_mask]])
        
        # Convert to integers, handling invalid values
        valid_years = []
        for y in year_strs:
            try:
                valid_years.append(int(y))
            except ValueError:
                valid_years.append(-1)
        valid_years = np.array(valid_years)
        
        # Clip and normalize
        clipped = np.clip(valid_years, MIN_YEAR, MAX_YEAR)
        years[valid_mask] = (clipped - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
        return years
    
    if not release_date_str:
        return -1.0
    try:
        year_str = release_date_str[:4]
        year = int(year_str)
        year = max(min(year, MAX_YEAR), MIN_YEAR)
        return (year - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
    except (ValueError, IndexError):
        return -1.0

def normalize_popularity(popularity):
    """Normalize track popularity to [0, 1] using square root scaling."""
    if isinstance(popularity, np.ndarray):
        clipped = np.clip(popularity, 0, 100)
        return np.sqrt(clipped / 100.0)
    if popularity is None:
        return -1.0
    popularity = max(min(popularity, 100), 0)
    return math.sqrt(popularity / 100.0)

def normalize_followers(followers):
    """Normalize artist followers to [0, 1] using log10 scaling."""
    if isinstance(followers, np.ndarray):
        clipped = np.clip(followers, 0, MAX_FOLLOWERS)
        return np.log10(clipped + 1) / np.log10(MAX_FOLLOWERS + 1)
    if followers is None or followers < 0:
        return -1.0
    clipped = min(followers, MAX_FOLLOWERS)
    return math.log10(clipped + 1) / math.log10(MAX_FOLLOWERS + 1)

def validate_vector(vector: List[float]) -> Tuple[bool, str]:
    """Comprehensive vector validation with detailed error messages"""
    if len(vector) != 32:
        return False, f"Vector length {len(vector)} != 32"
    
    for i, val in enumerate(vector):
        # Check if value is a number (including numpy floats)
        if not isinstance(val, (int, float, np.floating)):
            return False, f"Dimension {i+1}: Value {val} is not a number"
        
        # Check range
        if val < -1.0 or val > 1.0:
            return False, f"Dimension {i+1}: Value {val} out of [-1,1] range"
    
    return True, "Valid"

def build_track_vector(track_dict):
    """Convert track dictionary to 32-dimensional vector."""
    return build_track_vectors_batch([track_dict])[0]

def build_track_vectors_batch(track_dicts: List[dict]) -> List[List[float]]:
    """Batch-convert a list of track dictionaries to a list of 32-dimensional vectors."""
    n = len(track_dicts)
    if n == 0:
        return []
    
    # Create structured array for efficient access
    dtype = [
        ('acousticness', 'f4'), ('instrumentalness', 'f4'), ('speechiness', 'f4'),
        ('valence', 'f4'), ('danceability', 'f4'), ('energy', 'f4'), ('liveness', 'f4'),
        ('loudness', 'f4'), ('key', 'f4'), ('mode', 'f4'), ('tempo', 'f4'),
        ('time_signature', 'f4'), ('duration_ms', 'f4'), ('release_date', 'U10'),
        ('popularity', 'f4'), ('max_followers', 'f4'), ('genres', 'O')
    ]
    
    # Create array from track data
    data = np.zeros(n, dtype=dtype)
    for i, track in enumerate(track_dicts):
        for field in dtype:
            name = field[0]
            if name in track:
                data[i][name] = track[name]
    
    # Preallocate result vectors
    vectors = np.full((n, 32), -1.0, dtype=np.float32)
    
    # Dim 1-7: Direct features
    feature_names = ['acousticness', 'instrumentalness', 'speechiness', 'valence', 
                     'danceability', 'energy', 'liveness']
    for j, name in enumerate(feature_names):
        vectors[:, j] = data[name]
    
    # Dim 8: Loudness
    vectors[:, 7] = normalize_loudness(data['loudness'])
    
    # Dim 9: Key (with validation)
    vectors[:, 8] = normalize_key(data['key'], data['mode'])
    
    # Dim 10: Mode
    vectors[:, 9] = data['mode']
    
    # Dim 11: Tempo
    vectors[:, 10] = normalize_tempo(data['tempo'])
    
    # Dim 12-15: Time signature
    time_vecs = normalize_time_signature(data['time_signature'])
    vectors[:, 11:15] = time_vecs
    
    # Dim 16: Duration
    vectors[:, 15] = normalize_duration(data['duration_ms'])
    
    # Dim 17: Release year
    vectors[:, 16] = normalize_release_date(data['release_date'])
    
    # Dim 18: Artist popularity
    vectors[:, 17] = normalize_popularity(data['popularity'])
    
    # Dim 19: Artist followers
    vectors[:, 18] = normalize_followers(data['max_followers'])
    
    # Dim 20-32: Genre intensities
    genre_lists = [list(g) for g in data['genres']]
    genre_intensities = compute_genre_intensities_batch(genre_lists)
    vectors[:, 19:32] = genre_intensities
    
    # Validate vectors and log errors
    errors = []
    for i in range(n):
        # Convert numpy array to Python list before validation
        vector_list = vectors[i].tolist()
        valid, message = validate_vector(vector_list)
        if not valid:
            track_id = track_dicts[i].get('track_id', f'index {i}')
            error_msg = f"Invalid vector for track {track_id}: {message}"
            errors.append(error_msg)
    
    if errors:
        # Write errors to log file
        with open("vector_errors.log", "a") as f:
            for error in errors:
                f.write(error + "\n")
        
        # Raise exception with first error
        raise ValueError(errors[0])
    
    return vectors.tolist()
