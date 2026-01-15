# core/vectorization/track_vectorizer.py
import numpy as np
import csv
import os
from config import PathConfig
from typing import List
from numba import njit, prange
import math

# Preload genre mapping at module level
GENRE_MAPPING = {}
GENRE_ID_MAP = {}
GENRE_INTENSITY_MAP = {}

def _preload_genre_data():
    """Preload genre mapping data once at module import"""
    csv_path = PathConfig.get_genre_mapping()
    if not csv_path.exists():
        return
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 4:
                continue
            try:
                meta_genre = int(row[1])
                genre = row[2].strip().lower()
                intensity = float(row[3])
                
                # Store mappings
                GENRE_MAPPING[genre] = (meta_genre, intensity)
                GENRE_ID_MAP[genre] = len(GENRE_ID_MAP)
                GENRE_INTENSITY_MAP[genre] = intensity
            except (ValueError, IndexError):
                continue

# Load genre data on import
_preload_genre_data()

# Constants for vectorization
MIN_TEMPO = 40.0
MAX_TEMPO = 250.0
MIN_YEAR = 1900
MAX_YEAR = 2025
MAX_FOLLOWERS = 141_174_367

@njit(fastmath=True)
def safe_float(value, default=-1.0):
    """Safely convert value to float with numba acceleration."""
    if value == -1.0:
        return default
    return value

@njit(fastmath=True)
def normalize_loudness_scalar(loudness_db):
    """Linearly normalize a single loudness value from [-60, 0] dB to [0, 1]."""
    if loudness_db == -1.0:
        return -1.0
    clipped = max(min(loudness_db, 0.0), -60.0)
    return (clipped + 60.0) / 60.0

@njit(fastmath=True)
def normalize_loudness_array(loudness_db):
    """Linear normalization of loudness array from [-60, 0] dB to [0, 1]."""
    result = np.empty_like(loudness_db)
    for i in range(len(loudness_db)):
        val = loudness_db[i]
        if val == -1.0:
            result[i] = -1.0
        else:
            result[i] = max(min(val, 0.0), -60.0)
            result[i] = (result[i] + 60.0) / 60.0
    return result

def normalize_loudness(loudness_db):
    """Wrapper that selects the appropriate linear normalization function."""
    if isinstance(loudness_db, np.ndarray):
        return normalize_loudness_array(loudness_db)
    else:
        return normalize_loudness_scalar(loudness_db)

@njit(fastmath=True)
def normalize_key_scalar(key_number, mode):
    """Linearly normalize key to [0, 1] based on the number of accidentals (0-6)."""
    if key_number == -1.0 or mode == -1.0:
        return -1.0
        
    adjusted_key = (key_number + (3 if mode == 0 else 0)) % 12
    accidentals_map = np.array([0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5], dtype=np.float32)
    return accidentals_map[int(adjusted_key)] / 6.0

@njit(fastmath=True)
def normalize_key_array(key_numbers, modes):
    """Vectorized normalization of key values."""
    result = np.empty_like(key_numbers)
    for i in range(len(key_numbers)):
        key_number = key_numbers[i]
        mode = modes[i]
        if key_number == -1.0 or mode == -1.0:
            result[i] = -1.0
        else:
            adjusted_key = (key_number + (3 if mode == 0 else 0)) % 12
            accidentals_map = np.array([0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5], dtype=np.float32)
            result[i] = accidentals_map[int(adjusted_key)] / 6.0
    return result

def normalize_key(key_number, mode):
    """Wrapper that selects the appropriate key normalization function."""
    if isinstance(key_number, np.ndarray):
        return normalize_key_array(key_number, mode)
    else:
        return normalize_key_scalar(key_number, mode)

@njit(fastmath=True)
def normalize_tempo_scalar(tempo_bpm):
    """Normalize a single tempo value from [40, 250] to [0, 1] using log2 scaling."""
    if tempo_bpm == -1.0:
        return -1.0
    clipped = min(max(tempo_bpm, MIN_TEMPO), MAX_TEMPO)
    log_min = np.log2(MIN_TEMPO)
    log_max = np.log2(MAX_TEMPO)
    return (np.log2(clipped) - log_min) / (log_max - log_min)

@njit(fastmath=True)
def normalize_tempo_array(tempo_bpm):
    """Vectorized normalization of tempo array from [40, 250] to [0, 1] using log2 scaling."""
    result = np.empty_like(tempo_bpm)
    for i in range(len(tempo_bpm)):
        val = tempo_bpm[i]
        if val == -1.0:
            result[i] = -1.0
        else:
            clipped = val
            if clipped < MIN_TEMPO:
                clipped = MIN_TEMPO
            elif clipped > MAX_TEMPO:
                clipped = MAX_TEMPO
            log_min = np.log2(MIN_TEMPO)
            log_max = np.log2(MAX_TEMPO)
            result[i] = (np.log2(clipped) - log_min) / (log_max - log_min)
    return result

def normalize_tempo(tempo_bpm):
    """Wrapper that selects the appropriate tempo normalization function."""
    if isinstance(tempo_bpm, np.ndarray):
        return normalize_tempo_array(tempo_bpm)
    else:
        return normalize_tempo_scalar(tempo_bpm)

@njit(fastmath=True)
def normalize_time_signature_scalar(time_sig):
    """One-hot encode time signature into a 4D binary vector."""
    if time_sig == -1.0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
    if time_sig == 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif time_sig == 3:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    elif time_sig == 5:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    else:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

@njit(fastmath=True)
def normalize_time_signature_array(time_sigs):
    """Vectorized normalization of time signatures."""
    result = np.empty((len(time_sigs), 4), dtype=np.float32)
    for i in range(len(time_sigs)):
        time_sig = time_sigs[i]
        if time_sig == -1.0:
            result[i] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif time_sig == 4:
            result[i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif time_sig == 3:
            result[i] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        elif time_sig == 5:
            result[i] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        else:
            result[i] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return result

def normalize_time_signature(time_sig):
    """Wrapper that selects the appropriate time signature normalization function."""
    if isinstance(time_sig, np.ndarray):
        return normalize_time_signature_array(time_sig)
    else:
        return normalize_time_signature_scalar(time_sig)

@njit(fastmath=True)
def normalize_duration_scalar(duration_ms, c=5.0):
    """Normalize duration to [0, 1) using a rational function."""
    if duration_ms == -1.0:
        return -1.0
        
    minutes = abs(duration_ms) / 60000.0
    return minutes / (minutes + abs(c))

@njit(fastmath=True)
def normalize_duration_array(duration_ms, c=5.0):
    """Vectorized normalization of duration values."""
    result = np.empty_like(duration_ms)
    for i in range(len(duration_ms)):
        val = duration_ms[i]
        if val == -1.0:
            result[i] = -1.0
        else:
            minutes = abs(val) / 60000.0
            result[i] = minutes / (minutes + abs(c))
    return result

def normalize_duration(duration_ms, c=5.0):
    """Wrapper that selects the appropriate duration normalization function."""
    if isinstance(duration_ms, np.ndarray):
        return normalize_duration_array(duration_ms, c)
    else:
        return normalize_duration_scalar(duration_ms, c)

@njit(fastmath=True)
def normalize_release_year_scalar(year):
    """Linearly normalize release year to [0, 1]."""
    if year == -1.0:
        return -1.0
    year_clipped = max(min(year, MAX_YEAR), MIN_YEAR)
    return (year_clipped - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)

@njit(fastmath=True)
def normalize_release_year_array(years):
    """Vectorized normalization of release years."""
    result = np.empty_like(years)
    for i in range(len(years)):
        year = years[i]
        if year == -1.0:
            result[i] = -1.0
        else:
            year_clipped = max(min(year, MAX_YEAR), MIN_YEAR)
            result[i] = (year_clipped - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
    return result

def normalize_release_year(year):
    """Wrapper that selects the appropriate year normalization function."""
    if isinstance(year, np.ndarray):
        return normalize_release_year_array(year)
    else:
        return normalize_release_year_scalar(year)

@njit(fastmath=True)
def normalize_popularity_scalar(popularity):
    """Normalize track popularity to [0, 1] using square root scaling."""
    if popularity == -1.0:
        return -1.0
        
    popularity = max(min(popularity, 100), 0)
    return np.sqrt(popularity / 100.0)

@njit(fastmath=True)
def normalize_popularity_array(popularity):
    """Vectorized normalization of popularity values."""
    result = np.empty_like(popularity)
    for i in range(len(popularity)):
        val = popularity[i]
        if val == -1.0:
            result[i] = -1.0
        else:
            clipped = max(min(val, 100), 0)
            result[i] = np.sqrt(clipped / 100.0)
    return result

def normalize_popularity(popularity):
    """Wrapper that selects the appropriate popularity normalization function."""
    if isinstance(popularity, np.ndarray):
        return normalize_popularity_array(popularity)
    else:
        return normalize_popularity_scalar(popularity)

@njit(fastmath=True)
def normalize_followers_scalar(followers):
    """Normalize artist followers to [0, 1] using log10 scaling."""
    if followers == -1.0 or followers <= 0:
        return -1.0
        
    clipped = min(followers, MAX_FOLLOWERS)
    return np.log10(clipped + 1) / np.log10(MAX_FOLLOWERS + 1)

@njit(fastmath=True)
def normalize_followers_array(followers):
    """Vectorized normalization of follower counts."""
    result = np.empty_like(followers)
    for i in range(len(followers)):
        val = followers[i]
        if val == -1.0 or val <= 0:
            result[i] = -1.0
        else:
            clipped = min(val, MAX_FOLLOWERS)
            result[i] = np.log10(clipped + 1) / np.log10(MAX_FOLLOWERS + 1)
    return result

def normalize_followers(followers):
    """Wrapper that selects the appropriate followers normalization function."""
    if isinstance(followers, np.ndarray):
        return normalize_followers_array(followers)
    else:
        return normalize_followers_scalar(followers)

@njit(fastmath=True)
def extract_year_from_string(date_str):
    """Numba-compatible year extraction from string"""
    if len(date_str) < 4:
        return -1.0
    year_chars = date_str[:4]
    
    # Check if all characters are digits
    for i in range(4):
        if year_chars[i] < '0' or year_chars[i] > '9':
            return -1.0
    
    # Convert to float
    year = 0.0
    for i in range(4):
        digit = ord(year_chars[i]) - 48  # ASCII '0' is 48
        year = year * 10 + digit
    
    return year

@njit(parallel=True, fastmath=True)
def extract_years_batch(date_strings):
    """Vectorized year extraction optimized for Numba"""
    n = len(date_strings)
    years = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        years[i] = extract_year_from_string(date_strings[i])
    
    return years

def compute_genre_intensities_batch(genres_list: List[List[str]]) -> np.ndarray:
    """
    Vectorized genre intensity calculation using precomputed IDs.
    Returns an array of shape (batch_size, 13) with genre intensities.
    """
    batch_size = len(genres_list)
    intensities = np.full((batch_size, 13), -1.0, dtype=np.float32)
    
    for i in range(batch_size):
        genres = genres_list[i]
        if not genres:
            continue
            
        # Track max intensity per meta-genre
        meta_intensity = np.full(13, -1.0, dtype=np.float32)
        
        for genre in genres:
            genre_lower = genre.strip().lower()
            if genre_lower in GENRE_MAPPING:
                meta_idx, intensity = GENRE_MAPPING[genre_lower]
                # Convert to zero-based index (meta_idx is 1-13)
                array_idx = meta_idx - 1
                if array_idx < 0 or array_idx >= 13:
                    continue
                if intensity > meta_intensity[array_idx]:
                    meta_intensity[array_idx] = intensity
        
        # Fill intensities
        for j in range(13):
            if meta_intensity[j] > -1.0:
                intensities[i, j] = meta_intensity[j]
    
    return intensities

@njit(fastmath=True)
def validate_vector_batch(vectors: np.ndarray) -> bool:
    """Batch vector validation with numba acceleration."""
    for i in range(vectors.shape[0]):
        for j in range(32):
            val = vectors[i, j]
            if val < -1.0 or val > 1.0 or np.isnan(val):
                return False
    return True

def build_track_vectors_batch(track_dicts: List[dict]) -> np.ndarray:
    """Optimized batch vector construction with vectorized operations."""
    n = len(track_dicts)
    vectors = np.full((n, 32), -1.0, dtype=np.float32)
    
    # Pre-extract all features into arrays with explicit dtype
    acousticness = np.array([safe_float(t.get('acousticness', -1.0)) for t in track_dicts], dtype=np.float32)
    instrumentalness = np.array([safe_float(t.get('instrumentalness', -1.0)) for t in track_dicts], dtype=np.float32)
    speechiness = np.array([safe_float(t.get('speechiness', -1.0)) for t in track_dicts], dtype=np.float32)
    valence = np.array([safe_float(t.get('valence', -1.0)) for t in track_dicts], dtype=np.float32)
    danceability = np.array([safe_float(t.get('danceability', -1.0)) for t in track_dicts], dtype=np.float32)
    energy = np.array([safe_float(t.get('energy', -1.0)) for t in track_dicts], dtype=np.float32)
    liveness = np.array([safe_float(t.get('liveness', -1.0)) for t in track_dicts], dtype=np.float32)
    loudness = np.array([safe_float(t.get('loudness', -1.0)) for t in track_dicts], dtype=np.float32)
    key = np.array([safe_float(t.get('key', -1.0)) for t in track_dicts], dtype=np.float32)
    mode = np.array([safe_float(t.get('mode', -1.0)) for t in track_dicts], dtype=np.float32)
    tempo = np.array([safe_float(t.get('tempo', -1.0)) for t in track_dicts], dtype=np.float32)
    time_signature = np.array([safe_float(t.get('time_signature', -1.0)) for t in track_dicts], dtype=np.float32)
    duration_ms = np.array([safe_float(t.get('duration_ms', -1.0)) for t in track_dicts], dtype=np.float32)
    popularity = np.array([safe_float(t.get('popularity', -1.0)) for t in track_dicts], dtype=np.float32)
    max_followers = np.array([safe_float(t.get('max_followers', -1.0)) for t in track_dicts], dtype=np.float32)
    genres_list = [t.get('genres', []) for t in track_dicts]
    
    # Extract date strings
    date_strings = [t.get('release_date', '') for t in track_dicts]
    
    # Convert using Numba-optimized function
    release_years = extract_years_batch(date_strings)
    
    # Apply vectorized normalization
    vectors[:, 0] = acousticness
    vectors[:, 1] = instrumentalness
    vectors[:, 2] = speechiness
    vectors[:, 3] = valence
    vectors[:, 4] = danceability
    vectors[:, 5] = energy
    vectors[:, 6] = liveness
    vectors[:, 7] = normalize_loudness(loudness)
    vectors[:, 8] = normalize_key(key, mode)
    vectors[:, 9] = mode
    vectors[:, 10] = normalize_tempo(tempo)
    
    # Time signature normalization
    time_vecs = normalize_time_signature(time_signature)
    vectors[:, 11:15] = time_vecs
    
    vectors[:, 15] = normalize_duration(duration_ms)
    vectors[:, 16] = normalize_release_year(release_years)
    vectors[:, 17] = normalize_popularity(popularity)
    vectors[:, 18] = normalize_followers(max_followers)
    
    # Genre intensities
    genre_intensities = compute_genre_intensities_batch(genres_list)
    vectors[:, 19:32] = genre_intensities
    
    # Replace any NaNs with -1.0 (sentinel value)
    vectors = np.where(np.isnan(vectors), -1.0, vectors)
    
    # Batch validation
    if not validate_vector_batch(vectors):
        # Fallback to individual validation with error logging
        for i in range(n):
            track_id = track_dicts[i].get('track_id', f'index {i}')
            for j in range(32):
                val = vectors[i, j]
                if val < -1.0 or val > 1.0 or np.isnan(val):
                    error_msg = f"Invalid vector for track {track_id}: dimension {j+1} = {val}"
                    with open("vector_errors.log", "a") as f:
                        f.write(error_msg + "\n")
    
    return vectors

# Backward compatibility alias
def build_track_vector(track_dict):
    """Alias for single track processing (uses batch function internally)"""
    return build_track_vectors_batch([track_dict])[0]
