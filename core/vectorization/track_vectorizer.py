# core/vectorization/track_vectorizer.py
import math
import numpy as np
from .genre_mapper import compute_genre_intensities

def safe_float(value, default=-1.0):
    """Safely convert a value to float, returning default if conversion fails."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def normalize_loudness(loudness_db):
    """Normalize loudness from [-60, 0] dB to [0, 1]."""
    if loudness_db is None:
        return -1.0
    return (max(min(loudness_db, 0.0), -60.0) + 60.0) / 60.0

def normalize_key(key_number, mode):
    """Normalize key to [0, 1] based on the number of accidentals (0-6)."""
    if key_number is None or mode is None:
        return -1.0
    
    # If mode is minor, relative major is 3 semitones higher
    adjusted_key = (key_number + (3 if mode == 0 else 0)) % 12
    
    # Map the key by number of accidentals: 
    # Major: 0=C, 1=G/F, 2=D/Bb, 3=A/Eb, 4=E/Ab, 5=B/Db,  6=F#
    # Minor: 0=a, 1=e/d, 2=b/g,  3=f#/c, 4=c#/f, 5=g#/bb, 6=d# 
    accidentals_map = [0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5]
    return accidentals_map[adjusted_key] / 6.0

def normalize_tempo(tempo_bpm):
    """Normalize tempo from [40, 250] to [0, 1] using log2 scaling."""
    if tempo_bpm is None:
        return -1.0
    
    MIN_TEMPO = 40.0
    MAX_TEMPO = 250.0
    
    # Clip to valid range
    tempo = max(min(tempo_bpm, MAX_TEMPO), MIN_TEMPO)
    
    # Log2 normalization
    return float((np.log2(tempo) - np.log2(MIN_TEMPO)) / (np.log2(MAX_TEMPO) - np.log2(MIN_TEMPO)))

def normalize_time_signature(time_sig):
    """One-hot encode time signature into a 4D binary vector."""
    if time_sig is None:
        return [0.0, 0.0, 0.0, 0.0]  # All zeros if unknown
    
    if time_sig == 4:
        return [1.0, 0.0, 0.0, 0.0]  # 4/4
    elif time_sig == 3:
        return [0.0, 1.0, 0.0, 0.0]  # 3/4
    elif time_sig == 5:
        return [0.0, 0.0, 1.0, 0.0]  # 5/4
    else:
        return [0.0, 0.0, 0.0, 1.0]    # Other (bogus time signatures like 0/4 and 1/4)

def normalize_duration(duration_ms, c=5.0):
    """Normalize duration to [0, 1) using a rational function."""
    if duration_ms is None:
        return -1.0
    
    # Convert to minutes
    minutes = abs(duration_ms) / 60000.0
    
    # Rational normalization
    return minutes / (minutes + abs(c))

def normalize_release_date(release_date_str):
    """Linearly normalize release year to [0, 1]."""
    if not release_date_str:
        return -1.0
    
    try:
        # First 4 characters represent the year
        # (eg. "2025", "1993-09", or "2006-12-31")
        year_str = release_date_str[:4]
        year = int(year_str)
        
        MIN_YEAR = 1900
        MAX_YEAR = 2025 

        # Clip to valid range
        year = max(min(year, MAX_YEAR), MIN_YEAR)

        # Linear normalization
        return (year - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
    except (ValueError, IndexError):
        return -1.0

def normalize_popularity(popularity):
    """Normalize track popularity to [0, 1] using square root scaling."""
    if popularity is None:
        return -1.0
    
    # Clip to valid range
    popularity = max(min(popularity, 100), 0)

    # Square root normalization
    return math.sqrt(popularity / 100.0)

def normalize_followers(followers):
    """Normalize artist followers to [0, 1] using log10 scaling."""
    if followers is None or followers < 0:
        return -1.0
    
    MAX_FOLLOWERS = 141_174_367 # highest follower count in the database

    # Clip to max
    clipped = min(followers, MAX_FOLLOWERS)

    # Log10 normalization
    return math.log10(clipped + 1) / math.log10(MAX_FOLLOWERS + 1)

def build_track_vector(track_dict):
    """Convert track dictionary to 32-dimensional vector."""
    vector = [-1.0] * 32
    
    # Dim 1-7: Acoustic attributes (normalized by default to [0, 1])
    features = ['acousticness', 'instrumentalness', 'speechiness', 'valence', 
                'danceability', 'energy', 'liveness']
    for i, feature in enumerate(features):
        vector[i] = safe_float(track_dict.get(feature), -1.0)
    
    # Dim 8: Loudness
    loudness = track_dict.get('loudness')
    vector[7] = normalize_loudness(loudness)
    
    # Dim 9: Key
    key = track_dict.get('key')
    mode = track_dict.get('mode')
    vector[8] = normalize_key(key, mode)
    
    # Dim 10: Mode
    vector[9] = safe_float(mode, -1.0)

    # Dim 11: Tempo
    tempo = track_dict.get('tempo')
    vector[10] = normalize_tempo(tempo)

    # Dim 12-15: Time signature
    time_sig = track_dict.get('time_signature')
    time_vec = normalize_time_signature(time_sig)
    vector[11:15] = time_vec

    # Dim 16: Duration
    duration = track_dict.get('duration_ms')
    vector[15] = normalize_duration(duration)

    # Dim 17: Release year
    release_date = track_dict.get('release_date')
    vector[16] = normalize_release_date(release_date)

    # Dim 18: Artist popularity
    popularity = track_dict.get('popularity')
    vector[17] = normalize_popularity(popularity)
    
    # Dim 19: Artist followers
    followers = track_dict.get('max_followers')
    vector[18] = normalize_followers(followers)

    # Dim 20-32: Genre intensities
    genre_list = track_dict.get('genres', [])
    genre_intensities = compute_genre_intensities(genre_list)
    vector[19:32] = genre_intensities
    
    return vector
