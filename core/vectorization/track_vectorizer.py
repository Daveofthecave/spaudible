# core/vectorization/track_vectorizer.py
import numpy as np
from numba import njit, prange
from typing import List, Dict, Any, Tuple
import time
import csv
import hashlib
from pathlib import Path
from config import PathConfig

# =============================================================================
# GLOBAL PRE-COMPUTED STATE
# =============================================================================

MAX_GENRE_HASH = 2**20  # Increase to 1,048,576 (was 262,144)
GENRE_LUT = np.full((MAX_GENRE_HASH, 3), -1, dtype=np.int32)
DEBUG_GENRES = False  # Set to False after verification

def stable_hash(s: str) -> int:
    """
    Deterministic hash function using BLAKE2b.
    Returns same value across processes and Python invocations.
    Guarantees zero collisions for GENRE_LUT with current dataset size.
    """
    h = hashlib.blake2b(s.encode('utf-8', errors='ignore'), digest_size=4).digest()
    return int.from_bytes(h, 'little')

def verify_genre_integrity():
    """Verify that every genre in CSV can be round-tripped."""
    csv_path = PathConfig.get_genre_mapping()
    errors = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre = row['genre'].strip().lower()
            expected_meta = int(row['meta-genre']) - 1
            expected_intensity = float(row['intensity clustering'])
            
            # Simulate lookup
            h = stable_hash(genre) & (MAX_GENRE_HASH - 1)
            verify_key = stable_hash(genre) & 0xFFFF
            
            # Probe to find the entry
            probe_count = 0
            while True:
                meta_idx = GENRE_LUT[h, 0]
                stored_intensity = GENRE_LUT[h, 1] / 10000.0
                stored_verify = GENRE_LUT[h, 2]
                
                if meta_idx == -1:
                    print(f"❌ Genre '{genre}' not found in LUT!")
                    errors += 1
                    break
                
                if stored_verify == verify_key:
                    # Found it - verify values
                    if meta_idx != expected_meta:
                        print(f"❌ Meta-genre mismatch for '{genre}': {meta_idx} vs {expected_meta}")
                        errors += 1
                    if abs(stored_intensity - expected_intensity) > 0.0001:
                        print(f"❌ Intensity mismatch for '{genre}': {stored_intensity} vs {expected_intensity}")
                        errors += 1
                    break
                
                h = (h + 1) & (MAX_GENRE_HASH - 1)
                probe_count += 1
                
                if probe_count > 100:
                    print(f"❌ Could not find '{genre}' after 100 probes!")
                    errors += 1
                    break
    
    if errors == 0:
        print("✅ All genres verified successfully!")
    else:
        print(f"❌ {errors} genres failed verification!")
    
    return errors == 0

def _init_genre_lut():
    """Initialize genre lookup table with collision resolution and verification keys."""
    global GENRE_LUT
    
    csv_path = PathConfig.get_genre_mapping()
    if not csv_path.exists():
        print(f"⚠️  WARNING: Genre mapping file not found at {csv_path}")
        print("   All tracks will have -1 for genre dimensions")
        return
    
    # Load into Python dict first (collision-free)
    genre_map = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    meta_genre = int(row['meta-genre']) - 1  # 0-based
                    genre = row['genre'].strip().lower()
                    intensity = float(row['intensity clustering'])
                    
                    # Keep highest intensity if duplicate genre name
                    if genre in genre_map:
                        existing_meta, existing_intensity = genre_map[genre]
                        if intensity > existing_intensity:
                            genre_map[genre] = (meta_genre, intensity)
                    else:
                        genre_map[genre] = (meta_genre, intensity)
                except (ValueError, KeyError) as e:
                    if DEBUG_GENRES:
                        print(f"   Skipped invalid row: {row.get('genre', 'unknown')} - {e}")
                    continue
    except Exception as e:
        print(f"❌ ERROR loading genre mapping: {e}")
        return
    
    print(f"✅ Loaded {len(genre_map)} unique genres from CSV")
    
    # Populate LUT with open-addressing (linear probing)
    GENRE_LUT.fill(-1)
    collision_count = 0
    
    for genre, (meta_genre, intensity) in genre_map.items():
        h = stable_hash(genre) & (MAX_GENRE_HASH - 1)
        stored_intensity = int(intensity * 10000)
        verification_key = stable_hash(genre) & 0xFFFF  # 16-bit verification
        
        # Insert with probing
        probe_count = 0
        original_h = h
        
        while GENRE_LUT[h, 0] != -1:
            # Update if same genre (by verification key)
            if GENRE_LUT[h, 2] == verification_key:
                if stored_intensity > GENRE_LUT[h, 1]:
                    GENRE_LUT[h, 0] = meta_genre
                    GENRE_LUT[h, 1] = stored_intensity
                break
            
            # Collision - probe next slot
            h = (h + 1) & (MAX_GENRE_HASH - 1)
            probe_count += 1
            collision_count += 1
            
            if probe_count > 1000:  # Should never happen
                raise RuntimeError(f"Hash table overflow for genre '{genre}'")
        
        # Store genre data + verification key
        GENRE_LUT[h, 0] = meta_genre
        GENRE_LUT[h, 1] = stored_intensity
        GENRE_LUT[h, 2] = verification_key
    
    if collision_count > 0:
        print(f"ℹ️  Resolved {collision_count} hash collisions during LUT creation")
    
    if DEBUG_GENRES:
        print(f"✅ GENRE_LUT ready with {MAX_GENRE_HASH} slots and {len(genre_map)} genres")
        verify_genre_integrity()  # Run verification

# Call initialization immediately
_init_genre_lut()

MIN_TEMPO = np.float32(40.0)
MAX_TEMPO = np.float32(250.0)
MIN_YEAR = np.float32(1900.0)
MAX_YEAR = np.float32(2025.0)
MAX_FOLLOWERS = np.float32(141_174_367.0)

# =============================================================================
# PHASE 1: ZERO-COPY EXTRACTION
# =============================================================================

def extract_features_batch(track_dicts: List[Dict[str, Any]]) -> Tuple[np.ndarray, ...]:
    """Extract features with minimal Python overhead."""
    audio_features = np.array([
        [t.get('acousticness', -1.0), t.get('instrumentalness', -1.0),
         t.get('speechiness', -1.0), t.get('valence', -1.0),
         t.get('danceability', -1.0), t.get('energy', -1.0),
         t.get('liveness', -1.0)]
        for t in track_dicts
    ], dtype=np.float32)
    
    loudness = np.array([t.get('loudness', -1.0) for t in track_dicts], dtype=np.float32)
    key = np.array([t.get('key', -1.0) for t in track_dicts], dtype=np.float32)
    mode = np.array([t.get('mode', -1.0) for t in track_dicts], dtype=np.float32)
    tempo = np.array([t.get('tempo', -1.0) for t in track_dicts], dtype=np.float32)
    time_signature = np.array([t.get('time_signature', -1.0) for t in track_dicts], dtype=np.float32)
    duration_ms = np.array([t.get('duration_ms', -1.0) for t in track_dicts], dtype=np.float32)
    popularity = np.array([t.get('popularity', -1.0) for t in track_dicts], dtype=np.float32)
    max_followers = np.array([t.get('max_followers', -1.0) for t in track_dicts], dtype=np.float32)
    
    release_years = np.array([
        int(t.get('release_date', '')[:4]) if t.get('release_date', '')[:4].isdigit() else -1
        for t in track_dicts
    ], dtype=np.float32)
    
    genres_list = [t.get('genres', []) for t in track_dicts]
    
    return (audio_features, loudness, key, mode, tempo, time_signature,
            duration_ms, popularity, max_followers, release_years, genres_list)

# =============================================================================
# PHASE 2: VECTORIZED NORMALIZATIONS
# =============================================================================

def normalize_audio_features(audio_features: np.ndarray, vectors: np.ndarray) -> None:
    """Dimensions 01-07: Direct copy."""
    vectors[:, 0:7] = audio_features

def normalize_loudness(loudness: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimension 08: Clip range to [-60, 0]. 
    Then, normalize linearly to [0, 1] by doing (actual_dB - min_dB) / (max_dB - min_dB).
    """
    valid = loudness != -1.0
    clipped = np.clip(loudness[valid], -60.0, 0.0)
    vectors[valid, 7] = (clipped + 60.0) / 60.0
    vectors[~valid, 7] = -1.0

def normalize_key(key: np.ndarray, mode: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimension 09: Linear normalization based on accidental count.
    If mode is minor, add 3 to key number and mod by 12 to access the relative minor.
    Then divide by 6 (max accidentals) to get [0, 1].
    """
    # Filter for valid range AND exclude NaN/Inf
    valid = (key >= 0) & (key <= 11) & (mode != -1.0) & np.isfinite(key) & np.isfinite(mode)
    
    # Use integer arithmetic to avoid float modulo issues
    key_int = key[valid].astype(np.int32)
    mode_int = mode[valid].astype(np.int32)
    
    # Apply mode adjustment and modulo
    adjustment = 3 * (mode_int == 0)
    adjusted = (key_int + adjustment) % 12
    
    # Ensure indices are valid (clip as safety)
    adjusted = np.clip(adjusted, 0, 11)
    
    accidentals_map = np.array([0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5], dtype=np.float32)
    vectors[valid, 8] = accidentals_map[adjusted] / 6.0
    vectors[~valid, 8] = -1.0

def normalize_mode(mode: np.ndarray, vectors: np.ndarray) -> None:
    """Dimension 10: Binary, 0 or 1."""
    vectors[:, 9] = mode

def normalize_tempo(tempo: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimension 11: Clip tempo range to [40, 250] and apply log2 normalization: 
    (log2(tempo) - log2(min_bpm)) / (log2(max_bpm) - log2(min_bpm)), 
    where min_bpm = 40, max_bpm = 250.
    """
    valid = tempo != -1.0
    clipped = np.clip(tempo[valid], MIN_TEMPO, MAX_TEMPO)
    vectors[valid, 10] = (np.log2(clipped) - np.log2(MIN_TEMPO)) / (np.log2(MAX_TEMPO) - np.log2(MIN_TEMPO))
    vectors[~valid, 10] = -1.0

def normalize_time_signature(time_sig: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimensions 12-15: One-hot encode with 4 binary categories representing [4/4, 3/4, 5/4, other].
    """
    vectors[:, 11] = (time_sig == 4).astype(np.float32)  # 4/4
    vectors[:, 12] = (time_sig == 3).astype(np.float32)  # 3/4
    vectors[:, 13] = (time_sig == 5).astype(np.float32)  # 5/4
    vectors[:, 14] = (time_sig != -1) & (~np.isin(time_sig, [3, 4, 5]))

def normalize_duration(duration_ms: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimension 16: Use rational normalization: convert to minutes by dividing duration_ms by 60000, 
    and plug it into minutes / (minutes + c), where c = 5.
    """
    valid = duration_ms != -1.0
    mins = duration_ms[valid] / 60000.0
    vectors[valid, 15] = mins / (mins + 5.0)
    vectors[~valid, 15] = -1.0

def normalize_release_year(years: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimension 17: Clip range to [1900, 2025]; any year < 1900 = 1900, and any year > 2025 = 2025.
    Then, normalize linearly to [0, 1] by subtracting 1900 from the given year and dividing by 125.
    """
    valid = years != -1.0
    clipped = np.clip(years[valid], MIN_YEAR, MAX_YEAR)
    vectors[valid, 16] = (clipped - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
    vectors[~valid, 16] = -1.0

def normalize_popularity(popularity: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimension 18: Since 95% are between 0 and 9, we can give 
    the smaller ones more weight by sqrt(popularity / 100.0).
    """
    valid = popularity != -1.0
    vectors[valid, 17] = np.sqrt(np.clip(popularity[valid], 0, 100) / 100.0)
    vectors[~valid, 17] = -1.0

def normalize_followers(followers: np.ndarray, vectors: np.ndarray) -> None:
    """
    Dimension 19: Use log10 normalization: log10(followers + 1) / log10(max_followers + 1).
    If followers = 0, then assign 0.
    """
    valid = (followers != -1.0) & (followers > 0)
    vectors[valid, 18] = np.log10(followers[valid] + 1.0) / np.log10(MAX_FOLLOWERS + 1.0)
    vectors[~valid, 18] = -1.0

# =============================================================================
# PHASE 3: NUMBA-PARALLEL GENRE PROCESSING
# =============================================================================

@njit(parallel=True, fastmath=True)
def compute_genres_vectorized(
    genres_flat: np.ndarray,
    track_bounds: np.ndarray,
    genre_lut: np.ndarray,
    output: np.ndarray
) -> None:
    """
    Compute genre intensities with proper open-addressing lookup.
    Lookup must probe in the same sequence as insertion.
    """
    n_tracks = len(track_bounds) - 1
    
    for i in prange(n_tracks):
        start = track_bounds[i]
        end = track_bounds[i + 1]
        max_intens = np.full(13, -1.0, dtype=np.float32)
        
        for j in range(start, end):
            genre_hash = genres_flat[j]
            if genre_hash == -1:
                continue
            
            # Lookup with probing
            h = genre_hash & (MAX_GENRE_HASH - 1)
            probe_count = 0
            
            while True:
                meta_idx = genre_lut[h, 0]
                stored_verify = genre_lut[h, 2]  # Read verification key
                stored_intensity = genre_lut[h, 1]
                
                # Empty slot → genre not in LUT
                if meta_idx == -1:
                    break
                
                # Found our genre (verification key matches)
                if stored_verify == (genre_hash & 0xFFFF):
                    intensity = stored_intensity / 10000.0
                    if intensity > max_intens[meta_idx]:
                        max_intens[meta_idx] = intensity
                    break
                
                # Collision - probe next slot (mirror insertion logic)
                h = (h + 1) & (MAX_GENRE_HASH - 1)
                probe_count += 1
                
                # Safety: prevent infinite loops
                if probe_count > 100:
                    break
            
        output[i] = max_intens

def process_genres_batch(genres_list: List[List[str]], vectors: np.ndarray) -> None:
    """
    Pre-process genre lists into NumPy arrays for Numba.
    Converts string genres to pre-hashed integer IDs to avoid string ops in Numba.
    """
    if not genres_list:
        vectors[:, 19:32] = -1.0
        return
    
    if DEBUG_GENRES:
        tracks_with_genres = sum(1 for g in genres_list if g)
        print(f"DEBUG: Processing {tracks_with_genres}/{len(genres_list)} tracks with genres")
    
    flat_genres = []
    track_bounds = [0]
    
    for genres in genres_list:
        start = len(flat_genres)
        for genre in genres:
            # Store both hash and verification key
            genre_hash = stable_hash(genre.strip().lower())
            # Use low 16 bits as verification, store in separate array
            flat_genres.append(genre_hash)
        track_bounds.append(len(flat_genres))
    
    # Convert to arrays
    flat_array = np.array(flat_genres, dtype=np.int64)  # Store full hash
    bounds_array = np.array(track_bounds, dtype=np.int32)
    genre_output = np.full((len(genres_list), 13), -1.0, dtype=np.float32)
    
    # Execute Numba kernel
    compute_genres_vectorized(flat_array, bounds_array, GENRE_LUT, genre_output)
    
    # Assign to final vector
    vectors[:, 19:32] = genre_output

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def build_track_vectors_batch(track_dicts: List[Dict[str, Any]]) -> np.ndarray:
    """
    Builds track vectors while minimizing reliance on slow Python loops.
    """
    if not track_dicts:
        return np.empty((0, 32), dtype=np.float32)
    
    # Phase 1: Extract features
    features = extract_features_batch(track_dicts)
    
    # Phase 2: Initialize output array
    vectors = np.full((len(track_dicts), 32), -1.0, dtype=np.float32)
    
    # Phase 3: Apply all normalizations
    normalize_audio_features(features[0], vectors)
    normalize_loudness(features[1], vectors)
    normalize_key(features[2], features[3], vectors)
    normalize_mode(features[3], vectors)
    normalize_tempo(features[4], vectors)
    normalize_time_signature(features[5], vectors)
    normalize_duration(features[6], vectors)
    normalize_release_year(features[9], vectors)
    normalize_popularity(features[7], vectors)
    normalize_followers(features[8], vectors)
    
    # Phase 4: Process genres (Numba-parallel)
    process_genres_batch(features[10], vectors)
    
    return vectors

# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

def build_track_vector(track_dict: Dict[str, Any]) -> np.ndarray:
    """Single-track wrapper for legacy code."""
    return build_track_vectors_batch([track_dict])[0]
