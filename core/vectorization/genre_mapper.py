# core/vectorization/genre_mapper.py
import csv
from config import PathConfig

# Global dictionary to cache the meta-genre mappings
_GENRE_MAPPING = None

def load_genre_mapping(csv_path=None):
    """Load the genre intensity mapping from the CSV file."""
    global _GENRE_MAPPING
    
    if _GENRE_MAPPING is not None:
        return _GENRE_MAPPING
    
    _GENRE_MAPPING = {}

    if csv_path is None:
        csv_path = PathConfig.get_genre_mapping()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 4:
                    continue
                
                try:
                    # meta-genre is 1-13; convert to 0-based index
                    meta_genre = int(row[1]) - 1
                    subgenre = row[2].strip().lower()
                    intensity = float(row[3])
                    
                    # Store mapping
                    _GENRE_MAPPING[subgenre] = (meta_genre, intensity)
                except (ValueError, IndexError):
                    continue
        
    except FileNotFoundError:
        print(f"Warning: Genre mapping file not found at {csv_path}")
        _GENRE_MAPPING = {}
    except Exception as e:
        print(f"Error loading genre mapping: {e}")
        _GENRE_MAPPING = {}
    
    return _GENRE_MAPPING

def compute_genre_intensities(genre_list):
    """Compute genre intensity values for dimensions 20-32 (13 meta-genres)."""
    if not genre_list:
        return [-1.0] * 13
    
    genre_mapping = load_genre_mapping()
    if not genre_mapping:
        return [-1.0] * 13
    
    # Initialize with -1.0 (sentinel for missing)
    intensities = [-1.0] * 13
    
    for genre in genre_list:
        genre_lower = genre.strip().lower()
        if genre_lower in genre_mapping:
            meta_idx, intensity = genre_mapping[genre_lower]
            
            # Take the maximum intensity value for each meta-genre
            if intensities[meta_idx] == -1.0 or intensity > intensities[meta_idx]:
                intensities[meta_idx] = intensity
    
    return intensities

def compute_genre_intensities_batch(genre_lists):
    """Compute genre intensities for a batch of genre lists."""
    genre_mapping = load_genre_mapping()
    if not genre_mapping:
        return [[-1.0] * 13 for _ in genre_lists]
    
    batch_intensities = []
    
    for genre_list in genre_lists:
        if not genre_list:
            batch_intensities.append([-1.0] * 13)
            continue
            
        # Initialize with -1.0 (sentinel for missing)
        intensities = [-1.0] * 13
        
        for genre in genre_list:
            genre_lower = genre.strip().lower()
            if genre_lower in genre_mapping:
                meta_idx, intensity = genre_mapping[genre_lower]
                
                # Take the maximum intensity value for each meta-genre
                if intensities[meta_idx] == -1.0 or intensity > intensities[meta_idx]:
                    intensities[meta_idx] = intensity
        
        batch_intensities.append(intensities)
    
    return batch_intensities
