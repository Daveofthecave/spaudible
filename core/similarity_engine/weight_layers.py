# core/similarity_engine/weight_layers.py
import numpy as np

class WeightLayers:
    """Manages feature weighting for similarity calculations"""
    
    # Feature indices for easy reference
    GENRE_START = 19
    GENRE_END = 31
    
    def __init__(self):
        # Baseline weights for each dimension
        self.baseline_weights = np.array([
            1.0,  # acousticness
            1.0,  # instrumentalness
            1.0,  # speechiness
            1.0,  # valence
            1.0,  # danceability
            1.0,  # energy
            1.0,  # liveness
            1.0,  # loudness
            1.0,  # key
            1.0,  # mode
            1.0,  # tempo
            1.0,  # time_signature_4_4
            1.0,  # time_signature_3_4
            1.0,  # time_signature_5_4
            1.0,  # time_signature_other
            1.0,  # duration
            1.0,  # release_date
            1.0,  # popularity
            1.0,  # artist_followers
            # Genre weights
            1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 
            1.2, 1.2, 1.2, 1.2, 1.2, 1.2
        ], dtype=np.float32)
        
        # Availability boost factor for present attributes
        self.availability_boost = 1.3

        # Reduction factor for when genre data is missing
        self.genre_reduction = 0.8    

        # Precomputed constants
        self.genre_mask = np.zeros(32, dtype=bool)
        self.genre_mask[19:32] = True   
        
    def get_weights(self, u, v):
        """
        Calculate combined weights considering:
        - Baseline feature importance
        - Data availability in both vectors
        """
        # Create mask of valid dimensions
        mask = (u != -1) & (v != -1)
        
        # Apply availability boost to dimensions present in both vectors
        availability_weights = np.where(mask, self.availability_boost, 1.0)
        
        # Combine weights
        combined_weights = self.baseline_weights * availability_weights
        
        # Genre-specific adjustment: Only boost if both vectors have genre data
        genre_mask = np.zeros(32, dtype=bool)
        genre_mask[self.GENRE_START:self.GENRE_END+1] = True
        
        u_has_genre = np.any(u[genre_mask] != -1)
        v_has_genre = np.any(v[genre_mask] != -1)
        
        if not (u_has_genre and v_has_genre):
            # Reduce genre weights if either vector lacks genre data
            combined_weights[genre_mask] = \
                self.baseline_weights[genre_mask] * self.genre_reduction
            
        return combined_weights, mask
