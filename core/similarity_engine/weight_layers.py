# core/similarity_engine/weight_layers.py
import numpy as np

class WeightLayers:
    """Manages feature weighting for similarity calculations"""
    
    # Feature indices for easy reference
    GENRE_START = 19
    GENRE_END = 31
    
    def __init__(self):
        # Baseline weights are now managed by ConfigManager directly.
        # We only keep dynamic modifiers here.
        
        # Availability boost factor for present attributes
        self.availability_boost = 1.0

        # Reduction factor for when genre data is missing
        self.genre_reduction = 1.0    

        # Precomputed constants
        self.genre_mask = np.zeros(32, dtype=bool)
        self.genre_mask[19:32] = True   
        
    def get_weights(self, u, v):
        """
        Calculate combined weights considering:
        - Data availability in both vectors
        - User defined weights (now passed in directly, not stored here)
        """
        # Create mask of valid dimensions
        mask = (u != -1) & (v != -1)
        
        # Apply availability boost to dimensions present in both vectors
        availability_weights = np.where(mask, self.availability_boost, 1.0)
        
        # Note: The user weights are now applied in VectorOps directly.
        # This method is effectively deprecated for weight calculation 
        # but kept for potential future logic or if availability logic needs to return weights.
        # For now, returning just availability weights to be multiplied by user weights in VectorOps.
        return availability_weights, mask
