# core/utilities/config_manager.py
import json
from pathlib import Path
from config import PathConfig
from datetime import datetime

class ConfigManager:
    ALGORITHM_CHOICES = {
        'cosine': 'Cosine Similarity',
        'cosine-euclidean': 'Cosine-Euclidean Similarity',
        'euclidean': 'Euclidean Similarity'
    }
    
    DEFAULT_VECTOR_WEIGHTS = [
                              1.25,  # acousticness
                              1.25,  # instrumentalness
                              1.25,  # speechiness
                              0.95,  # valence
                              1.25,  # danceability
                              1.15,  # energy
                              1.25,  # liveness
                              1.0,   # loudness
                              0.28,  # key
                              1.0,   # mode
                              1.05,  # tempo
                              1.0,   # time_signature_4_4
                              1.0,   # time_signature_3_4
                              1.0,   # time_signature_5_4
                              1.0,   # time_signature_other
                              0.8,   # duration
                              1.8,   # release_date
                              0.6,   # popularity
                              0.8,   # artist_followers
                              # Genre weights
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0, 
                              1.0
                             ]

    DEFAULT_SETTINGS = {
        'vector_weights': DEFAULT_VECTOR_WEIGHTS,
        'similarity_algorithm': 'cosine-euclidean',
        'force_cpu': False,
        'force_gpu': False,
        'deduplicate': True,
        'dedupe_threshold': 0.92,
        "region_strength": 1.0,
        'top_k': 25  # How many songs in the output songlist
    }
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load()
        return cls._instance
    
    def load(self):
        self.config_path = PathConfig.get_config_path()
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.settings = json.load(f)
                
                # Handle the transition from 'weights' to 'vector_weights'.
                # If the old key 'weights' exists, we need to decide what to do.
                # If 'weights' was the default [1.0]*32, we just use the new defaults.
                # If 'weights' was custom, we might want to migrate it, but given the 
                # semantic change (multipliers vs absolute weights), it's safer 
                # to just drop the old key and use the new defaults to ensure correctness.
                
                if 'weights' in self.settings:
                    # If the old key exists, we remove it so the new key takes over.
                    # This prevents the old multiplier logic from interfering.
                    del self.settings['weights']
                    # We don't save here to avoid unnecessary writes, 
                    # but it will be saved on next change.

                # Ensure new settings exist
                for key, default in self.DEFAULT_SETTINGS.items():
                    if key not in self.settings:
                        self.settings[key] = default
            else:
                self.settings = self.DEFAULT_SETTINGS.copy()
        except:
            self.settings = self.DEFAULT_SETTINGS.copy()
    
    def save(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def get(self, key, default=None):
        return self.settings.get(key, default)
    
    def set(self, key, value):
        self.settings[key] = value
        self.save()
    
    def get_force_cpu(self):
        return self.get('force_cpu', False)
    
    def set_force_cpu(self, value):
        self.set('force_cpu', value)

    def get_force_gpu(self):
        return self.get('force_gpu', False)
    
    def set_force_gpu(self, value):
        self.set('force_gpu', value)

    def get_algorithm(self):
        return self.get('similarity_algorithm', 'cosine-euclidean')
    
    def set_algorithm(self, value):
        if value not in self.ALGORITHM_CHOICES:
            raise ValueError(f"Invalid algorithm: {value}")
        self.set('similarity_algorithm', value)
    
    def get_algorithm_name(self):
        algo = self.get_algorithm()
        return self.ALGORITHM_CHOICES.get(algo, 'Unknown')
    
    def get_weights(self):
        return self.settings.get('vector_weights', self.DEFAULT_VECTOR_WEIGHTS)
    
    def set_weights(self, weights):
        """Set vector weights directly."""
        if len(weights) != 32:
            raise ValueError("Weights must have exactly 32 values")
        if not all(isinstance(w, (int, float)) for w in weights):
            raise ValueError("All weights must be numbers")
        self.settings['vector_weights'] = weights
        self.save()
    
    def reset_weights(self):
        """Reset weights to baseline defaults."""
        self.settings['vector_weights'] = self.DEFAULT_VECTOR_WEIGHTS.copy()
        self.save()
    
    def get_deduplicate(self):
        return self.get('deduplicate', True)
    
    def set_deduplicate(self, value):
        self.set('deduplicate', bool(value))

    def set_dedupe_threshold(self, value):
        threshold = float(value)
        if not 0.5 <= threshold <= 1.0:
            raise ValueError("Deduplication threshold must be between 0.5 and 1.0")
        self.set('dedupe_threshold', threshold)

    def get_region_strength(self) -> float:
        """Get current region filter strength."""
        return self.get('region_strength', 1.0)
    
    def set_region_strength(self, value: float):
        """Set region filter strength (0.0-1.0)."""
        value = max(0.0, min(1.0, float(value)))
        self.set('region_strength', value)

    def get_optimal_chunk_size(self) -> int:
        """Get stored optimal chunk size for this hardware (default 200k)."""
        return self.get('optimal_chunk_size', 200_000)
    
    def set_optimal_chunk_size(self, size: int):
        """Store optimal chunk size and timestamp."""
        self.set('optimal_chunk_size', size)
        self.set('chunk_size_last_updated', datetime.now().isoformat())

    def get_benchmark_result(self):
        """Get cached benchmark result if available and valid."""
        result = self.get('benchmark_result', None)
        if result is None:
            return None
        
        # Validate structure
        required_keys = ['recommended_device', 'cpu_speed', 'gpu_speed', 'optimal_chunk_size']
        if not all(key in result for key in required_keys):
            # print("  ⚠️  Cached benchmark result is corrupted, ignoring...")
            return None
        
        return result

    def set_benchmark_result(self, result: dict):
        """Save benchmark result to config file for persistence."""
        self.set('benchmark_result', result)

    def clear_benchmark_result(self):
        """Clear cached benchmark result (eg. when toggling force modes)."""
        self.set('benchmark_result', None)

    def get_top_k(self) -> int:
        """Get number of results to return per search."""
        return self.get('top_k', 25)

    def set_top_k(self, value: int):
        """Set number of results (1-1M)."""
        value = max(1, min(1_000_000, int(value)))
        self.set('top_k', value)

# Singleton access
config_manager = ConfigManager()
