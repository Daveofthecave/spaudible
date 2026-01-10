# core/utilities/config_manager.py
import json
from pathlib import Path
from config import PathConfig

class ConfigManager:
    ALGORITHM_CHOICES = {
        'cosine': 'Cosine Similarity',
        'cosine-euclidean': 'Cosine-Euclidean Similarity',
        'euclidean': 'Euclidean Similarity'
    }
    
    DEFAULT_WEIGHTS = [1.0] * 32
    DEFAULT_SETTINGS = {
        'weights': DEFAULT_WEIGHTS,
        'similarity_algorithm': 'cosine-euclidean',
        'force_cpu': False,
        'force_gpu': False,
        'deduplicate': True,
        'dedupe_threshold': 0.92,
        'region_filter': 1.0
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
        return self.settings.get('weights', self.DEFAULT_WEIGHTS)
    
    def set_weights(self, weights):
        if len(weights) != 32:
            raise ValueError("Weights must have exactly 32 values")
        if not all(isinstance(w, (int, float)) for w in weights):
            raise ValueError("All weights must be numbers")
        self.settings['weights'] = weights
        self.save()
    
    def reset_weights(self):
        self.settings['weights'] = self.DEFAULT_WEIGHTS
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

    # Region filter methods
    def get_region_filter(self):
        return self.get('region_filter', 1.0)
    
    def set_region_filter(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Region filter must be between 0.0 and 1.0")
        self.set('region_filter', value)

# Singleton access
config_manager = ConfigManager()
