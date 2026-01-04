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
            else:
                self.settings = {}
        except:
            self.settings = {}
    
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

# Singleton access
config_manager = ConfigManager()