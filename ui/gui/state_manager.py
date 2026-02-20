# ui/gui/state_manager.py
import json
from pathlib import Path
from typing import Dict, Any, Optional
from config import PathConfig

class GUIStateManager:
    """
    Manages persistent GUI-specific settings and window state.
    This is separate from config.py which handles core application settings.
    """
    
    DEFAULT_STATE = {
        'window_size': [1200, 800],
        'window_pos': [100, 100],
        'window_maximized': False,
        'last_query': '',
        'settings_expanded': True,
        'feature_weights_expanded': False,
        'last_results_collapsed': True,
        'theme': 'dark'
    }
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize with default paths and settings."""
        self.config_path = PathConfig.BASE_DIR / "data" / "gui_state.json"
        self.load()
    
    def load(self):
        """Load GUI state from JSON file, creating with defaults if missing."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.state = json.load(f)
                
                # Merge with defaults to handle new keys from updates
                for key, default in self.DEFAULT_STATE.items():
                    if key not in self.state:
                        self.state[key] = default
            else:
                self.state = self.DEFAULT_STATE.copy()
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, start fresh with defaults
            self.state = self.DEFAULT_STATE.copy()
    
    def save(self):
        """Save current GUI state to JSON file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.state, f, indent=2)
        except IOError as e:
            # Log error but don't crash the GUI
            print(f"  ⚠️ Failed to save GUI state: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a GUI state value by key."""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a GUI state value and auto-save."""
        self.state[key] = value
        self.save()
    
    def get_window_geometry(self) -> tuple:
        """Get window position and size as a tuple."""
        return (
            self.get('window_pos', [100, 100]),
            self.get('window_size', [1200, 800]),
            self.get('window_maximized', False)
        )
    
    def set_window_geometry(self, pos: list, size: list, maximized: bool):
        """Set window position, size, and maximized state."""
        self.set('window_pos', pos)
        self.set('window_size', size)
        self.set('window_maximized', maximized)
    
    def get_last_query(self) -> str:
        """Get the last search query."""
        return self.get('last_query', '')
    
    def set_last_query(self, query: str):
        """Set and persist the last search query."""
        self.set('last_query', query)

# Singleton accessor
gui_state_manager = GUIStateManager()
