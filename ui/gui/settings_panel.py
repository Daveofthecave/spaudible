# ui/gui/settings_panel.py
import dearpygui.dearpygui as dpg
from ui.gui.theme import add_gradient_button
from core.utilities.config_manager import config_manager
from core.similarity_engine.orchestrator import SearchOrchestrator

class SettingsPanel:
    """Left sidebar containing all configuration controls."""
    
    def __init__(self, dpi_scale: float = 1.0):
        self.dpi_scale = float(dpi_scale)
        self.tag = "settings_panel"

    def _s(self, value) -> int:
        """Scale a pixel value by DPI scale factor."""
        return int(float(value) * self.dpi_scale)

    def build(self):
        """Build the left sidebar with all settings controls."""
        dpg.add_text("Settings", color=(100, 200, 255))
        dpg.add_separator()
        
        if self.dpi_scale != 1.0:
            dpg.add_text(f"Scaling: {self.dpi_scale}x", color=(150, 150, 150))
        
        dpg.add_spacer(height=self._s(10))
        
        # Mode selector
        dpg.add_text("Processing Mode")
        force_cpu = config_manager.get_force_cpu()
        force_gpu = config_manager.get_force_gpu()
        if force_cpu:
            current_mode = "CPU Only"
        elif force_gpu:
            current_mode = "GPU Only"
        else:
            current_mode = "Auto"
            
        dpg.add_radio_button(
            items=["Auto", "CPU Only", "GPU Only"],
            default_value=current_mode,
            callback=self._on_mode_changed
        )
        
        dpg.add_spacer(height=self._s(10))
        
        # Algorithm selector
        dpg.add_text("Similarity Algorithm")
        current_algo = config_manager.get_algorithm_name()
        dpg.add_combo(
            items=["Cosine-Euclidean", "Cosine", "Euclidean"],
            default_value=current_algo,
            callback=self._on_algorithm_changed,
            width=self._s(200)
        )
        
        dpg.add_spacer(height=self._s(10))
        
        # Deduplication toggle
        current_dedupe = config_manager.get_deduplicate()
        dpg.add_checkbox(
            label="Deduplicate Results",
            default_value=current_dedupe,
            callback=self._on_dedupe_changed
        )
        
        dpg.add_spacer(height=self._s(10))
        
        # Region filter slider
        dpg.add_text("Region Filter Strength")
        current_region = config_manager.get_region_strength()
        dpg.add_slider_float(
            default_value=current_region,
            min_value=0.0,
            max_value=1.0,
            width=self._s(250),
            callback=self._on_region_changed
        )
        
        dpg.add_spacer(height=self._s(10))
        
        # Number of results
        dpg.add_text("Number of Results")
        current_topk = config_manager.get_top_k()
        dpg.add_input_int(
            default_value=current_topk,
            min_value=1,
            max_value=1000,
            width=self._s(100),
            callback=self._on_topk_changed
        )
        
        dpg.add_spacer(height=self._s(20))
        
        # Feature weights (collapsible)
        with dpg.tree_node(label="Feature Weights", default_open=False):
            self._build_feature_weights()
        
        dpg.add_separator()
        add_gradient_button(
            label="Reset to Defaults",
            width=self._s(150),
            height=self._s(24),
            callback=self._reset_settings
        )

    def _build_feature_weights(self):
        """Build the feature weight sliders."""
        features = [
            "Acousticness", "Danceability", "Energy", "Valence", 
            "Tempo", "Popularity"
        ]
        current_weights = config_manager.get_weights()
        
        for i, feature in enumerate(features):
            default_val = current_weights[i] if i < len(current_weights) else 1.0
            dpg.add_slider_float(
                label=feature,
                default_value=default_val,
                min_value=0.0,
                max_value=10.0,
                width=self._s(220),
                user_data=i,
                callback=self._on_weight_changed
            )

    def _on_mode_changed(self, sender, app_data):
        """Handle processing mode change."""
        if app_data == "CPU Only":
            config_manager.set_force_cpu(True)
            config_manager.set_force_gpu(False)
        elif app_data == "GPU Only":
            config_manager.set_force_cpu(False)
            config_manager.set_force_gpu(True)
        else:
            config_manager.set_force_cpu(False)
            config_manager.set_force_gpu(False)
        SearchOrchestrator.clear_benchmark_cache()

    def _on_algorithm_changed(self, sender, app_data):
        """Handle algorithm selection change."""
        algo_map = {
            "Cosine-Euclidean": "cosine-euclidean",
            "Cosine": "cosine",
            "Euclidean": "euclidean"
        }
        if app_data in algo_map:
            config_manager.set_algorithm(algo_map[app_data])

    def _on_dedupe_changed(self, sender, app_data):
        """Handle deduplication toggle."""
        config_manager.set_deduplicate(bool(app_data))

    def _on_region_changed(self, sender, app_data):
        """Handle region filter slider change."""
        config_manager.set_region_strength(float(app_data))

    def _on_topk_changed(self, sender, app_data):
        """Handle number of results change."""
        config_manager.set_top_k(int(app_data))

    def _on_weight_changed(self, sender, app_data, user_data):
        """Handle weight slider change."""
        weights = config_manager.get_weights()
        if user_data is not None and 0 <= user_data < len(weights):
            weights[user_data] = float(app_data)
            config_manager.set_weights(weights)

    def _reset_settings(self, sender=None, app_data=None):
        """Reset all settings to defaults."""
        config_manager.reset_weights()
        config_manager.set_force_cpu(False)
        config_manager.set_force_gpu(False)
        config_manager.set_algorithm("cosine-euclidean")
        config_manager.set_deduplicate(True)
        config_manager.set_region_strength(1.0)
        config_manager.set_top_k(25)
