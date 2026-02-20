# ui/gui/main_window.py
import dearpygui.dearpygui as dpg
import sys
from pathlib import Path
from typing import Optional
from ui.gui.state_manager import gui_state_manager
from ui.gui.theme import apply_theme
from core.utilities.setup_validator import is_setup_complete

class MainWindow:
    """Main GUI window orchestrator for Spaudible."""
    
    def __init__(self):
        self.state_manager = gui_state_manager
        self.window_tag = "main_window"
        self.search_panel_tag = "search_panel"
        self.results_panel_tag = "results_panel"
        self.settings_panel_tag = "settings_panel"
        self._is_context_created = False
        
    def run(self):
        """
        Main entry point. Handles setup wizard vs main window logic,
        initializes DPG, and runs the event loop.
        """
        try:
            # Check if setup is needed first
            if not is_setup_complete():
                self._run_setup_wizard()
                return
                
            # Initialize DPG context and create UI
            self._initialize_dpg()
            
            # Run the render loop
            self._main_loop()
            
        except Exception as e:
            print(f"❗ Fatal GUI error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            self._cleanup()
    
    def _initialize_dpg(self):
        """Initialize Dear PyGui context, viewport, and all UI elements."""
        if self._is_context_created:
            return
            
        # Create context FIRST (this must happen before any other DPG call)
        dpg.create_context()
        self._is_context_created = True
        
        # Apply theme
        apply_theme()
        
        # Get window geometry
        pos, size, maximized = self.state_manager.get_window_geometry()
        
        # Validate geometry (prevent segfault from invalid sizes)
        if size[0] < 400 or size[1] < 300:
            size = [1200, 800]
            pos = [100, 100]
        
        # Create viewport
        dpg.create_viewport(
            title='Spaudible - Music Similarity Search',
            width=size[0],
            height=size[1],
            x_pos=pos[0],
            y_pos=pos[1],
            min_width=800,
            min_height=600
        )
        
        # Build the UI
        self._build_ui()
        
        # Setup and show
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        # Handle maximized state (platform-specific)
        if maximized:
            # DPG doesn't have direct maximize API, but we can approximate
            # by setting to screen size. For now, we skip this.
            pass
        
        dpg.set_primary_window(self.window_tag, True)
    
    def _build_ui(self):
        """Build the main application UI."""
        with dpg.window(tag=self.window_tag, label="Spaudible"):
            # Menu bar for tools/about
            with dpg.menu_bar():
                with dpg.menu(label="Tools"):
                    dpg.add_menu_item(label="System Status", callback=self._show_system_status)
                    dpg.add_menu_item(label="Performance Test", callback=self._show_performance_test)
                    dpg.add_menu_item(label="Check for Updates", callback=self._check_updates)
                
                with dpg.menu(label="Help"):
                    dpg.add_menu_item(label="About", callback=self._show_about)
            
            # Main horizontal layout
            with dpg.group(horizontal=True):
                # Left sidebar - Settings (300px fixed width)
                with dpg.child_window(
                    tag=self.settings_panel_tag,
                    width=300,
                    border=True,
                    autosize_x=False,
                    autosize_y=True
                ):
                    self._build_settings_panel()
                
                # Right area - Search & Results (flexible width)
                with dpg.child_window(
                    tag=self.results_panel_tag,
                    border=True,
                    autosize_x=True,
                    autosize_y=True
                ):
                    self._build_search_panel()
                    self._build_results_panel()
    
    def _build_settings_panel(self):
        """Build the left sidebar with all settings controls."""
        dpg.add_text("Settings", color=(100, 200, 255))
        dpg.add_separator()
        
        # Mode selector (Auto/CPU/GPU)
        dpg.add_text("Processing Mode")
        dpg.add_radio_button(
            items=["Auto", "CPU Only", "GPU Only"],
            default_value="Auto",
            callback=self._on_mode_changed
        )
        dpg.add_spacer(height=10)
        
        # Algorithm selector
        dpg.add_text("Similarity Algorithm")
        dpg.add_combo(
            items=["Cosine-Euclidean", "Cosine", "Euclidean"],
            default_value="Cosine-Euclidean",
            callback=self._on_algorithm_changed
        )
        dpg.add_spacer(height=10)
        
        # Deduplication toggle
        dpg.add_checkbox(
            label="Deduplicate Results",
            default_value=True,
            callback=self._on_dedupe_changed
        )
        dpg.add_spacer(height=10)
        
        # Region filter slider
        dpg.add_text("Region Filter Strength")
        dpg.add_slider_float(
            default_value=1.0,
            min_value=0.0,
            max_value=1.0,
            callback=self._on_region_changed
        )
        dpg.add_spacer(height=10)
        
        # Number of results
        dpg.add_text("Number of Results")
        dpg.add_input_int(
            default_value=25,
            min_value=1,
            max_value=1000,
            callback=self._on_topk_changed
        )
        dpg.add_spacer(height=20)
        
        # Feature weights (collapsible)
        with dpg.tree_node(label="Feature Weights", default_open=False):
            self._build_feature_weights()
        
        dpg.add_separator()
        dpg.add_button(
            label="Reset to Defaults",
            callback=self._reset_settings
        )
    
    def _build_feature_weights(self):
        """Build the 32 feature weight sliders."""
        # Simplified version - full implementation would have all 32
        features = [
            "Acousticness", "Danceability", "Energy", 
            "Valence", "Tempo", "Popularity"
        ]
        for feature in features:
            dpg.add_slider_float(
                label=feature,
                default_value=1.0,
                min_value=0.0,
                max_value=10.0,
                width=-1
            )
    
    def _build_search_panel(self):
        """Build the search input section."""
        dpg.add_text("Search", color=(100, 200, 255))
        dpg.add_separator()
        
        dpg.add_input_text(
            tag="search_input",
            hint="Enter song, artist, track ID, ISRC, or drag audio file...",
            width=-1,
            callback=self._on_search_enter,
            on_enter=True
        )
        
        with dpg.group(horizontal=True):
            dpg.add_button(
                tag="search_button",
                label="Find Similar Songs",
                callback=self._handle_search
            )
            dpg.add_button(
                label="Clear",
                callback=self._clear_search
            )
        
        dpg.add_spacer(height=10)
    
    def _build_results_panel(self):
        """Build the results display section."""
        dpg.add_text("Results", color=(100, 200, 255))
        dpg.add_separator()
        
        # Expand/Collapse all button
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Expand All",
                callback=self._expand_all_results
            )
            dpg.add_button(
                label="Collapse All",
                callback=self._collapse_all_results
            )
            dpg.add_button(
                label="Save Playlist",
                callback=self._save_playlist
            )
        
        dpg.add_spacer(height=5)
        
        # Results container
        with dpg.child_window(
            tag="results_container",
            autosize_x=True,
            autosize_y=True,
            border=False
        ):
            dpg.add_text(
                tag="results_placeholder",
                default_value="Enter a search query above to find similar songs.",
                color=(150, 150, 150)
            )
    
    def _main_loop(self):
        """Run the Dear PyGui render loop."""
        print("DEBUG: Entering render loop...")
        
        # Track geometry for save-on-exit
        last_save_time = 0
        save_interval = 5.0  # Save geometry every 5 seconds if changed
        
        # Store initial geometry to detect changes
        prev_pos = dpg.get_viewport_pos()
        prev_size = [dpg.get_viewport_width(), dpg.get_viewport_height()]
        
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            
            # Periodic geometry save (optional - remove if not needed)
            import time
            current_time = time.time()
            if current_time - last_save_time > save_interval:
                current_pos = dpg.get_viewport_pos()
                current_size = [dpg.get_viewport_width(), dpg.get_viewport_height()]
                
                # Only save if actually changed
                if (current_pos != prev_pos or current_size != prev_size):
                    self._save_window_geometry()
                    prev_pos = current_pos
                    prev_size = current_size
                
                last_save_time = current_time
        
        print("DEBUG: Render loop exited")

    def _cleanup(self):
        """Save state and cleanup DPG resources."""
        if not self._is_context_created:
            return
            
        try:
            # Save window geometry before destroying
            self._save_window_geometry()
        except Exception as e:
            print(f"⚠️ Error saving geometry: {e}")
        
        try:
            dpg.destroy_context()
            self._is_context_created = False
            print("DEBUG: Context destroyed")
        except Exception as e:
            print(f"⚠️ Error destroying context: {e}")
    
    def _run_setup_wizard(self):
        """Launch setup wizard (placeholder for future implementation)."""
        # For now, fall back to CLI setup
        print("Setup required. Falling back to CLI setup...")
        # In full implementation, this would show a DPG-based wizard
        from ui.cli.menu_system.database_check import screen_database_check
        screen_database_check()
    
    # Callback methods
    def _handle_search(self):
        """Handle search button click."""
        query = dpg.get_value("search_input")
        if not query.strip():
            dpg.set_value("results_placeholder", "Please enter a search query.")
            return
        
        # TODO: Integrate with actual search logic from core.similarity_engine
        dpg.set_value("results_placeholder", f"Searching for: {query}...\n\n(Integration pending)")
    
    def _on_search_enter(self, sender, app_data):
        """Handle Enter key in search box."""
        if app_data:  # Only trigger if there's text
            self._handle_search()
    
    def _clear_search(self, sender=None, app_data=None):
        """Clear the search input."""
        dpg.set_value("search_input", "")
        dpg.focus_item("search_input")
    
    def _on_mode_changed(self, sender, app_data):
        """Handle processing mode change."""
        # Update config manager based on selection
        pass
    
    def _on_algorithm_changed(self, sender, app_data):
        """Handle algorithm selection change."""
        pass
    
    def _on_dedupe_changed(self, sender, app_data):
        """Handle deduplication toggle."""
        pass
    
    def _on_region_changed(self, sender, app_data):
        """Handle region filter slider change."""
        pass
    
    def _on_topk_changed(self, sender, app_data):
        """Handle number of results change."""
        pass
    
    def _reset_settings(self, sender=None, app_data=None):
        """Reset all settings to defaults."""
        pass
    
    def _expand_all_results(self, sender=None, app_data=None):
        """Expand all result rows."""
        pass
    
    def _collapse_all_results(self, sender=None, app_data=None):
        """Collapse all result rows."""
        pass
    
    def _save_playlist(self, sender=None, app_data=None):
        """Save current results as playlist."""
        pass
    
    def _show_system_status(self, sender=None, app_data=None):
        """Show system status modal."""
        pass
    
    def _show_performance_test(self, sender=None, app_data=None):
        """Run/show performance test."""
        pass
    
    def _check_updates(self, sender=None, app_data=None):
        """Check for updates."""
        pass
    
    def _show_about(self, sender=None, app_data=None):
        """Show about dialog."""
        with dpg.window(
            label="About Spaudible",
            modal=True,
            width=400,
            height=300,
            no_resize=True
        ):
            dpg.add_text("Spaudible v0.3.0")
            dpg.add_text("Music Similarity Search Tool")
            dpg.add_separator()
            dpg.add_text("By Daveofthecave")
            dpg.add_button(label="Close", callback=lambda: dpg.delete_item(dpg.last_container()))
    
    def _save_window_geometry(self):
        """Save current window position and size."""
        try:
            pos = dpg.get_viewport_pos()
            width = dpg.get_viewport_width()
            height = dpg.get_viewport_height()
            
            self.state_manager.set_window_geometry(
                pos=[int(pos[0]), int(pos[1])],
                size=[int(width), int(height)],
                maximized=False  # TODO: Detect maximized state
            )
        except Exception as e:
            print(f"⚠️ Failed to save window geometry: {e}")
