# ui/gui/main_window.py
import dearpygui.dearpygui as dpg
import sys
import platform
from pathlib import Path
from typing import Optional, Union
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
        self.dpi_scale = 1.0

    def _get_dpi_scale(self) -> float:
        """Detect DPI scale factor for high-DPI displays."""
        saved_scale = self.state_manager.get('dpi_scale', 0.0)
        if saved_scale > 0:
            return saved_scale
        
        try:
            if platform.system() == 'Windows':
                import ctypes
                user32 = ctypes.windll.user32
                try:
                    user32.SetProcessDpiAwarenessContext(-4)
                except:
                    user32.SetProcessDPIAware()
                dc = user32.GetDC(0)
                dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)
                user32.ReleaseDC(0, dc)
                scale = dpi / 96.0
            elif platform.system() == 'Darwin':
                scale = 2.0  # macOS Retina default
            else:  # Linux
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    dpi = root.winfo_fpixels('1i')
                    root.destroy()
                    scale = dpi / 96.0
                except:
                    scale = 1.0
            return max(0.75, min(3.0, scale))
        except:
            return 1.0

    def _s(self, value: Union[int, float]) -> int:
        """Scale a pixel value by DPI scale factor."""
        return int(value * self.dpi_scale)

    def run(self):
        """
        Main entry point. Handles setup wizard vs main window logic, initializes DPG, and runs the event loop.
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

        # Create context first (this must happen before any other DPG call)
        dpg.create_context()
        self._is_context_created = True

        # Detect actual screen resolution
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except:
            screen_width, screen_height = 1920, 1080

        # Calculate scale based on 1080p baseline (1.0)
        # This gives us ~2.0 for 4K monitors
        self.dpi_scale = min(screen_width / 1920, screen_height / 1080)
        self.dpi_scale = max(1.0, min(self.dpi_scale, 2.5))  # Clamp 1.0-2.5

        apply_theme()

        # The UI elements will be scaled via _s() method
        dpg.set_global_font_scale(self.dpi_scale)

        # Create viewport at native high resolution (80% of screen)
        # This ensures crisp rendering instead of upscaling a small buffer
        width = int(screen_width * 0.8)
        height = int(screen_height * 0.8)
        x_pos = int((screen_width - width) / 2)
        y_pos = int((screen_height - height) / 2)

        dpg.create_viewport(
            title='Spaudible - Music Similarity Search',
            width=width,
            height=height,
            x_pos=x_pos,
            y_pos=y_pos,
            min_width=self._s(800),
            min_height=self._s(600)
        )

        self._build_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
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
                    dpg.add_separator()
                    dpg.add_menu_item(label="UI Scale: Auto", callback=lambda: self._set_dpi_scale(0))
                    dpg.add_menu_item(label="UI Scale: 100%", callback=lambda: self._set_dpi_scale(1.0))
                    dpg.add_menu_item(label="UI Scale: 150%", callback=lambda: self._set_dpi_scale(1.5))
                    dpg.add_menu_item(label="UI Scale: 200%", callback=lambda: self._set_dpi_scale(2.0))
                with dpg.menu(label="Help"):
                    dpg.add_menu_item(label="About", callback=self._show_about)

            # Main horizontal layout
            with dpg.group(horizontal=True):
                # Left sidebar - Settings (300px fixed width)
                with dpg.child_window(
                    tag=self.settings_panel_tag,
                    width=self._s(300),
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
        
        # Show current scale indicator
        if self.dpi_scale != 1.0:
            dpg.add_text(f"Scale: {self.dpi_scale:.2f}x", color=(150, 150, 150))
            dpg.add_spacer(height=self._s(10))

        # Mode selector (Auto/CPU/GPU)
        dpg.add_text("Processing Mode")
        dpg.add_radio_button(
            items=["Auto", "CPU Only", "GPU Only"],
            default_value="Auto",
            callback=self._on_mode_changed
        )
        dpg.add_spacer(height=self._s(10))

        # Algorithm selector
        dpg.add_text("Similarity Algorithm")
        dpg.add_combo(
            items=["Cosine-Euclidean", "Cosine", "Euclidean"],
            default_value="Cosine-Euclidean",
            callback=self._on_algorithm_changed,
            width=self._s(200)
        )
        dpg.add_spacer(height=self._s(10))

        # Deduplication toggle
        dpg.add_checkbox(
            label="Deduplicate Results",
            default_value=True,
            callback=self._on_dedupe_changed
        )
        dpg.add_spacer(height=self._s(10))

        # Region filter slider
        dpg.add_text("Region Filter Strength")
        dpg.add_slider_float(
            default_value=1.0,
            min_value=0.0,
            max_value=1.0,
            width=self._s(250),
            callback=self._on_region_changed
        )
        dpg.add_spacer(height=self._s(10))

        # Number of results
        dpg.add_text("Number of Results")
        dpg.add_input_int(
            default_value=25,
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
        dpg.add_button(
            label="Reset to Defaults",
            width=self._s(150),
            callback=self._reset_settings
        )

    def _build_feature_weights(self):
        """Build the 32 feature weight sliders."""
        # Simplified version - full implementation would have all 32
        features = [
            "Acousticness", "Danceability", "Energy", "Valence", 
            "Tempo", "Popularity"
        ]
        for feature in features:
            dpg.add_slider_float(
                label=feature,
                default_value=1.0,
                min_value=0.0,
                max_value=10.0,
                width=self._s(220)
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
                width=self._s(150),
                callback=self._handle_search
            )
            dpg.add_button(
                label="Clear",
                width=self._s(80),
                callback=self._clear_search
            )
        dpg.add_spacer(height=self._s(10))

    def _build_results_panel(self):
        """Build the results display section."""
        dpg.add_text("Results", color=(100, 200, 255))
        dpg.add_separator()

        # Expand/Collapse all button
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Expand All",
                width=self._s(100),
                callback=self._expand_all_results
            )
            dpg.add_button(
                label="Collapse All",
                width=self._s(100),
                callback=self._collapse_all_results
            )
            dpg.add_button(
                label="Save Playlist",
                width=self._s(120),
                callback=self._save_playlist
            )
        dpg.add_spacer(height=self._s(5))

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

    def _set_dpi_scale(self, scale: float):
        """Change DPI scale at runtime (requires restart)."""
        self.state_manager.set('dpi_scale', scale)
        with dpg.window(
            label="Restart Required",
            modal=True,
            width=self._s(300),
            height=self._s(100),
            no_resize=True
        ):
            dpg.add_text("UI scale will change on next restart.")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item(dpg.last_container()))

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
            width=self._s(400),
            height=self._s(300),
            no_resize=True
        ):
            dpg.add_text("Spaudible v0.3.0")
            dpg.add_separator()
            dpg.add_text("By Daveofthecave")
            dpg.add_button(label="Close", callback=lambda: dpg.delete_item(dpg.last_container()))

    def _save_window_geometry(self):
        """Save current window position and size."""
        try:
            pos = dpg.get_viewport_pos()
            width = dpg.get_viewport_width()
            height = dpg.get_viewport_height()
            
            # Save unscaled values so they work on different DPI screens
            self.state_manager.set_window_geometry(
                pos=[int(pos[0] / self.dpi_scale), int(pos[1] / self.dpi_scale)],
                size=[int(width / self.dpi_scale), int(height / self.dpi_scale)],
                maximized=False  # TODO: Detect maximized state
            )
        except Exception as e:
            print(f"⚠️ Failed to save window geometry: {e}")
