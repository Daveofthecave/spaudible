# ui/gui/main_window.py
import dearpygui.dearpygui as dpg
import sys
import platform
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Union

from ui.gui.state_manager import gui_state_manager
from ui.gui.theme import initialize_theme, add_gradient_button, Colors, _gradient_factory
from ui.gui.settings_panel import SettingsPanel
from ui.gui.search_panel import SearchPanel
from ui.gui.results_view import ResultsView
from core.utilities.setup_validator import is_setup_complete

# Similarity search imports
from core.utilities.text_search_utils import SearchResult
from core.similarity_engine.orchestrator import SearchOrchestrator
from core.vectorization.canonical_track_resolver import build_canonical_vector
from config import PathConfig
from core.utilities.config_manager import config_manager


class MainWindow:
    """Main GUI window orchestrator for Spaudible."""
    
    def __init__(self):
        self.state_manager = gui_state_manager
        self.window_tag = "main_window"
        
        # Initialize sub-panels
        self.dpi_scale = 1.0
        self.settings_panel: Optional[SettingsPanel] = None
        self.search_panel: Optional[SearchPanel] = None
        self.results_view: Optional[ResultsView] = None
        self._is_context_created = False
        self.theme = None  # SpaudibleTheme instance initialized in _initialize_dpg()
        
        # Similarity search threading
        self._similarity_thread: Optional[threading.Thread] = None
        self._orchestrator: Optional[SearchOrchestrator] = None
        self._cancel_event = threading.Event()
        
        # Thread-safe result marshalling
        self._search_results = None
        self._search_complete = False
        self._search_error = None
        self._search_error_flag = False

    def _get_dpi_scale(self) -> float:
        """Universal display scale detection using tkinter.
        
        Since DPI reporting is inconsistent across platforms and displays, we use 
        screen height as the primary heuristic for comfortable UI sizing.
        """
        try:
            import tkinter as tk
            root = tk.Tk()
            
            # Get physical screen dimensions (works everywhere tkinter works)
            screen_height = root.winfo_screenheight()
            # Alternative: root.winfo_screenmmheight() for physical mm, but pixels are more reliable
            
            # Get DPI if available (often returns 96 on many systems regardless of actual DPI)
            try:
                dpi = root.winfo_fpixels('1i')  # pixels per inch
            except Exception:
                dpi = 96
            root.destroy()
            
            # Heuristic: Scale based on vertical resolution for comfortable reading distance
            if screen_height >= 2800:      # 8K and above
                scale = 3
            elif screen_height >= 2100:    # 4K (UHD)
                scale = 2.5
            elif screen_height >= 1600:    # 2K (QHD)
                scale = 2
            elif screen_height >= 1000:    # 1080p (FHD)
                scale = 1.5
            else:                          # Lower resolutions (720p, etc.)
                scale = 1
            
            # Trust high DPI reports only if they're significantly above 96 (>120)
            # This catches Windows/macOS high-DPI modes without breaking Linux
            if dpi > 120:
                calculated_scale = dpi / 96.0
                # Use the higher of the two values, but cap at 2.0
                scale = max(scale, min(2.0, calculated_scale))
            
            # Round to nearest 0.25 to avoid rendering artifacts
            return round(max(0.5, scale) * 4) / 4
        except Exception:
            return 1.0  # Safe fallback

    def _s(self, value: Union[int, float]) -> int:
        """Scale a pixel value by DPI scale factor."""
        return int(value * self.dpi_scale)

    def run(self):
        """
        Main entry point. Handles setup wizard vs main window logic, initializes 
        DPG, and runs the event loop.
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
        
        # Get scale factor (1.0 = 1080p standard, 1.5 = 4K, etc.)
        self.dpi_scale = self._get_dpi_scale()
        
        # Initialize panels with scale
        self.settings_panel = SettingsPanel(self.dpi_scale)
        self.search_panel = SearchPanel(
            self.dpi_scale,
            on_suggestion_selected=self._on_suggestion_selected,
            on_cancel=self._on_search_cancelled
        )
        self.results_view = ResultsView(self.dpi_scale, header_font=getattr(self, 'header_font', None))
        
        # Load fonts at physical pixel size
        self._load_hidpi_font()
        
        # Initialize theme (loads background, creates styles)
        self.theme = initialize_theme()
        
        # Get screen dimensions using tkinter
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Define comfortable logical window size (same "feel" on all screens)
        # 1280x800 is large enough for the UI but fits on 1366x768 laptops
        logical_width = 1280
        logical_height = 800
        
        # Convert to physical pixels
        phys_width = int(logical_width * self.dpi_scale)
        phys_height = int(logical_height * self.dpi_scale)
        
        # Ensure window fits with padding (handles small screens like 1366x768)
        phys_width = min(phys_width, screen_width - 80)
        phys_height = min(phys_height, screen_height - 80)
        
        # Center on screen
        y_pos = (screen_height - phys_height) // 2
        x_pos = int(y_pos * (16 / 9))
        
        dpg.create_viewport(
            title='Spaudible',
            width=phys_width,
            height=phys_height,
            x_pos=x_pos,
            y_pos=y_pos,
            min_width=int(800 * self.dpi_scale),
            min_height=int(600 * self.dpi_scale)
        )
        
        self._build_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(self.window_tag, True)

    def _load_hidpi_font(self):
        """Load Spaudible Sans and Instrument Sans fonts with DPI-aware sizing."""
        # Updated path: data/gui/fonts/ instead of data/fonts/
        font_dir = Path(__file__).parent.parent.parent / "data" / "gui" / "fonts"
        regular_path = font_dir / "SpaudibleSans-Regular.ttf"
        semibold_path = font_dir / "InstrumentSans-SemiBold.ttf"
        
        # Calculate font size based on DPI (base 18pt * scale)
        font_size = int(18 * self.dpi_scale)
        
        with dpg.font_registry():
            # Load Regular as default
            if regular_path.exists():
                default_font = dpg.add_font(str(regular_path), font_size)
                dpg.bind_font(default_font)
                print(f"DEBUG: Loaded Spaudible Sans Regular at {font_size}px")
            else:
                # Fallback to default font with scale
                dpg.set_global_font_scale(self.dpi_scale)
                print("DEBUG: Spaudible Sans Regular not found; using default font")
            
            # Load SemiBold for headers
            if semibold_path.exists():
                self.header_font = dpg.add_font(str(semibold_path), font_size)
                print(f"DEBUG: Loaded Instrument Sans SemiBold at {font_size}px")
            else:
                self.header_font = None
                print(f"DEBUG: Instrument Sans SemiBold not found; using default font")

    def _build_ui(self):
        """Build the main application UI."""
        with dpg.window(tag=self.window_tag, label="Spaudible"):
            # Create background image first (behind everything)
            self.theme.create_background(self.window_tag)
            
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
                # Left sidebar - Settings (resizable horizontally)
                with dpg.child_window(
                    tag=self.settings_panel.tag,
                    width=self._s(300),
                    border=True,
                    autosize_x=False,
                    autosize_y=True,
                    resizable_x=True  # Enable horizontal resizing
                ):
                    self.settings_panel.build()
                
                # Right area - Search & Results (flexible width)
                with dpg.child_window(
                    tag=self.results_view.tag,
                    border=True,
                    autosize_x=True,
                    autosize_y=True
                ):
                    self.search_panel.build()
                    self.results_view.build()

    def _on_main_window_resize(self, sender, app_data):
        """Handle main window resize to keep right panel filling remaining space."""
        if not dpg.does_item_exist(self.settings_panel.tag):
            return
        
        # Get current panel widths
        left_width = dpg.get_item_rect_size(self.settings_panel.tag)[0]
        window_width = dpg.get_item_rect_size(self.window_tag)[0]
        
        # Calculate right panel width (window - left panel - borders)
        # The borders take up a few pixels on each side
        right_width = window_width - left_width - 2  # -2 for borders
        
        if right_width > 0:
            dpg.configure_item(self.results_view.tag, width=right_width)

    def _on_suggestion_selected(self, result: SearchResult):
        """Handle user selecting a song from the suggestion list.
        
        Starts the heavy similarity search in a background thread.
        """
        print(f"DEBUG: Selected track {result.track_id}: {result.track_name}")
        
        # Transition UI to similarity search mode
        self.search_panel.set_similarity_mode()
        
        # Clear previous results and show loading
        self.results_view.clear_results()
        self.results_view.show_loading(f"Finding songs similar to {result.track_name}...")
        
        # Start similarity search in background thread
        self._cancel_event.clear()
        self._similarity_thread = threading.Thread(
            target=self._run_similarity_search,
            args=(result.track_id, result.track_name),
            daemon=True
        )
        self._similarity_thread.start()

    def _run_similarity_search(self, track_id: str, track_name: str):
        """Run the heavy similarity search in a background thread.
        
        This performs the actual vector comparison against 256M tracks.
        """
        try:
            # Build vector for the selected track
            vector, track_data = build_canonical_vector(track_id)
            
            if vector is None or self._cancel_event.is_set():
                return
            
            # Initialize orchestrator with current settings from config
            use_gpu = not config_manager.get_force_cpu()
            force_cpu = config_manager.get_force_cpu()
            force_gpu = config_manager.get_force_gpu()
            
            self._orchestrator = SearchOrchestrator(
                vectors_path=str(PathConfig.get_vector_file()),
                index_path=str(PathConfig.get_index_file()),
                metadata_db=str(PathConfig.get_main_db()),
                use_gpu=use_gpu,
                force_cpu=force_cpu,
                force_gpu=force_gpu,
                skip_benchmark=True  # Skip auto-benchmark for quicker response
            )
            
            # Run the similarity search
            results = self._orchestrator.search(
                np.array(vector, dtype=np.float32),
                top_k=config_manager.get_top_k(),
                with_metadata=True,
                deduplicate=config_manager.get_deduplicate(),
                query_track_id=track_id,
                region_strength=config_manager.get_region_strength()
            )
            
            if not self._cancel_event.is_set():
                # Signal completion to main thread via flags
                self._search_results = results
                self._search_complete = True
                
        except Exception as e:
            print(f"Similarity search error: {e}")
            import traceback
            traceback.print_exc()
            if not self._cancel_event.is_set():
                self._search_error = str(e)
                self._search_error_flag = True
        finally:
            if self._orchestrator:
                try:
                    self._orchestrator.close()
                except Exception:
                    pass
                self._orchestrator = None

    def _on_similarity_complete(self, results):
        """Called on main thread when similarity search completes successfully."""
        self.results_view.update_results(results)
        self.search_panel.reset_to_idle()

    def _on_similarity_error(self, error_msg: str):
        """Called on main thread if similarity search fails."""
        self.results_view.show_loading(f"Search failed: {error_msg}")
        # Brief delay so user sees error before clearing
        import time
        time.sleep(0.1)
        self.results_view.hide_loading()
        self.search_panel.reset_to_idle()

    def _on_search_cancelled(self):
        """Handle user cancelling the search (from SearchPanel)."""
        print("DEBUG: Search cancelled by user")
        self._cancel_event.set()
        
        # Clean up orchestrator if running
        if self._orchestrator:
            try:
                self._orchestrator.close()
            except Exception:
                pass
            self._orchestrator = None
        
        # Clear results and reset UI
        self.results_view.clear_results()
        self.search_panel.reset_to_idle()

    def _main_loop(self):
        """Run the Dear PyGui render loop."""
        print("DEBUG: Entering render loop...")
        last_save_time = 0
        save_interval = 5.0
        prev_pos = dpg.get_viewport_pos()
        prev_size = [dpg.get_viewport_width(), dpg.get_viewport_height()]
        prev_left_width = None
        last_center_time = 0  # Track last centering update
        
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            
            # Check for text search completion from SearchPanel
            self.search_panel.update()
            
            # Check for completed search results from background thread
            if self._search_complete:
                self._on_similarity_complete(self._search_results)
                self._search_complete = False
                self._search_results = None
                
            # Check for search errors
            if self._search_error_flag:
                self._on_similarity_error(self._search_error)
                self._search_error_flag = False
                self._search_error = None
                
            # Update background size on viewport changes
            if self.theme:
                self.theme.update_background()
                
            # Update gradient button states each frame
            _gradient_factory.update_all_buttons()
            
            # Sync right panel when settings panel is resized by user
            try:
                if dpg.does_item_exist(self.settings_panel.tag):
                    current_left_width = dpg.get_item_rect_size(self.settings_panel.tag)[0]
                    if current_left_width != prev_left_width:
                        window_width = dpg.get_item_rect_size(self.window_tag)[0]
                        right_width = window_width - current_left_width - 2  # -2 for borders
                        if right_width > 0:
                            dpg.configure_item(self.results_view.tag, width=right_width)
                        prev_left_width = current_left_width
                        # Recenter results when panel resizes
                        if self.results_view:
                            self.results_view.update_centering()
            except Exception:
                pass  # Handle any errors gracefully during render loop
                
            # Check for window resize every frame (for centering), but save geometry only periodically
            current_time = time.time()
            current_pos = dpg.get_viewport_pos()
            current_size = [dpg.get_viewport_width(), dpg.get_viewport_height()]
            
            # Update centering immediately if window changed (throttle to ~30fps for performance)
            if (current_pos != prev_pos or current_size != prev_size):
                if current_time - last_center_time > 0.033:  # ~30fps max
                    if self.results_view:
                        self.results_view.update_centering()
                    last_center_time = current_time
                    
                # Save geometry only every 5 seconds
                if current_time - last_save_time > save_interval:
                    self._save_window_geometry()
                    prev_pos = current_pos
                    prev_size = current_size
                    last_save_time = current_time
                    
        print("DEBUG: Render loop exited")

    def _cleanup(self):
        """Save state and cleanup DPG resources."""
        if not self._is_context_created:
            return
        
        # Signal any running search to cancel
        self._cancel_event.set()
        
        try:
            # Save window geometry before destroying
            self._save_window_geometry()
        except Exception as e:
            print(f"⚠️ Error saving geometry: {e}")
        
        # Wait for search thread to finish if running
        if self._similarity_thread and self._similarity_thread.is_alive():
            self._similarity_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
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

    def _set_dpi_scale(self, scale: float):
        """Change DPI scale at runtime (requires restart)."""
        self.state_manager.set('dpi_scale', scale)
        
        if dpg.does_item_exist("restart_dialog"):
            dpg.delete_item("restart_dialog")
        
        # Get main window position and size
        win_pos = dpg.get_item_pos(self.window_tag)
        win_size = dpg.get_item_rect_size(self.window_tag)
        
        dialog_width = self._s(300)
        dialog_height = self._s(100)
        
        # Center within the main window
        center_x = int(win_pos[0] + (win_size[0] - dialog_width) // 2)
        center_y = int(win_pos[1] + (win_size[1] - dialog_height) // 2)
        
        def close_restart_dialog():
            if dpg.does_item_exist("restart_dialog"):
                dpg.delete_item("restart_dialog")
        
        with dpg.window(
            tag="restart_dialog",
            label="Restart Required",
            modal=True,
            width=dialog_width,
            height=dialog_height,
            pos=[center_x, center_y],
            no_resize=True
        ):
            dpg.add_text("UI scale will change on next restart.")
            add_gradient_button(
                label="OK",
                callback=close_restart_dialog
            )

    def _show_system_status(self, sender=None, app_data=None):
        """Show system status modal."""
        # TODO: Implement system status modal
        # This should port the CLI's _handle_system_status to GUI
        pass

    def _show_performance_test(self, sender=None, app_data=None):
        """Run/show performance test."""
        # TODO: Implement performance test modal
        # This should port the CLI's performance test to GUI
        pass

    def _check_updates(self, sender=None, app_data=None):
        """Check for updates."""
        # TODO: Implement update check
        pass

    def _show_about(self, sender=None, app_data=None):
        """Show about dialog centered on the main window."""
        # Delete existing dialog if it exists
        if dpg.does_item_exist("about_dialog"):
            dpg.delete_item("about_dialog")
        
        # Get main window position and size (NOT viewport)
        win_pos = dpg.get_item_pos(self.window_tag)
        win_size = dpg.get_item_rect_size(self.window_tag)
        
        dialog_width = self._s(400)
        dialog_height = self._s(300)
        
        # Center within the MAIN WINDOW, not the viewport
        center_x = int(win_pos[0] + (win_size[0] - dialog_width) // 2)
        center_y = int(win_pos[1] + (win_size[1] - dialog_height) // 2)
        
        # Debug output
        print(f"DEBUG: win_pos={win_pos}, win_size={win_size}, dialog_center=[{center_x}, {center_y}]")
        
        with dpg.window(
            tag="about_dialog",
            label="About Spaudible",
            modal=True,
            width=dialog_width,
            height=dialog_height,
            pos=[center_x, center_y],
            no_resize=True
        ):
            dpg.add_text("Spaudible v0.3.0", tag="about_title")
            if hasattr(self, 'header_font') and self.header_font:
                dpg.bind_item_font("about_title", self.header_font)
            dpg.add_separator()
            dpg.add_text("By Daveofthecave")
            
            def close_about_dialog():
                if dpg.does_item_exist("about_dialog"):
                    dpg.delete_item("about_dialog")
            
            add_gradient_button(
                label="Close",
                callback=close_about_dialog
            )

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
