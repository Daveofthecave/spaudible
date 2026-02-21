# main.py
import os
import sys
import time
import torch
import argparse
from config import PathConfig, FRAME_WIDTH
from pathlib import Path
from ui.cli.console_utils import clear_screen
from ui.cli.menu_system.database_check import screen_database_check
from ui.cli.menu_system.preprocessing_prompt import screen_preprocessing_prompt
from ui.cli.menu_system.preprocessing_screen import screen_preprocessing
from ui.cli.menu_system.processing_complete import screen_processing_complete
from ui.cli.menu_system.main_menu import screen_main_menu
from ui.cli.menu_system.download_screen import screen_download_databases, screen_download_vectors
from ui.cli.menu_system.extraction_screen import screen_extraction
from ui.cli.menu_system.vector_choice_screen import screen_vector_choice
from core.utilities.setup_validator import is_setup_complete

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

def get_gpu_info():
    """Get GPU information if available"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"GPU: {device} ({mem:.1f}GB VRAM)"
    return "No GPU detected"

def main():
    """Main entry point for Spaudible with optional GUI mode."""
    # Parse command-line arguments to determine UI mode
    parser = argparse.ArgumentParser(description="Spaudible")
    parser.add_argument(
        '--gui', 
        action='store_true', 
        help='Launch in GUI mode'
    )
    parser.add_argument(
        '--cli', 
        action='store_true', 
        help='Launch in CLI mode'
    )
    args = parser.parse_args()
    
    # Determine UI mode: CLI if explicitly requested, otherwise GUI
    # Default to CLI for now to maintain existing behavior until GUI is fully ready
    use_gui = args.gui and not args.cli
    use_gui = True  # REMOVE ME; THIS LINE IS ONLY FOR TESTING
    
    if use_gui:
        # Launch GUI mode
        try:
            from ui.gui.main_window import MainWindow
            # Note: No need to instantiate GUIStateManager here; MainWindow handles it internally
            
            print(f"\n  System Info: {get_gpu_info()}")
            print("  Launching Spaudible GUI...")
            
            # Create and run main window without parameters
            main_window = MainWindow()
            main_window.run()
            
            return  # Exit after GUI closes
            
        except ImportError as e:
            print(f"\n  ❗️ GUI dependencies not available: {e}")
            print("  Falling back to CLI mode...")
            use_gui = False
            time.sleep(7)
        except Exception as e:
            print(f"\n  ❗️ Failed to launch GUI: {e}")
            import traceback
            traceback.print_exc()
            print("  Falling back to CLI mode...")
            use_gui = False
            time.sleep(7)
    
    # Original CLI logic
    print(f"\n  System Info: {get_gpu_info()}")
    
    if is_setup_complete():
        current_screen = "main_menu"
    else:
        current_screen = "database_check"
    
    while True:
        clear_screen()
        
        if current_screen == "database_check":
            next_screen = screen_database_check()
            current_screen = next_screen
        elif current_screen == "download_screen":
            next_screen = screen_download_databases()
            current_screen = next_screen
        elif current_screen == "extraction_screen":
            next_screen = screen_extraction()
            current_screen = next_screen
        elif current_screen == "vector_choice":
            next_screen = screen_vector_choice()
            current_screen = next_screen
        elif current_screen == "download_vectors":
            next_screen = screen_download_vectors()
            current_screen = next_screen    
        elif current_screen == "preprocessing_prompt":
            next_screen = screen_preprocessing_prompt()
            current_screen = next_screen
        elif current_screen == "start_preprocessing":
            next_screen = screen_preprocessing()
            current_screen = next_screen
        elif current_screen == "processing_complete":
            next_screen = screen_processing_complete()
            current_screen = next_screen
        elif current_screen == "main_menu":
            next_screen = screen_main_menu()
            current_screen = next_screen
        elif current_screen == "core_search":
            from ui.cli.menu_system.main_menu_handlers.core_search import handle_core_search
            next_screen = handle_core_search()
            current_screen = next_screen
        elif current_screen == "settings":
            from ui.cli.menu_system.main_menu_handlers.settings_manager import handle_settings
            next_screen = handle_settings()
            current_screen = next_screen
        elif current_screen == "exit":
            print("\n" + "═" * FRAME_WIDTH)
            print("  Thank you for using Spaudible!\n")
            break
        else:
            print(f"\n  Error: Unknown screen '{current_screen}'")
            print("  Returning to database check...")
            time.sleep(2)
            current_screen = "database_check"

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
