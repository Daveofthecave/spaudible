# main.py
import os
import sys
import time
import torch
from config import PathConfig
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
            print("\n" + "═" * 65)
            print("  Thank you for using Spaudible!")
            print("  Exiting in 2 seconds...")
            time.sleep(2)
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
