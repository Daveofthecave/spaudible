# ui/cli/menu_system/database_check.py
from pathlib import Path
import json
import shutil
from ui.cli.console_utils import print_header, print_menu, get_choice
from config import PathConfig
from core.utilities.setup_validator import validate_vector_cache, is_setup_complete  # Import from shared module

def check_databases():
    """Check if required database files exist."""
    databases = [
        (PathConfig.get_main_db(), "Main database"),
        (PathConfig.get_audio_db(), "Audio features database")
    ]
    
    missing = []
    for db_path, description in databases:
        if not db_path.exists():
            missing.append((db_path.name, description))
    
    return missing

def screen_database_check():
    """Screen 1: Comprehensive database and vector cache validation."""
    print_header("Spaudible - System Check")

    # Check databases exist
    missing = check_databases()
    
    if missing:
        print("\n  ❗ Required database files are missing.\n")
        print("  To use Spaudible, you need to import these")
        print("  Spotify databases into the data/databases directory:\n")
        
        total_size = 0
        for filename, description in missing:
            if "main" in description.lower():
                size_info = "(~125 GB)"
                total_size += 125
            elif "audio" in description.lower():
                size_info = "(~42 GB)"
                total_size += 42
            else:
                size_info = ""
            
            print(f"    • {filename} {size_info} ({description})")
        
        if total_size > 0:
            print(f"\n  Total disk space required: ~{total_size} GB\n")

        print("  You can download the files from")
        print("  https://annas-archive.li/torrents/spotify")
        print("  (see README.md for more info).\n")
        
        print("  Once the files are in place, please press 1 to continue:")
        
        options = [
            "Re-check for database files",
            "Exit program"
        ]
        
        print_menu(options)
        choice = get_choice(len(options))
        
        if choice == 1:
            return "database_check"
        else:
            return "exit"
    
    # Validate vector cache
    is_valid, message = validate_vector_cache()
    
    if is_valid:
        print("\n  ✅ Spotify databases found!")
        print(f"  ✅ {message}\n")
        
        # Read metadata for display
        metadata_path = PathConfig.get_metadata_file()
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"     • Created at: {metadata.get('created_at', 'Unknown')}")
            print(f"     • Vector format: {metadata.get('vector_format', 'Unknown')}")
            print(f"     • ISRC coverage: {metadata.get('isrc_coverage', 'Unknown')}")
            print(f"     • Files: {', '.join(metadata.get('files', {}).values())}\n")
        except Exception as e:
            print(f"     ⚠️ Could not read metadata: {e}\n")
        
        # Offer options
        print("  Would you like to:")
        options = [
            "Go to Main Menu",
            "Re-run preprocessing",
            "Check again",
            "Exit"
        ]
        
        print_menu(options)
        choice = get_choice(len(options))
        
        if choice == 1:
            return "main_menu"
        elif choice == 2:
            print("\n  ⚠️  This will delete existing processed files.")
            confirm = input("  Are you sure? (yes/no): ").lower().strip()
            if confirm == 'yes':
                vector_files = [
                    PathConfig.get_vector_file(),
                    PathConfig.get_index_file(),
                    PathConfig.get_mask_file(),
                    PathConfig.get_metadata_file()
                ]
                for file_path in vector_files:
                    if file_path.exists():
                        file_path.unlink()
                return "preprocessing_prompt"
            else:
                return "database_check"
        elif choice == 3:
            return "database_check"
        else:
            return "exit"
    
    else:
        # Vector cache is invalid
        print("\n  ❗ Vector cache validation failed:")
        print(f"     {message}\n")
        
        print("  You can:")
        options = [
            "Re-run preprocessing (recommended)",
            "Check again",
            "Exit"
        ]
        
        print_menu(options)
        choice = get_choice(len(options))
        
        if choice == 1:
            print("\n  ⚠️  This will delete existing processed files.")
            confirm = input("  Are you sure? (yes/no): ").lower().strip()
            if confirm == 'yes':
                vector_files = [
                    PathConfig.get_vector_file(),
                    PathConfig.get_index_file(),
                    PathConfig.get_mask_file(),
                    PathConfig.get_metadata_file()
                ]
                for file_path in vector_files:
                    if file_path.exists():
                        file_path.unlink()
                return "preprocessing_prompt"
            else:
                return "database_check"
        elif choice == 2:
            return "database_check"
        else:
            return "exit"
