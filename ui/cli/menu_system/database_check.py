# ui/cli/menu_system/database_check.py
from pathlib import Path
import json
import shutil
from ui.cli.console_utils import print_header, print_menu, get_choice
from config import PathConfig

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

def check_processing_complete():
    """Check if processing is already complete."""
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    metadata_path = PathConfig.VECTORS / "metadata.json"
    
    if not (vectors_path.exists() and index_path.exists() and metadata_path.exists()):
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        tracks = metadata.get('total_tracks', 0)
        if tracks == 0:
            tracks = metadata.get('total_tracks_processed', 0)
        
        return tracks >= 256_000_000 * 0.95
    except:
        return False

def is_setup_complete():
    """Check if all required files exist."""
    return all(file.exists() for file in PathConfig.all_required_files())

def screen_database_check():
    """Screen 1: Check for required database files."""
    print_header("Spaudible - Database Check")

    if is_setup_complete():
        return "main_menu"
    
    missing = check_databases()
    
    if not missing:
        # All databases found - check if processing is complete
        is_complete = check_processing_complete()
        
        if is_complete:
            print("  ✅ Spotify databases found!")
            print("  ✅ Preprocessed track vectors found!\n")
            
            # Read metadata for display
            metadata_path = PathConfig.VECTORS / "metadata.json"
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                tracks = metadata.get('total_tracks', 0)
                if tracks == 0:
                    tracks = metadata.get('total_tracks_processed', 0)
                
                print(f"     • Tracks processed: {tracks:,}")
                print(f"     • Date created: {metadata.get('created_at', 'Unknown')}\n")
            except Exception as e:
                print(f"\n  ⚠️  Could not read metadata: {e}")
            
            print("  Would you like to:")
            
            options = [
                "Go to Main Menu",
                "Re-run preprocessing",
                "Check database files again",
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
                        PathConfig.VECTORS / "metadata.json"
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
            # Databases exist but processing is not complete
            print("\n  ✅ All required databases found!")
            print("\n  The system detected:")
            print("    • spotify_clean.sqlite3")
            print("    • spotify_clean_audio_features.sqlite3")
            
            # Check if any processing files exist
            vectors_exist = PathConfig.get_vector_file().exists()
            metadata_exist = (PathConfig.VECTORS / "metadata.json").exists()
            
            if vectors_exist or metadata_exist:
                print("\n  ⚠️  Partial or incomplete vector cache detected.")
                print("  You can:")
                
                options = [
                    "Start fresh (recommended)",
                    "Check processing status",
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
                            PathConfig.VECTORS / "metadata.json"
                        ]
                        for file_path in vector_files:
                            if file_path.exists():
                                file_path.unlink()
                    return "preprocessing_prompt"
                elif choice == 2:
                    if metadata_exist:
                        try:
                            metadata_path = PathConfig.VECTORS / "metadata.json"
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            tracks = metadata.get('total_tracks', 0)
                            if tracks == 0:
                                tracks = metadata.get('total_tracks_processed', 0)
                            
                            print(f"\n  📊 Current progress: {tracks:,} tracks")
                            print(f"     This is {(tracks / 256_000_000 * 100):.1f}% complete")
                        except Exception as e:
                            print(f"\n  ❗ Could not read processing metadata: {e}")
                    else:
                        print("\n  ❗ No processing metadata found")
                    input("\n  Press Enter to continue...")
                    return "database_check"
                else:
                    return "database_check"
            else:
                return "preprocessing_prompt"
    
    # Some databases are missing
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
        
        print(f"    • {filename} {size_info}")
    
    if total_size > 0:
        print(f"\n  Total disk space required: ~{total_size} GB\n")

    print("  You can download the files from")
    print("  https://annas-archive.org/torrents/spotify")
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
        print("\n  Goodbye! Place the database files in data/databases and restart.")
        return "exit"
