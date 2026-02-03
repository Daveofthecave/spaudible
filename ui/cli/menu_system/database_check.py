# ui/cli/menu_system/database_check.py
from pathlib import Path
import json
import shutil
from ui.cli.console_utils import print_header, print_menu, get_choice
from config import PathConfig
from core.utilities.setup_validator import validate_vector_cache, is_setup_complete, rebuild_index
from config import EXPECTED_VECTORS

# Import query index builder
from core.preprocessing.querying.build_query_index import build_query_index

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

# Check query index
def check_query_index():
    """Check if query index files exist and are valid"""
    marisa_path = PathConfig.get_query_marisa_file()
    postings_path = PathConfig.get_query_postings_file()
    
    if not marisa_path.exists() or not postings_path.exists():
        return False, "Query index not found"
    
    # Check file sizes are reasonable
    marisa_size = marisa_path.stat().st_size
    postings_size = postings_path.stat().st_size
    
    if marisa_size < 10_000_000:  # Should be ~1-2GB
        return False, "MARISA trie too small"
    if postings_size < 500_000_000:  # Should be ~2-4GB
        return False, "Postings file too small"
    
    return True, f"Query index ready ({marisa_size / (1024**3):.1f} GB + {postings_size / (1024**3):.1f} GB)"

def get_file_size(path):
    """Safely get file size, returning 0 if file doesn't exist."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0

# Get query index size for display
def get_query_index_size():
    """Get total size of query index files."""
    marisa_path = PathConfig.get_query_marisa_file()
    postings_path = PathConfig.get_query_postings_file()
    
    if not marisa_path.exists() or not postings_path.exists():
        return 0
    
    return marisa_path.stat().st_size + postings_path.stat().st_size

def screen_database_check():
    """Screen 1: Comprehensive database and vector cache validation."""
    print_header("Spaudible - System Check")

    # Check databases exist
    missing = check_databases()
    
    if missing:
        print("\n  ❗ Required database files are missing.\n")
        print("  To use Spaudible, you need to import these")
        print("  Spotify databases into the data/databases/ directory:\n")
        
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
        print("  [coming soon!]")
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
        
        # Display basic file info safely
        vectors_path = PathConfig.get_vector_file()
        index_path = PathConfig.get_index_file()
        
        try:
            vector_size = vectors_path.stat().st_size / (1024**3)
            index_size = index_path.stat().st_size / (1024**3)
            
            print(f"     • Vector file: {vector_size:.1f} GB")
            print(f"     • Index file: {index_size:.1f} GB")
        except FileNotFoundError as e:
            print(f"  ⚠️  File not found: {e}")
            print("  This indicates files were deleted after validation")
        
        # Check query index
        print("\n  🔍 Checking query index...")
        query_ok, query_msg = check_query_index()
        
        if query_ok:
            print(f"  ✅ {query_msg}")
            query_options = [
                "Go to Main Menu",
                "Re-run preprocessing",
                "Rebuild query index",
                "Check again",
                "Exit"
            ]
        else:
            print(f"  ⚠️  {query_msg}")
            print("  The query index enables fast text search (3-4 GB, ~3-5 hour build)")
            query_options = [
                "Go to Main Menu",
                "Re-run preprocessing",
                "Build query index",
                "Check again",
                "Exit"
            ]
        
        print("\n  Would you like to:")
        print_menu(query_options)
        choice = get_choice(len(query_options))
        
        if choice == 1:
            return "main_menu"
        elif choice == 2:
            print("\n  ⚠️  This will delete existing processed files.")
            confirm = input("  Are you sure? (yes/no): ").lower().strip()
            if confirm == 'yes':
                vector_files = [
                    PathConfig.get_vector_file(),
                    PathConfig.get_index_file()
                ]
                for file_path in vector_files:
                    if file_path.exists():
                        file_path.unlink()
                return "preprocessing_prompt"
            else:
                return "database_check"
        elif choice == 3:
            # Build or rebuild query index
            print("\n  🔧 Building query index...")
            print("  This will take approximately 2 hours and use 3-4 GB of disk space.")
            confirm = input("  Proceed? (yes/no): ").lower().strip()
            if confirm == 'yes':
                try:
                    build_query_index()
                    print("\n  ✅ Query index built successfully!")
                    input("\n  Press Enter to continue...")
                except Exception as e:
                    print(f"\n  ❌ Query index build failed: {e}")
                    input("\n  Press Enter to continue...")
            return "database_check"
        elif choice == 4:
            return "database_check"
        else:
            return "exit"
    
    else:
        # Vector cache is invalid
        print("\n  ❗ Vector cache validation failed:")
        print(f"     {message}\n")
        
        # Check if vectors file exists
        vectors_path = PathConfig.get_vector_file()
        if vectors_path.exists():
            try:
                vector_size = vectors_path.stat().st_size
                header_size = 16
                record_size = 104
                num_vectors = (vector_size - header_size) // record_size
                
                # Verify exact vector count
                if num_vectors == EXPECTED_VECTORS:
                    print(f"  ✅ Vector file complete with {EXPECTED_VECTORS:,} vectors")
                    print("  You can rebuild the index file instead of reprocessing all vectors")
                    
                    options = [
                        "Rebuild index file",
                        "Re-run full preprocessing",
                        "Check again",
                        "Exit"
                    ]
                    
                    print_menu(options)
                    choice = get_choice(len(options))
                    
                    if choice == 1:
                        success = rebuild_index()
                        if success:
                            print("\n  ✅ Index file successfully rebuilt!")
                            print("  Press Enter to return to main menu...")
                            input()
                            return "main_menu"
                        else:
                            print("\n  ❗ Failed to rebuild index")
                            print("  Press Enter to try again...")
                            input()
                            return "database_check"
                    elif choice == 2:
                        print("\n  ⚠️  This will delete existing processed files.")
                        confirm = input("  Are you sure? (yes/no): ").lower().strip()
                        if confirm == 'yes':
                            vector_files = [
                                PathConfig.get_vector_file(),
                                PathConfig.get_index_file()
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
                    print(f"  ❗ Vector file has {num_vectors:,} vectors, expected {EXPECTED_VECTORS:,}")
            except FileNotFoundError:
                print("  ❗ Vector file not found")
        else:
            print("  ❗ Vector file not found")
        
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
                    PathConfig.get_index_file()
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
