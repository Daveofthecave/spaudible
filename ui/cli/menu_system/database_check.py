# ui/cli/menu_system/database_check.py
"""Database check screen with intelligent routing to download/extraction/vector setup."""
from pathlib import Path
from ui.cli.console_utils import print_header, print_menu, get_choice
from config import PathConfig, DownloadConfig, EXPECTED_VECTORS
from core.utilities.setup_validator import validate_vector_cache, is_setup_complete, rebuild_index
from core.utilities.download_manager import SpaudibleDownloader
from core.utilities.extraction_manager import ZstExtractor
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

def check_query_index():
    """Check if query index files exist and are valid"""
    marisa_path = PathConfig.get_query_marisa_file()
    postings_path = PathConfig.get_query_postings_file()
    
    if not marisa_path.exists() or not postings_path.exists():
        return False, "Query index not found"
    
    # Check file sizes are reasonable
    marisa_size = marisa_path.stat().st_size
    postings_size = postings_path.stat().st_size
    
    if marisa_size < 10_000_000:
        return False, "MARISA trie too small"
    
    if postings_size < 500_000_000:
        return False, "Postings file too small"
    
    return True, f"Query index ready ({marisa_size / (1e9):.1f} GB + {postings_size / (1e9):.1f} GB)"

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
    """Comprehensive database check with intelligent routing."""
    print_header("Spaudible - System Check")

    # Determine current state
    has_main_db = PathConfig.get_main_db().exists()
    has_audio_db = PathConfig.get_audio_db().exists()
    dbs_extracted = has_main_db and has_audio_db
    has_zst_main = (PathConfig.DATABASES / "spotify_clean.sqlite3.zst").exists()
    has_zst_audio = (PathConfig.DATABASES / "spotify_clean_audio_features.sqlite3.zst").exists()
    has_any_zst = has_zst_main or has_zst_audio
    has_vectors = (PathConfig.get_vector_file().exists() and PathConfig.get_index_file().exists())

    # ROUTE 1: Nothing exists - need to download (show complete requirements)
    if not dbs_extracted and not has_any_zst and not has_vectors:
        missing = check_databases()
        if missing:
            print("\n  Welcome to Spaudible! Setup is required before first use.\n")
            print("  This application requires two types of data files to function:\n")
            print("1️⃣  Spotify Source Databases (Required for metadata & audio features)")
            print("  • spotify_clean.sqlite3 (Main database)")
            print("    - Contains track names, artists, albums, release dates")
            print("    - Download: ~36.7 GB → Extracts to: ~125 GB")
            print("  • spotify_clean_audio_features.sqlite3 (Audio analysis)")
            print("    - Contains tempo, energy, danceability, acousticness, etc.")
            print("    - Download: ~17.7 GB → Extracts to: ~42 GB\n")
            print("2️⃣  Vector Cache Files (Required for fast similarity search)")
            print("  • track_vectors.bin (26.6 GB)")
            print("    - Encoded representations of 256 million tracks for instant comparison")
            print("  • track_index.bin (6.7 GB)")
            print("    - Search index mapping track IDs to vector positions")
            print("  • Query Index files (4.9 GB)")
            print("    - Enables fast text search by song/artist name\n")
            # Calculate totals
            download_gb = DownloadConfig.get_required_space_gb(include_databases=True, include_vectors=True)
            extracted_gb = DownloadConfig.get_total_extracted_space_gb()
            print("  Disk Space Summary:")
            print(f"   Total download size: ~{download_gb:.1f} GB")
            print(f"   Space after extraction: ~{extracted_gb:.1f} GB")
            print(f"   (You can delete .zst archives after extraction to save space)\n")
            print("  You have two setup options:\n")
            print("  Option A: Download Everything (Recommended)")
            print("   • Downloads pre-built vector files (~30-60 minutes)")
            print("   • Ready to use immediately after download")
            print("   • No CPU processing required\n")
            print("  Option B: Download Databases Only + Build Locally")
            print("   • Downloads only Spotify databases (~2-3 hours)")
            print("   • Requires 5-10 hours of processing to build vector files")
            print("  Choose your preferred setup method:\n")
            options = [
                "Download everything automatically (fastest setup)",
                "Download databases only (build vectors locally)",
                "Check again (if manually placed files)",
                "Exit program"
            ]
            print_menu(options)
            choice = get_choice(len(options))
            if choice == 1:
                return "download_screen"
            elif choice == 2:
                return "download_screen"  # Will route to DB-only download flow
            elif choice == 3:
                return "database_check"
            else:
                return "exit"

    # ROUTE 2: Have .zst files, need extraction
    if has_any_zst:
        return "extraction_screen"

    # ROUTE 3a: DBs ready, need vectors
    if dbs_extracted and not has_vectors:
        print("\n✅ Databases found! Vector cache and index needs setup.\n")
        print("  What would you like to do?\n")
        print("  1. Download pre-built vectors (~30-60 minutes)")
        print("  2. Build locally from databases (~5-10 hours)")
        options = [
            "Download pre-built vectors",
            "Build locally",
            "Check again"
        ]
        print_menu(options)
        choice = get_choice(len(options))
        if choice == 1:
            return "download_vectors"
        elif choice == 2:
            return "preprocessing_prompt"
        else:
            return "database_check"

    # ROUTE 3b: Have vectors but no databases (incomplete manual setup)
    if has_vectors and not dbs_extracted:
        print("\n⚠️ Vector cache found, but Spotify databases are missing!\n")
        print("  The databases are required for:")
        print("   • Text search (query index)")
        print("   • Metadata lookup (song titles, artists)")
        print("   • Building new vectors from scratch\n")
        print("  You have vectors but need the source databases.\n")
        options = [
            "Download databases (required for full functionality)",
            "Delete vector files and start fresh",
            "Exit"
        ]
        print_menu(options)
        choice = get_choice(len(options))
        if choice == 1:
            return "download_screen"
        elif choice == 2:
            print("\n  Deleting vector cache to start fresh...")
            vector_files = [
                PathConfig.get_vector_file(),
                PathConfig.get_index_file()
            ]
            for file_path in vector_files:
                if file_path.exists():
                    file_path.unlink()
                    print(f"   Deleted: {file_path.name}")
            print("  Vectors deleted. Proceeding to database download.")
            input("\n  Press Enter to continue...")
            return "database_check"
        else:
            return "exit"

    # ROUTE 4: Everything exists - validate and show full original menu
    if dbs_extracted and has_vectors:
        is_valid, message = validate_vector_cache()
        if is_valid:
            print("\n✅ Spotify databases found!")
            print(f"  {message}\n")
            # Display file info (ORIGINAL)
            vectors_path = PathConfig.get_vector_file()
            index_path = PathConfig.get_index_file()
            try:
                vector_size = vectors_path.stat().st_size / (1e9)
                index_size = index_path.stat().st_size / (1e9)
                print(f"• Vector file: {vector_size:.1f} GB")
                print(f"• Index file: {index_size:.1f} GB")
            except FileNotFoundError:
                pass

            # Check query index (MODIFIED SECTION)
            print("\n  Checking query index...")
            query_ok, query_msg = check_query_index()
            if query_ok:
                print(f"✅ {query_msg}")
                print("\n  Would you like to:")
                options = ["Go to Main Menu", "Re-run preprocessing", "Rebuild query index", "Check again", "Exit"]
                print_menu(options)
                choice = get_choice(len(options))
                if choice == 1:
                    return "main_menu"
                elif choice == 2:
                    print("\n⚠️ This will delete existing processed files.")
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
                    print("\n🔧 Rebuilding query index...")
                    print("  This will take approximately 2 hours.")
                    confirm = input("  Proceed? (yes/no): ").lower().strip()
                    if confirm == 'yes':
                        try:
                            build_query_index()
                            print("\n✅ Query index rebuilt successfully!")
                            input("\n  Press Enter to continue...")
                        except Exception as e:
                            print(f"\n❌ Query index build failed: {e}")
                            input("\n  Press Enter to continue...")
                    return "database_check"
                elif choice == 4:
                    return "database_check"
                else:
                    return "exit"
            else:
                print(f"⚠️ {query_msg}")
                # Check if we can actually build the query index
                if not dbs_extracted:
                    print(f"\n⚠️ Cannot build query index: Spotify databases not found!")
                    print(f"  Query index requires access to track metadata (names, artists, albums).")
                    print(f"  Please ensure data/databases/ contains the Spotify databases.")
                    print("\n  Would you like to:")
                    options = ["Check again", "Exit"]
                    print_menu(options)
                    choice = get_choice(len(options))
                    if choice == 1:
                        return "database_check"
                    else:
                        return "exit"
                else:
                    print(f"  The query index enables fast text search.")
                    print(f"  You can download pre-built index files or build locally from databases.")
                    print("\n  Would you like to:")
                    options = ["Download query index", "Build query index locally", "Check again", "Exit"]
                    print_menu(options)
                    choice = get_choice(len(options))
                    if choice == 1:
                        return "download_vectors"
                    elif choice == 2:
                        print("\n🔧 Building query index...")
                        print("  This will take approximately 2 hours and use 3-4 GB of disk space.")
                        confirm = input("  Proceed? (yes/no): ").lower().strip()
                        if confirm == 'yes':
                            try:
                                build_query_index()
                                print("\n✅ Query index built successfully!")
                                input("\n  Press Enter to continue...")
                            except Exception as e:
                                print(f"\n❌ Query index build failed: {e}")
                                input("\n  Press Enter to continue...")
                        return "database_check"
                    elif choice == 3:
                        return "database_check"
                    else:
                        return "exit"
        else:
            # Vector cache invalid (ORIGINAL functionality with spacing fixes)
            print(f"\n❗ {message}")
            print("\n  You can:")
            options = [
                "Download pre-built vectors (recommended)",
                "Re-run preprocessing (build locally)",
                "Check again",
                "Exit"
            ]
            print_menu(options)
            choice = get_choice(len(options))
            if choice == 1:
                return "vector_choice"
            elif choice == 2:
                print("\n⚠️ This will delete existing processed files.")
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

    # Fallback: unclear state
    print("\n⚠️ System state unclear.")
    print("  Please check your data/ directory structure.")
    options = [
        "Check again",
        "Exit"
    ]
    print_menu(options)
    choice = get_choice(len(options))
    if choice == 1:
        return "database_check"
    else:
        return "exit"
