# ui/cli/menu_system/database_check.py
"""Simplified setup flow with automatic download and build options."""
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
from ui.cli.console_utils import (
    print_header, print_menu, get_choice, format_elapsed_time, clear_screen
)
from config import PathConfig, DownloadConfig
from core.utilities.download_manager import SpaudibleDownloader, DownloadError
from core.utilities.extraction_manager import ZstExtractor, ExtractionError

def screen_database_check() -> str:
    """
    Entry point for setup flow.
    Routes to appropriate screen based on current system state.
    """
    # Check what's already present
    main_db = PathConfig.get_main_db()
    audio_db = PathConfig.get_audio_db()
    vectors = PathConfig.get_vector_file()
    index = PathConfig.get_index_file()
    query_marisa = PathConfig.get_query_marisa_file()
    query_postings = PathConfig.get_query_postings_file()
    
    has_dbs = main_db.exists() and audio_db.exists()
    has_vectors = vectors.exists() and index.exists()
    has_query = query_marisa.exists() and query_postings.exists()
    
    # Fully set up
    if has_dbs and has_vectors and has_query:
        return "main_menu"
    
    # Partial states - route to appropriate completion
    if has_dbs and not has_vectors:
        return screen_vectors_choice()
    
    if has_vectors and not has_dbs:
        return screen_databases_only_choice()

    if has_dbs and has_vectors and not has_query:
        return screen_query_index_choice()
    
    # Check for partial downloads (resumable)
    downloader = SpaudibleDownloader()
    if downloader.is_database_download_complete() and not has_dbs:
        # DBs downloaded but not extracted
        return screen_extraction_auto()
    
    # Fresh start
    return screen_welcome()

def screen_welcome() -> str:
    """
    Screen 1: Welcome screen showing required files and sizes.
    Waits for user to proceed to configuration.
    """
    clear_screen()
    print_header("Spaudible - Setup")
    
    # Calculate sizes from config
    db_files = DownloadConfig.DATABASE_FILES
    vec_files = DownloadConfig.VECTOR_FILES
    
    db_compressed = sum(compressed for _, compressed, _ in db_files)
    db_extracted = sum(extracted for _, _, extracted in db_files)
    
    print("\n  🎵 Welcome to Spaudible!\n")

    print("  In order to start searching for similar songs, several files")
    print("  need to be downloaded first. These files contain the song data")
    print("  that Spaudible uses to perform its similarity searches. Namely:\n")

    print("  2 databases with song metadata and audio features:")
    for fname, compressed, extracted in db_files:
        name = fname.replace('.zst', '')
        print(f"    • {name} ({compressed:.1f} GB compressed)")
    
    print("\n  2 track vector files that enable fast similarity searches:")
    for fname, subdir, size in vec_files:
        if fname in ['track_vectors.bin', 'track_index.bin']:
            print(f"    • {fname} ({size:.1f} GB)")
    
    print("\n  2 query index files that enable fast text-based song querying:")
    for fname, subdir, size in vec_files:
        if fname in ['inverted_index.bin', 'marisa_trie.bin']:
            print(f"    • {fname} ({size:.1f} GB)")
    
    try:
        input("\n  Press Enter to configure your options...")
    except KeyboardInterrupt:
        return "exit"
    return screen_setup_choice()

def screen_setup_choice() -> str:
    """Screen 2: Disk space verification and main choice."""
    clear_screen()
    print_header("Spaudible - Setup")
    
    # Calculate space requirements
    db_compressed = sum(c for _, c, _ in DownloadConfig.DATABASE_FILES)
    db_extracted = sum(e for _, _, e in DownloadConfig.DATABASE_FILES)
    vectors_size = sum(s for _, _, s in DownloadConfig.VECTOR_FILES)
    total_final = db_extracted + vectors_size
    temp_needed = db_compressed + db_extracted
    
    # Check available disk space
    try:
        available_gb = shutil.disk_usage(PathConfig.BASE_DIR).free / (1024**3)
    except Exception:
        available_gb = 0

    def print_space_info():
        print(f"\n  The required files will take up {math.ceil(total_final)} GB of disk space,")
        print(f"  with {math.ceil(temp_needed)} GB of temporary disk space needed for")
        print(f"  extracting the compressed downloaded databases.\n")
    
    print_space_info()
    
    # Insufficient space - show warning with option to proceed anyway
    if available_gb < total_final:
        shortfall = total_final - available_gb
        print(f"  ⚠️  You have {available_gb:.1f} GB of free space on your disk.")
        print(f"  Please free up at least {shortfall:.1f} GB before continuing the setup.\n")

        print("  Once done, please type 1 to verify that there's enough disk space.")
        print("  Or, if you wish to proceed without a disk space check, type 2.")
        
        options = [
            "Check disk space again",
            "Ignore disk space check and proceed to configuration screen",
            "Previous screen",
            "Exit program"
        ]
        print_menu(options)
        
        try:
            choice = get_choice(len(options))
        except KeyboardInterrupt:
            return "exit"
        
        if choice == 1:
            return screen_setup_choice()
        elif choice == 2:
            pass  # Fall through to configuration screen below
        elif choice == 3:
            return screen_welcome()
        else:
            return "exit"
    
    # Configuration screen (reached if sufficient space OR user chose to ignore)
    clear_screen()
    print_header("Spaudible - Setup")
    print_space_info()

    free_space_indicator = "✅" if available_gb >= total_final else "⚠️ "
    print(f"  {free_space_indicator} You have {available_gb:.1f} GB of free space on your disk.\n")
    
    print("  You can select [1] to download all 6 files automatically from their")
    print("  HuggingFace repositories. If your internet download speeds average")
    print("  around 100 Mbit/s, this process will take about 2 hours.\n")

    print("  Alternatively, you can select [2] to only download the 2 databases,")
    print("  after which Spaudible will build the remaining 4 vector files from")
    print("  scratch, a process that takes an additional 5-10 hours.\n")

    print("  What would you like to do?")
    
    options = [
        "Download all files (recommended)",
        "Download databases only and build vectors locally (slow)",
        "Previous screen",
        "Exit program"
    ]
    print_menu(options)
    
    try:
        choice = get_choice(len(options))
    except KeyboardInterrupt:
        return "exit"
    
    if choice == 1:
        return execute_download_all()
    elif choice == 2:
        return execute_build_locally()
    elif choice == 3:
        return screen_welcome()
    else:
        return "exit"

def execute_download_all() -> str:
    """
    Automatic download path: Downloads DBs, extracts, downloads vectors.
    No user input required until completion.
    """
    clear_screen()
    print_header("Spaudible - Downloading Files")
    start_time = time.time()
    total_data_gb = 0
    
    try:
        downloader = SpaudibleDownloader()
        
        # Step 1: Download databases
        print("\n  📥 Step 1/3: Downloading databases...\n")
        if not downloader.download_databases():
            raise Exception("Database download failed or was cancelled")
        
        for _, compressed, _ in DownloadConfig.DATABASE_FILES:
            total_data_gb += compressed
        
        elapsed = time.time() - start_time
        avg_mbps = (total_data_gb * 8000) / elapsed if elapsed > 0 else 0

        # Step 2: Extract databases
        print("\n  📦 Step 2/3: Extracting databases...\n")
        extractor = ZstExtractor()
        for fname, _, _ in DownloadConfig.DATABASE_FILES:
            zst_path = PathConfig.DATABASES / fname
            db_path = PathConfig.DATABASES / fname.replace('.zst', '')
            if zst_path.exists() and not db_path.exists():
                print(f"  Extracting {fname}...")
                extractor.extract_database(fname)
        
        # Cleanup archives to save space
        print("  Cleaning up compressed archives...")
        extractor.cleanup_archives()
        
        # Step 3: Download vector files
        print("\n  📥 Step 3/3: Downloading vector files...\n")
        if not downloader.download_vector_cache():
            raise Exception("Vector download failed or was cancelled")
        
        for _, _, size in DownloadConfig.VECTOR_FILES:
            total_data_gb += size
        
        # Success
        elapsed = time.time() - start_time
        
        print_header("Spaudible - Setup Complete!")
        print("\n  ✅ All files have been downloaded!\n")

        print(f"  • Total time taken: {format_elapsed_time(elapsed).strip()}")
        print(f"  • Average download speed: {avg_mbps:.1f} Mbit/s\n")

        print("  Spaudible is now ready to use.\n")

        print("  Press Enter to go to the main menu...")
        input()
        return "main_menu"
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Download interrupted. Progress has been saved.")
        print("  Run the program again to resume downloading.")
        input("\n  Press Enter to exit...")
        return "exit"
    
    except Exception as e:
        print(f"\n  ❌ Setup failed: {e}")
        input("\n  Press Enter to return to setup...")
        return "database_check"

def execute_build_locally() -> str:
    """
    Local build path: Downloads DBs, extracts, builds vectors and query index.
    No user input required until completion.
    """
    clear_screen()
    print_header("Spaudible - Building Locally")
    start_time = time.time()
    
    try:
        downloader = SpaudibleDownloader()
        
        # Step 1: Download databases
        print("\n  📥 Step 1/4: Downloading databases...")
        if not downloader.download_databases():
            raise Exception("Database download failed")
        
        # Step 2: Extract
        print("\n  📦 Step 2/4: Extracting databases...")
        extractor = ZstExtractor()
        for fname, _, _ in DownloadConfig.DATABASE_FILES:
            zst_path = PathConfig.DATABASES / fname
            if zst_path.exists():
                print(f"  Extracting {fname}...")
                extractor.extract_database(fname)
        extractor.cleanup_archives()
        
        # Step 3: Build vectors
        print("\n  ⚙️  Step 3/4: Building vector cache from databases...")
        print("  This process takes 1-4 hours depending on your hardware.")
        print("  Progress will update automatically. You can leave this running.\n")
        
        from core.preprocessing.db_to_vectors import PreprocessingEngine
        engine = PreprocessingEngine(
            main_db_path=str(PathConfig.get_main_db()),
            audio_db_path=str(PathConfig.get_audio_db()),
            output_dir=str(PathConfig.VECTORS),
            enable_profiling=False  # Silent mode
        )
        
        if not engine.run():
            raise Exception("Preprocessing failed")
        
        # Step 4: Build query index
        print("\n  🔍 Step 4/4: Building query index...")
        print("  This will take approximately 4-6 hours.\n")
        
        from core.preprocessing.querying.build_query_index import build_query_index
        build_query_index()
        
        # Success
        elapsed = time.time() - start_time
        print_header("Spaudible - Setup Complete!")
        print(f"\n  ✅ All vector/index files have been built!")
        print(f"  Total time taken: {format_elapsed_time(elapsed).strip()}")
        print(f"\n  Spaudible is now ready to use.")
        print("\n  Press any key to continue to the main menu...")
        input()
        return "main_menu"
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Build interrupted. You can resume by running the program again.")
        print("  The system will continue from where it left off.")
        input("\n  Press Enter to exit...")
        return "exit"
    except Exception as e:
        print(f"\n  ❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        input("\n  Press Enter to return to setup...")
        return "database_check"

def screen_vectors_choice() -> str:
    """
    Shown when databases exist but vectors are missing.
    Allows user to download pre-built vectors or build locally.
    """
    clear_screen()
    print_header("Spaudible - Setup")
    print("\n  ✅ Databases found! Vector cache needs setup.\n")
    print("  What would you like to do?")
    
    options = [
        "Download pre-built vectors (~5-30 minutes)",
        "Build vectors locally from databases (~5-10 hours)",
        "Exit program"
    ]
    print_menu(options)
    
    try:
        choice = get_choice(len(options))
    except KeyboardInterrupt:
        return "exit"
    
    if choice == 1:
        return execute_download_vectors_only()
    elif choice == 2:
        return execute_build_vectors_only()
    else:
        return "exit"

def execute_download_vectors_only() -> str:
    """Download only vector files (when DBs already exist)."""
    clear_screen()
    print_header("Spaudible - Downloading Vectors")
    start_time = time.time()
    
    try:
        downloader = SpaudibleDownloader()
        print("\n  📥 Downloading vector files...")
        
        if not downloader.download_vector_cache():
            raise Exception("Download failed")
        
        elapsed = time.time() - start_time
        print_header("Spaudible - Setup Complete!")
        print(f"\n  ✅ Vector files downloaded in {format_elapsed_time(elapsed).strip()}!")
        print("\n  Press any key to continue to the main menu...")
        input()
        return "main_menu"
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Download interrupted.")
        input("\n  Press Enter...")
        return "database_check"
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        input("\n  Press Enter...")
        return "database_check"

def execute_build_vectors_only() -> str:
    """Build vectors when databases already exist."""
    clear_screen()
    print_header("Spaudible - Building Vectors")
    start_time = time.time()
    
    try:
        print("\n  ⚙️  Building vector cache...")
        from core.preprocessing.db_to_vectors import PreprocessingEngine
        engine = PreprocessingEngine(
            main_db_path=str(PathConfig.get_main_db()),
            audio_db_path=str(PathConfig.get_audio_db()),
            output_dir=str(PathConfig.VECTORS),
            enable_profiling=False
        )
        
        if not engine.run():
            raise Exception("Preprocessing failed")
        
        print("\n  🔍 Building query index...")
        from core.preprocessing.querying.build_query_index import build_query_index
        build_query_index()
        
        elapsed = time.time() - start_time
        print_header("Spaudible - Setup Complete!")
        print(f"\n  ✅ Build complete! Time: {format_elapsed_time(elapsed).strip()}")
        print("\n  Press any key to continue...")
        input()
        return "main_menu"
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Build interrupted.")
        input("\n  Press Enter...")
        return "database_check"
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        input("\n  Press Enter...")
        return "database_check"

def screen_databases_only_choice() -> str:
    """
    Shown when vectors exist but databases don't (rare case).
    """
    clear_screen()
    print_header("Spaudible - Setup")
    print("\n  ⚠️  Vector cache found but databases are missing!")
    print("  The databases are required for metadata lookup and text search.")
    
    options = ["Download databases", "Exit program"]
    print_menu(options)
    
    choice = get_choice(len(options))
    if choice == 1:
        return execute_download_databases_only()
    return "exit"

def execute_download_databases_only() -> str:
    """Download databases when vectors already exist."""
    clear_screen()
    print_header("Spaudible - Downloading Databases")
    
    try:
        downloader = SpaudibleDownloader()
        print("\n  📥 Downloading databases...")
        
        if not downloader.download_databases():
            raise Exception("Download failed")
        
        print("\n  📦 Extracting...")
        extractor = ZstExtractor()
        for fname, _, _ in DownloadConfig.DATABASE_FILES:
            if (PathConfig.DATABASES / fname).exists():
                extractor.extract_database(fname)
        extractor.cleanup_archives()
        
        print_header("Spaudible - Setup Complete!")
        print("\n  ✅ Databases ready!")
        print("\n  Press any key to continue...")
        input()
        return "main_menu"
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        input("\n  Press Enter...")
        return "database_check"

def screen_query_index_choice() -> str:
    """Shown when everything exists except query index files."""
    clear_screen()
    print_header("Spaudible - Setup")
    print("\n  ⚠️  Query index files missing! Text search requires these files.")
    
    options = [
        "Download query index files (recommended)", 
        "Build query index locally (~4-6 hours)", 
        "Exit program"
    ]
    print_menu(options)
    
    try:
        choice = get_choice(len(options))
    except KeyboardInterrupt:
        return "exit"
        
    if choice == 1:
        return execute_download_query_index_only()
    elif choice == 2:
        return execute_build_query_index_only()
    else:
        return "exit"

def execute_download_query_index_only() -> str:
    """Download only query index files (when everything else exists)."""
    clear_screen()
    print_header("Spaudible - Downloading Query Index")
    start_time = time.time()
    
    try:
        downloader = SpaudibleDownloader()
        print("\n  📥 Downloading query index files...")
        
        # Download only query index files specifically
        from config import DownloadConfig
        query_files = [f for f in DownloadConfig.VECTOR_FILES if f[0] in ['inverted_index.bin', 'marisa_trie.bin']]
        
        for filename, subdir, size in query_files:
            state_key = f"vec_{filename}"
            if downloader.state["completed"].get(state_key):
                print(f"\n  ✅ {filename} already downloaded")
                continue
                
            print(f"\n  Downloading {filename} ({size} GB)...")
            if subdir:
                download_filename = f"{subdir}/{filename}"
            else:
                download_filename = filename
                
            downloader._download_file(
                repo_id=DownloadConfig.REPO_VECTORS,
                filename=download_filename,
                local_dir=PathConfig.VECTORS,
                state_key=state_key,
                repo_type="model"
            )
            print(f"  ✅ {filename} complete")
        
        elapsed = time.time() - start_time
        print_header("Spaudible - Setup Complete!")
        print(f"\n  ✅ Query index downloaded in {format_elapsed_time(elapsed).strip()}!")
        print("\n  Press any key to continue to the main menu...")
        input()
        return "main_menu"
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Download interrupted.")
        input("\n  Press Enter...")
        return "database_check"
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        input("\n  Press Enter...")
        return "database_check"

def execute_build_query_index_only() -> str:
    """Build query index when databases exist."""
    clear_screen()
    print_header("Spaudible - Setup")
    
    try:
        print("\n 🔍 Building query index from databases...")
        print(" This will take approximately 4-6 hours.\n")
        
        from core.preprocessing.querying.build_query_index import build_query_index
        build_query_index()
        
        print_header("Spaudible - Setup Complete!")
        print("\n ✅ Query index built successfully!")
        print("\n Press any key to continue...")
        input()
        return "main_menu"
        
    except KeyboardInterrupt:
        print("\n\n ⚠️ Build interrupted.")
        input("\n Press Enter...")
        return "database_check"
    except Exception as e:
        print(f"\n ❌ Error: {e}")
        input("\n Press Enter...")
        return "database_check"

def screen_extraction_auto() -> str:
    """
    Automatic extraction screen when downloads complete but not extracted.
    """
    print_header("Spaudible - Extracting Databases")
    
    try:
        extractor = ZstExtractor()
        print("\n  📦 Extracting downloaded databases...")
        
        for fname, _, _ in DownloadConfig.DATABASE_FILES:
            zst_path = PathConfig.DATABASES / fname
            if zst_path.exists():
                print(f"  Extracting {fname}...")
                extractor.extract_database(fname)
        
        print("  Cleaning up archives...")
        extractor.cleanup_archives()
        
        print("\n  ✅ Extraction complete!")
        print("  Checking for vector files...")
        
        # After extraction, check if we need vectors
        return screen_database_check()
        
    except Exception as e:
        print(f"\n  ❌ Extraction failed: {e}")
        input("\n  Press Enter...")
        return "database_check"
