# ui/cli/menu_system/main_menu_handlers/settings_manager.py
import os
import random
import time
from config import VERSION
from pathlib import Path
from ui.cli.console_utils import (
    print_header, 
    print_menu, 
    get_choice,
    format_elapsed_time
)
from .utils import (
    check_preprocessed_files,
    format_elapsed_time,
    get_metadata_db_path,
    format_file_size
)
from core.similarity_engine.orchestrator import SearchOrchestrator
from core.vectorization.canonical_track_resolver import build_canonical_vector
from config import PathConfig

def handle_settings() -> str:
    """Handle settings and tools menu."""
    print_header("Settings & Tools")
    
    print("\n  ⚙️  Configuration & Diagnostics")
    
    options = [
        "❔ Check System Status",
        "📊 Performance Test",
        "🔄 Re-run Setup",
        "ℹ️  About Spaudible",
        "⬅️  Back to Main Menu"
    ]
    
    print_menu(options)
    choice = get_choice(len(options))
    
    if choice == 1:
        return _handle_system_status()
    elif choice == 2:
        return _handle_performance_test()
    elif choice == 3:
        return _handle_rerun_setup()
    elif choice == 4:
        return _handle_about()
    else:
        return "main_menu"

def _handle_system_status() -> str:
    """Display comprehensive system status."""
    print_header("System Status")
    
    print("\n  📊 Database Metrics:\n")
    
    # Main Spotify database
    main_db = PathConfig.get_main_db()
    main_db_size = format_file_size(main_db.stat().st_size) if main_db.exists() else "Not found"
    print(f"   • Main Database: {main_db.name}")
    print(f"       Size: {main_db_size}")
    
    # Audio features database
    audio_db = PathConfig.get_audio_db()
    audio_db_size = format_file_size(audio_db.stat().st_size) if audio_db.exists() else "Not found"
    print(f"   • Audio Features Database: {audio_db.name}")
    print(f"       Size: {audio_db_size}")
    
    # Vector files
    vector_file = PathConfig.get_vector_file()
    vector_size = format_file_size(vector_file.stat().st_size) if vector_file.exists() else "Not found"
    print(f"   • Vector Cache: {vector_file.name}")
    print(f"       Size: {vector_size}")
    
    index_file = PathConfig.get_index_file()
    index_size = format_file_size(index_file.stat().st_size) if index_file.exists() else "Not found"
    print(f"   • Vector Index: {index_file.name}")
    print(f"      Size: {index_size}")
    
    metadata_file = PathConfig.get_metadata_file()
    metadata_size = format_file_size(metadata_file.stat().st_size) if metadata_file.exists() else "Not found"
    print(f"   • Vector Metadata: {metadata_file.name}")
    print(f"       Size: {metadata_size}")
    
    # Genre mapping
    genre_file = PathConfig.get_genre_mapping()
    genre_size = format_file_size(genre_file.stat().st_size) if genre_file.exists() else "Not found"
    print(f"   • Genre Mapping: {genre_file.name}")
    print(f"       Size: {genre_size}")

    # Total disk usage
    total_size = 0
    for file in [main_db, audio_db, vector_file, index_file, metadata_file, genre_file]:
        if file.exists():
            total_size += file.stat().st_size
    print(f"\n  💾 Total Disk Usage: {format_file_size(total_size)}")            
    
    # Check canonical resolver
    try:
        build_canonical_vector("0eGsygTp906u18L0Oimnem")  # Test track
        print("\n  ✅ Canonical Track ID Resolver: Ready")
    except Exception as e:
        print(f"\n  ⚠️  Canonical Track ID Resolver: Error - {str(e)}")
    
    # Check similarity engine
    try:
        orchestrator = SearchOrchestrator()
        orchestrator.close()
        print("  ✅ Similarity Engine: Ready")
    except Exception as e:
        print(f"  ⚠️  Similarity engine: Error - {str(e)}")
    
    input("\n  Press Enter to continue...")
    return "settings"

def _handle_performance_test() -> str:
    """Run performance tests."""
    print_header("Performance Test")
    
    # Check if files exist first
    files_exist, error_msg = check_preprocessed_files()
    if not files_exist:
        print(f"  ❌ {error_msg}")
        input("\n  Press Enter to continue...")
        return "settings"
    
    print("🧪 Running performance test...\n")
    
    try:
        orchestrator = SearchOrchestrator()
        test_vector = [random.uniform(-1, 1) for _ in range(32)]
        num_vectors = 10_000_000
        
#       print("\n   Testing sequential scan (10M vectors)...")
        start_time = time.time()
        results = orchestrator.search(
            query_vector=test_vector,
            search_mode="sequential",
            max_vectors=num_vectors,
            top_k=5,
            with_metadata=False
        )
        seq_time = time.time() - start_time

        # Estimate full vector cache scan time
        avg_speed = num_vectors / seq_time
        total_vectors = orchestrator.vector_reader.get_total_vectors()
        full_scan_seconds = total_vectors / avg_speed
        formatted_time = format_elapsed_time(full_scan_seconds)

        print(f"   Extrapolated time to scan all 256M vectors: {formatted_time}")

#       print(f"     Time: {format_elapsed_time(seq_time)}")
#       print(f"     Speed: {num_vectors/seq_time:,.0f} vectors/second")
        
#        print("\n  Testing progressive search...")
#        start_time = time.time()
#        results = orchestrator.search(
#            query_vector=test_vector,
#            search_mode="progressive",
#            top_k=5,
#            quality_threshold=0.95,
#            with_metadata=False
#        )

#        prog_time = time.time() - start_time
#        print(f"     Time: {format_elapsed_time(prog_time)}")
        
        orchestrator.close()
        print("\n  ✅ Performance test complete!")
        
    except Exception as e:
        print(f"  ❌ Performance test error: {e}")
    
    input("\n  Press Enter to continue...")
    return "settings"

def _handle_rerun_setup() -> str:
    """Handle re-running the setup process."""
    print_header("Re-run Setup")
    
    print("\n  ⚠️  This will delete existing processed files.")
    print("  You will need to re-run preprocessing (several hours).")
    
    confirm = input("\n  Are you sure? (yes/no): ").lower().strip()
    
    if confirm == 'yes':
        vector_files = [
            PathConfig.get_vector_file(),
            PathConfig.get_index_file(),
            PathConfig.VECTORS / "metadata.json"
        ]
        for file_path in vector_files:
            if file_path.exists():
                file_path.unlink()
        print("  ✅ Processed files removed.")
        print("  Please restart Spaudible to run setup.")
        input("\n  Press Enter to exit...")
        return "exit"
    
    return "settings"

def _handle_about() -> str:
    """Display about information."""
    print_header("About Spaudible")
    
    print("\n  Spaudible - Music Discovery Tool\n")

    print(f"  Version {VERSION}")
    print("  by Daveofthecave")
    
    input("\n  Press Enter to continue...")
    return "settings"
