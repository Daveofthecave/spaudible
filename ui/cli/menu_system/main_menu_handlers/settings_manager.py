# ui/cli/menu_system/main_menu_handlers/settings_manager.py
import numpy as np
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
from core.utilities.gpu_utils import get_gpu_info, print_gpu_info
from core.similarity_engine.vector_comparer import ChunkedSearch

try:
    import torch
except ImportError:
    torch = None

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
    
    # # Check similarity engine
    # try:
    #     orchestrator = SearchOrchestrator()
    #     orchestrator.close()
    #     print("  ✅ Similarity Engine: Ready")
    # except Exception as e:
    #     print(f"  ⚠️  Similarity engine: Error - {str(e)}")
    
    input("\n  Press Enter to continue...")
    return "settings"

def _handle_performance_test() -> str:
    """Run performance tests."""
    print_header("Performance Test")
    
    # Print GPU information
    print_gpu_info()
    
    # Check if files exist first
    files_exist, error_msg = check_preprocessed_files()
    if not files_exist:
        print(f"  ❌ {error_msg}")
        input("\n  Press Enter to continue...")
        return "settings"
    
    print("🧪 Running performance test...\n")
    
    # Test parameters
    test_vector = np.random.rand(32).astype(np.float32)
    test_track_id = "0eGsygTp906u18L0Oimnem"  # Sample track ID
    
    # Section 1: CPU Chunk Size Optimization
    print("=" * 70)
    print("  🔧 CPU Chunk Size Optimization")
    print("=" * 70)
    print("  Testing various chunk sizes with 500,000 vectors\n")
    
    cpu_chunk_sizes = [5_000, 10_000, 15_000, 20_000, 50_000, 
                       75_000, 100_000, 125_000, 150_000, 200_000, 
                       300_000, 400_000, 500_000]
    
    cpu_results = []
    cpu_orchestrator = SearchOrchestrator(
        use_gpu=False,
        skip_benchmark=True
    )
    
    # Run CPU tests without progress bars
    for chunk_size in cpu_chunk_sizes:
        print(f"  Testing chunk size: {chunk_size:>7,}", end="", flush=True)
        cpu_orchestrator.chunk_size = chunk_size
        cpu_orchestrator.chunked_search = ChunkedSearch(chunk_size, use_gpu=False)
        
        start_time = time.time()
        cpu_orchestrator.search(
            test_vector,
            max_vectors=500_000
        )
        elapsed = time.time() - start_time
        speed = 500_000 / elapsed if elapsed > 0 else 0
        
        print(f" - {speed/1e6:.2f}M vec/sec")
        cpu_results.append((chunk_size, speed))
    
    # Find optimal CPU chunk size
    optimal_cpu_chunk, optimal_cpu_speed = max(cpu_results, key=lambda x: x[1])
    print(f"\n  ✅ Optimal CPU chunk size: {optimal_cpu_chunk:,} ({optimal_cpu_speed/1e6:.2f}M vec/sec)")
    
    # Section 2: GPU Batch Scaling
    print("\n" + "=" * 70)
    print("  🚀 GPU Batch Scaling Performance")
    print("=" * 70)
    
    gpu_results = []
    if torch.cuda.is_available():
        # Determine max batch size based on VRAM
        gpu_info = get_gpu_info()
        if gpu_info:
            free_vram = gpu_info[0]['free_vram']
            bytes_per_vector = 32 * 4  # 32 floats * 4 bytes
            max_batch = int((free_vram * 0.8) // bytes_per_vector)
            
            # Generate batch sizes
            batch_sizes = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 
                           50_000_000, 100_000_000, max_batch]
            batch_sizes = sorted(set([bs for bs in batch_sizes if bs <= max_batch]))
            
            gpu_orchestrator = SearchOrchestrator(
                use_gpu=True,
                skip_benchmark=True
            )
            
            print(f"  Testing batch sizes up to {max_batch:,} vectors\n")
            
            for batch_size in batch_sizes:
                print(f"  Testing batch size: {batch_size:>10,}", end="", flush=True)
                gpu_orchestrator.chunk_size = batch_size
                gpu_orchestrator.chunked_search = ChunkedSearch(batch_size, use_gpu=True)
                
                start_time = time.time()
                gpu_orchestrator.search(
                    test_vector,
                    max_vectors=batch_size
                )
                elapsed = time.time() - start_time
                speed = batch_size / elapsed if elapsed > 0 else 0
                
                print(f" - {speed/1e6:.2f}M vec/sec")
                gpu_results.append((batch_size, speed))
            
            # Find fastest GPU batch size
            if gpu_results:
                best_batch, best_speed = max(gpu_results, key=lambda x: x[1])
                print(f"\n  🚀 Fastest GPU batch: {best_batch:,} ({best_speed/1e6:.2f}M vec/sec)")
        else:
            print("  ⚠️  Could not get GPU information")
    else:
        print("  ⚠️  No GPU available - skipping GPU tests")
    
    # Section 3: Track Search Performance
    print("\n" + "=" * 70)
    print("  🔍 Track Search Performance")
    print("=" * 70)
    
    track_vector, _ = build_canonical_vector(test_track_id)
    if track_vector is None:
        print("  ❗ Could not build test track vector")
        return "settings"
    
    # Use best configuration
    if gpu_results:
        # Use fastest GPU batch size
        best_batch, best_speed = max(gpu_results, key=lambda x: x[1])
        print(f"  Using GPU with batch size: {best_batch:,}")
        track_orchestrator = SearchOrchestrator(
            use_gpu=True,
            chunk_size=best_batch,
            skip_benchmark=True
        )
    else:
        # Use optimal CPU chunk size
        print(f"  Using CPU with chunk size: {optimal_cpu_chunk:,}")
        track_orchestrator = SearchOrchestrator(
            use_gpu=False,
            chunk_size=optimal_cpu_chunk,
            skip_benchmark=True
        )
    
    # Run track search
    print("\n  Searching for similar tracks...")
    start_time = time.time()
    results = track_orchestrator.search(
        track_vector,
        top_k=10,
        with_metadata=False
    )
    search_time = time.time() - start_time
    
    print(f"\n  ✅ Found {len(results)} similar tracks in {format_elapsed_time(search_time)}")
    track_orchestrator.close()
    
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
