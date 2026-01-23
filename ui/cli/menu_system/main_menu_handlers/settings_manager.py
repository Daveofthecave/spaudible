# ui/cli/menu_system/main_menu_handlers/settings_manager.py
import numpy as np
import time
from config import VERSION, VRAM_SAFETY_FACTOR
from pathlib import Path
from ui.cli.console_utils import (
    print_header, 
    print_menu, 
    get_choice,
    format_elapsed_time,
    clear_screen
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
from core.utilities.config_manager import config_manager
from core.similarity_engine.vector_math import VectorOps

try:
    import torch
except ImportError:
    torch = None

def handle_settings() -> str:
    """Handle settings and tools menu."""
    print_header("Settings & Tools")

    # Get current settings
    force_cpu = config_manager.get_force_cpu()
    force_gpu = config_manager.get_force_gpu()
    algorithm_name = config_manager.get_algorithm_name()
    deduplicate = config_manager.get_deduplicate()
    region_strength = config_manager.get_region_strength()

    # Ensure mutual exclusivity
    if force_cpu and force_gpu:
        config_manager.set_force_gpu(False)
        force_gpu = False

    cpu_status = "ON" if force_cpu else "OFF"
    gpu_status = "ON" if force_gpu else "OFF"
    deduplicate_status = "ON" if deduplicate else "OFF"
    region_strength_str = f"{region_strength:.2f}"
    
    print("\n  ⚙️  Configuration & Diagnostics")
    
    options = [
        f"🐌 Force CPU Mode: {cpu_status}",
        f"🐆 Force GPU Mode: {gpu_status}",
        f"🧮 Select Similarity Algorithm: {algorithm_name}", 
        f"🧦 Deduplicate Results: {deduplicate_status}",
        f"🌎️ Region Filter Strength: {region_strength_str}",
        "⚖️  Adjust Feature Weights",
        "❔ Check System Status",
        "📊 Performance Test",
        "🔄 Re-run Setup",
        "ℹ️  About Spaudible",
        "⬅️  Back to Main Menu"
    ]
    
    print_menu(options)
    choice = get_choice(len(options))
    
    if choice == 1:
        return _force_cpu_mode()
    elif choice == 2:
        return _force_gpu_mode()
    elif choice == 3:
        return _select_algorithm()
    elif choice == 4:
        return _toggle_deduplicate()
    elif choice == 5:
        return _adjust_region_strength()
    elif choice == 6:
        return _adjust_feature_weights()
    elif choice == 7:
        return _handle_system_status()
    elif choice == 8:
        return _handle_performance_test()
    elif choice == 9:
        return _handle_rerun_setup()
    elif choice == 10:
        return _handle_about()
    else:
        return "main_menu"

def _force_cpu_mode() -> str:
    """Toggle CPU mode setting"""
    current = config_manager.get_force_cpu()
    new_setting = not current
    
    # Disable GPU mode if enabling CPU mode
    if new_setting:
        config_manager.set_force_gpu(False)
    
    config_manager.set_force_cpu(new_setting)
    
    # Clear benchmark cache
    SearchOrchestrator.clear_benchmark_cache()
    
    status = "ON" if new_setting else "OFF"
    print(f"\n  ✅ CPU mode set to: {status}")
    
    input("\n  Press Enter to continue...")
    return "settings"

def _force_gpu_mode() -> str:
    """Toggle GPU mode setting"""
    current = config_manager.get_force_gpu()
    new_setting = not current
    
    # Disable CPU mode if enabling GPU mode
    if new_setting:
        config_manager.set_force_cpu(False)
    
    config_manager.set_force_gpu(new_setting)
    
    # Clear benchmark cache
    SearchOrchestrator.clear_benchmark_cache()
    
    status = "ON" if new_setting else "OFF"
    print(f"\n  ✅ GPU mode set to: {status}")
    
    input("\n  Press Enter to continue...")
    return "settings"

def _select_algorithm() -> str:
    """Select similarity algorithm"""
    print_header("Select Similarity Algorithm")
    
    algorithms = config_manager.ALGORITHM_CHOICES
    current = config_manager.get_algorithm()
    
    print("\n  Available algorithms:\n")
    for i, (key, name) in enumerate(algorithms.items(), 1):
        current_indicator = " ← CURRENT" if key == current else ""
        print(f"  [{i}] {name}{current_indicator}")
    
    choice = get_choice(len(algorithms))
    selected_key = list(algorithms.keys())[choice-1]
    
    config_manager.set_algorithm(selected_key)
    print(f"\n  ✅ Algorithm set to: {algorithms[selected_key]}")
    
    input("\n  Press Enter to continue...")
    return "settings" 

def _toggle_deduplicate() -> str:
    """Toggle deduplication setting"""
    current = config_manager.get_deduplicate()
    new_setting = not current
    
    config_manager.set_deduplicate(new_setting)
    
    status = "ON" if new_setting else "OFF"
    print(f"\n  ✅ Deduplication set to: {status}")
    
    input("\n  Press Enter to continue...")
    return "settings"

def _adjust_region_strength() -> str:
    """Adjust region filter strength."""
    print_header("Adjust Region Filter Strength")
    
    current = config_manager.get_region_strength()
    print(f"\n  Current region filter strength: {current:.2f}")
    print("  (1.0 = Stick to the same region, 0.0 = Any region is ok)")
    
    while True:
        try:
            new_value = float(input("\n  Enter new strength [0.0-1.0]: "))
            if 0.0 <= new_value <= 1.0:
                config_manager.set_region_strength(new_value)
                print(f"\n  ✅ Region filter strength set to: {new_value:.2f}")
                input("\n  Press Enter to continue...")
                return "settings"
            else:
                print("  ❌ Value must be between 0.0 and 1.0")
        except ValueError:
            print("  ❌ Please enter a valid number")

def _adjust_feature_weights() -> str:
    """Adjust feature weights for similarity calculations."""
    print_header("Adjust Feature Weights")
    
    # Get current weights
    weights = config_manager.get_weights()
    
    # Feature names
    features = [
        "Acousticness", "Instrumentalness", "Speechiness", "Valence", "Danceability",
        "Energy", "Liveness", "Loudness", "Key", "Mode", "Tempo", "Time Signature 4/4",
        "Time Signature 3/4", "Time Signature 5/4", "Time Signature Other", "Duration",
        "Release Year", "Popularity", "Artist Followers", "Electronic & Dance", 
        "Rock & Alternative", "World & Traditional", "Latin", "Hip Hop & Rap", "Pop",
        "Classical & Art Music", "Jazz & Blues", "Christian & Religious", "Country & Folk",
        "R&B & Soul", "Reggae & Caribbean", "Other Genres"
    ]
    
    print("\n  Current feature weights:\n")
    for i, (feature, weight) in enumerate(zip(features, weights)):
        print(f"  {i+1:2d}. {feature:25} : {weight:.2f}")
    
    print("\n  Options:")
    print("  1. Edit individual weights")
    print("  2. Reset all weights to default")
    print("  3. Back to settings")
    
    choice = get_choice(3)
    
    if choice == 1:
        return _edit_weights(weights, features)
    elif choice == 2:
        config_manager.reset_weights()
        print("\n  ✅ All weights reset to 1.0")
        input("\n  Press Enter to continue...")
        return "settings"
    else:
        return "settings"

def _edit_weights(weights, features):
    """Edit individual feature weights."""
    while True:
        clear_screen()
        print_header("Edit Feature Weights")
        print("\n  Select a feature to adjust:\n")
        
        # Print features with current weights
        for i, (feature, weight) in enumerate(zip(features, weights)):
            print(f"  {i+1:2d}. {feature:25} : {weight:.2f}")
        
        print("\n  99. Save and return to settings")
        print("  00. Cancel without saving")
        
        try:
            choice = int(input("\n  Enter feature number: "))
            if choice == 99:
                config_manager.set_weights(weights)
                print("\n  ✅ Weights saved!")
                input("\n  Press Enter to continue...")
                return "settings"
            elif choice == 0:
                return "settings"
            elif 1 <= choice <= 32:
                feature_idx = choice - 1
                new_weight = float(input(f"  Enter new weight for '{features[feature_idx]}' (current: {weights[feature_idx]:.2f}): "))
                weights[feature_idx] = max(0.0, min(10.0, new_weight))  # Clamp to [0,10]
                print(f"  Updated {features[feature_idx]} weight to {weights[feature_idx]:.2f}")
                input("\n  Press Enter to continue...")
            else:
                print("  ❌ Invalid choice")
                time.sleep(1)
        except ValueError:
            print("  ❌ Please enter a valid number")
            time.sleep(1)       

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
    
    # Vector files (new unified format)
    vector_file = PathConfig.get_vector_file()
    vector_size = format_file_size(vector_file.stat().st_size) if vector_file.exists() else "Not found"
    print(f"   • Vector Cache: {vector_file.name}")
    print(f"       Size: {vector_size}")
    
    index_file = PathConfig.get_index_file()
    index_size = format_file_size(index_file.stat().st_size) if index_file.exists() else "Not found"
    print(f"   • Vector Index: {index_file.name}")
    print(f"       Size: {index_size}")
    
    # Note: track_masks.bin, track_regions.bin, and metadata.json 
    # are obsolete in the new format.
    # They are now embedded within the unified vector file.
    
    # Genre mapping
    genre_file = PathConfig.get_genre_mapping()
    genre_size = format_file_size(genre_file.stat().st_size) if genre_file.exists() else "Not found"
    print(f"   • Genre Mapping: {genre_file.name}")
    print(f"       Size: {genre_size}")

    # Total disk usage (only active files)
    total_size = 0
    files_to_check = [
        main_db, audio_db, vector_file, index_file, genre_file
    ]
    for file in files_to_check:
        if file.exists():
            total_size += file.stat().st_size
    print(f"\n  💾 Total Disk Usage: {format_file_size(total_size)}")            
    
    # Check canonical resolver
    try:
        build_canonical_vector("0eGsygTp906u18L0Oimnem")  # Test track
        print("\n  ✅ Canonical Track ID Resolver: Ready")
    except Exception as e:
        print(f"\n  ⚠️  Canonical Track ID Resolver: Error - {str(e)}")
    
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
    print("  Testing various chunk sizes with 1,000,000 vectors\n")
    
    cpu_chunk_sizes = [1_000_000, 750_000, 500_000, 300_000, 200_000, 150_000, 
                       125_000, 100_000, 75_000, 50_000, 30_000, 20_000, 
                       15_000, 10_000, 5_000]
    
    cpu_results = []
    cpu_orchestrator = SearchOrchestrator(
        skip_cpu_benchmark=True,
        skip_gpu_benchmark=True,
        use_gpu=False
    )
    
    # Initialize vector_ops for CPU orchestrator
    cpu_orchestrator.vector_ops = VectorOps(algorithm=config_manager.get_algorithm())
    cpu_orchestrator.vector_ops.set_user_weights(config_manager.get_weights())
    
    # Run CPU tests without progress bars
    for chunk_size in cpu_chunk_sizes:
        print(f"  Testing chunk size: {chunk_size:>9,}", end="", flush=True)
        
        # Create new ChunkedSearch with vector_ops
        cpu_orchestrator.chunked_search = ChunkedSearch(
            chunk_size,
            use_gpu=False,
            vector_ops=cpu_orchestrator.vector_ops
        )
        
        start_time = time.time()
        cpu_orchestrator.search(
            test_vector,
            max_vectors=1_000_000,
            show_progress=False
        )
        elapsed = time.time() - start_time
        speed = 1_000_000 / elapsed if elapsed > 0 else 0
        
        print(f" - {speed/1e6:.2f}M vec/sec")
        cpu_results.append((chunk_size, speed))
    
    # Find optimal CPU chunk size
    optimal_cpu_chunk, optimal_cpu_speed = max(cpu_results, key=lambda x: x[1])
    print(f"\n  Optimal CPU chunk size: {optimal_cpu_chunk:,} ({optimal_cpu_speed/1e6:.2f}M vec/sec)")
    
    # Section 2: GPU Batch Scaling
    print("\n" + "=" * 70)
    print("  🚀 GPU Batch Scaling Performance")
    print("=" * 70)
    
    gpu_results = []
    if torch.cuda.is_available():
        gpu_orchestrator = SearchOrchestrator(
            skip_cpu_benchmark=True,
            skip_gpu_benchmark=True,
            use_gpu=True
        )
        
        # Get max batch size from vector reader
        max_batch = gpu_orchestrator.vector_reader.get_max_batch_size()
        
        # Generate batch sizes
        batch_sizes = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 
                       50_000_000, 100_000_000, max_batch]
        batch_sizes = sorted(set([bs for bs in batch_sizes if bs <= max_batch]))
        
        print(f"  Testing batch sizes up to {max_batch:,} vectors\n")
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size:>10,}", end="", flush=True)
            
            # Create new ChunkedSearch with vector_ops
            gpu_orchestrator.chunked_search = ChunkedSearch(
                batch_size,
                use_gpu=True,
                vector_ops=gpu_orchestrator.vector_ops
            )
            
            start_time = time.time()
            gpu_orchestrator.search(
                test_vector,
                max_vectors=batch_size,
                show_progress=False
            )
            elapsed = time.time() - start_time
            speed = batch_size / elapsed if elapsed > 0 else 0
            
            print(f" - {speed/1e6:.2f}M vec/sec")
            gpu_results.append((batch_size, speed))
        
        # Find fastest GPU batch size
        if gpu_results:
            best_batch, best_speed = max(gpu_results, key=lambda x: x[1])
            print(f"\n  🚀 Fastest GPU batch: {best_batch:,} ({best_speed/1e6:.2f}M vec/sec)")
        gpu_orchestrator.close()
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
    
    print(f"\n  ✅ Found {len(results)} similar tracks in {format_elapsed_time(search_time).strip()}")
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
