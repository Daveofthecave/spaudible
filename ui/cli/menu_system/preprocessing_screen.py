# ui/cli/menu_system/preprocessing_screen.py
from pathlib import Path
from ui.cli.console_utils import print_header
from core.preprocessing.db_to_vectors import PreprocessingEngine
from config import PathConfig
import sys

def screen_preprocessing():
    """Screen 3: Run the preprocessing engine with profiling options."""
    print_header("Spaudible - Setup in Progress")
    
    # Ask user about profiling
    enable_profiling = False
    profile_interval = 4_000_000
    
    print("\n  ⚙️  Performance Profiling Options:")
    print("     1. No profiling (fastest)")
    print("     2. Full profiling (save at end)")
    print("     3. Periodic profiling (save every N million vectors)")
    
    choice = input("\n  Choose profiling option (1-3): ").strip()
    
    if choice == '2':
        enable_profiling = True
        print("  ✅ Full profiling enabled")
    elif choice == '3':
        enable_profiling = True
        try:
            interval = int(input("  Enter profile interval (millions of vectors, eg. 4): "))
            profile_interval = interval * 1_000_000
            print(f"  ✅ Periodic profiling every {profile_interval:,} vectors")
        except ValueError:
            print("  ❗ Invalid input, using default 4 million vectors")
            profile_interval = 4_000_000
    else:
        print("  ⏭️ Profiling disabled")
    
    try:
        print("\n  Initializing preprocessing engine...")
        
        # Create preprocessing engine with profiling options
        engine = PreprocessingEngine(
            main_db_path=str(PathConfig.get_main_db()),
            audio_db_path=str(PathConfig.get_audio_db()),
            output_dir=str(PathConfig.VECTORS),
            enable_profiling=enable_profiling,
            profile_interval=profile_interval
        )
        
        # Run preprocessing
        success = engine.run()
        
        if success:
            print("\n  ✅ Preprocessing completed successfully!")
            print("  Your music database is ready for searching.")
            print("\n  Press Enter to continue...")
            input()
            return "processing_complete"
        else:
            print("\n  ❗ Preprocessing failed or was interrupted.")
            print("  Please check the error messages above.")
            print("\n  Press Enter to return to database check...")
            input()
            return "database_check"
        
    except Exception as e:
        print(f"\n  ❗ Unexpected error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Press Enter to return to database check...")
        input()
        return "database_check"
