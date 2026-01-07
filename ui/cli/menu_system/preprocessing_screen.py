# ui/cli/menu_system/preprocessing_screen.py
import sys
import time
from pathlib import Path
from ui.cli.console_utils import print_header
from core.preprocessing.db_to_vectors import PreprocessingEngine
from config import PathConfig

def screen_preprocessing():
    """Screen 3: Run the preprocessing engine."""
    print_header("Spaudible - Setup in Progress")
    
    try:
        print("\n  Initializing preprocessing engine...")
        
        # Create preprocessing engine with proper paths
        engine = PreprocessingEngine(
            main_db_path=str(PathConfig.get_main_db()),
            audio_db_path=str(PathConfig.get_audio_db()),
            output_dir=str(PathConfig.VECTORS)
        )
        
        # Run the preprocessing pipeline
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
