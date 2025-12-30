# ui/cli/menu_system/processing_complete.py
from ui.cli.console_utils import print_header, print_menu, get_choice
from pathlib import Path
import json
from config import PathConfig

def check_processing_status():
    """Check if processing is complete and valid."""
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    metadata_path = PathConfig.VECTORS / "metadata.json"
    
    if not (vectors_path.exists() and index_path.exists() and metadata_path.exists()):
        return False, "Processing files not found"
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        expected_tracks = 256_000_000
        actual_tracks = metadata.get('total_tracks', 0)
        
        if actual_tracks < expected_tracks * 0.95:
            return False, f"Incomplete processing: {actual_tracks:,} tracks"
        
        return True, f"Processing complete: {actual_tracks:,} tracks"
    except Exception as e:
        return False, f"Metadata error: {str(e)}"

def get_file_size_gb(file_path):
    """Get file size in GB."""
    if file_path.exists():
        return file_path.stat().st_size / (1024**3)
    return 0.0

def screen_processing_complete():
    """Screen 4: Processing completion confirmation."""
    print_header("Spaudible - Setup Complete")
    
    is_complete, message = check_processing_status()
    
    if is_complete:
        print(f"\n ✅ {message}")
        
        # Read metadata
        metadata_path = PathConfig.VECTORS / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Calculate file sizes
        vectors_path = PathConfig.get_vector_file()
        index_path = PathConfig.get_index_file()
        
        vectors_gb = get_file_size_gb(vectors_path)
        index_gb = get_file_size_gb(index_path)
        total_gb = vectors_gb + index_gb
        
        print("\n 📊 Processing Statistics:")
        print(f"    • Total tracks: {metadata.get('total_tracks', 0):,}")
        print(f"    • Vector cache: {vectors_gb:.1f} GB")
        print(f"    • Vector index: {index_gb:.1f} GB")
        print(f"    • Total size: {total_gb:.1f} GB")
        print(f"    • Created: {metadata.get('created_at', 'Unknown')}")
        
        print("\n     Ready for similarity searching!")
        
        options = [
            "Continue to Main Menu",
            "Validate Database Integrity",
            "Return to Setup Check"
        ]
        
        print_menu(options)
        choice = get_choice(len(options))
        
        if choice == 1:
            return "main_menu"
        elif choice == 2:
            print("\n  🔍 Running quick integrity check...")
            expected_bytes = metadata.get('total_tracks', 0) * 128
            actual_bytes = vectors_path.stat().st_size if vectors_path.exists() else 0
            
            if expected_bytes > 0:
                match_percent = (actual_bytes / expected_bytes) * 100
                print(f"  Vector file integrity: {match_percent:.1f}% of expected size")
                if match_percent < 95:
                    print("  ⚠️  Vector file may be incomplete")
                else:
                    print("  ✅ Vector file appears complete")
            
            input("\n  Press Enter to continue...")
            return "processing_complete"
        else:
            return "database_check"
    
    else:
        print(f"\n  ❗ {message}")
        print("\n  Setup appears incomplete or corrupted.")
        
        options = [
            "Re-run preprocessing",
            "Check database files",
            "Exit"
        ]
        
        print_menu(options)
        choice = get_choice(len(options))
        
        if choice == 1:
            import shutil
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
            return "database_check"
        else:
            return "exit"
