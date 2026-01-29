# ui/cli/menu_system/processing_complete.py
from ui.cli.console_utils import print_header, print_menu, get_choice
from pathlib import Path
import json
from config import PathConfig, EXPECTED_VECTORS
from core.utilities.setup_validator import validate_vector_cache

def check_processing_status():
    """Check if processing is complete and valid."""
    vectors_path = PathConfig.get_vector_file()
    index_path = PathConfig.get_index_file()
    
    # Check essential files exist
    if not vectors_path.exists():
        return False, "Vector file not found"
    if not index_path.exists():
        return False, "Index file not found"
    
    # Calculate actual track count from file size
    vectors_size = vectors_path.stat().st_size
    header_size = 16
    record_size = 104
    actual_tracks = (vectors_size - header_size) // record_size
    
    # Verify the correct number of vectors are present
    if actual_tracks != EXPECTED_VECTORS:
        return False, f"Incomplete processing: got {actual_tracks:,} tracks; expected {EXPECTED_VECTORS}."
    
    return True, f"Processing complete: {actual_tracks:,} tracks"

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
        
        # Calculate file sizes
        vectors_path = PathConfig.get_vector_file()
        index_path = PathConfig.get_index_file()
        
        vectors_gb = get_file_size_gb(vectors_path)
        index_gb = get_file_size_gb(index_path)
        total_gb = vectors_gb + index_gb
        
        # Calculate actual tracks from file size
        vectors_size = vectors_path.stat().st_size
        header_size = 16
        record_size = 104
        actual_tracks = (vectors_size - header_size) // record_size
        
        print("\n 📊 Processing Statistics:")
        print(f"    • Total tracks: {actual_tracks:,}")
        print(f"    • Vector cache: {vectors_gb:.1f} GB")
        print(f"    • Vector index: {index_gb:.1f} GB")
        print(f"    • Total size: {total_gb:.1f} GB")
        print(f"    • Created: Unknown")  # metadata.json not created by engine
        
        print("\n  🔍 Running final integrity validation...")
        is_valid, validation_msg = validate_vector_cache(checksum_validation=True)
        if is_valid:
            print(f"  ✅ {validation_msg}")
        else:
            print(f"  ⚠️  {validation_msg}")
        
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
            # Validate by checking file size match
            print("\n  🔍 Running quick integrity check...")
            expected_bytes = actual_tracks * record_size + header_size
            actual_bytes = vectors_path.stat().st_size
            
            if actual_bytes == expected_bytes:
                print("  ✅ Vector file size matches expected")
            else:
                print(f"  ⚠️  Size mismatch: {actual_bytes:,} vs {expected_bytes:,}")
            
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
            import os
            vector_files = [
                PathConfig.get_vector_file(),
                PathConfig.get_index_file()
            ]
            for file_path in vector_files:
                if file_path.exists():
                    os.remove(file_path)
            
            # Also clean up metadata.json if it exists from previous runs
            metadata_path = PathConfig.VECTORS / "metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
                
            return "preprocessing_prompt"
        elif choice == 2:
            return "database_check"
        else:
            return "exit"
