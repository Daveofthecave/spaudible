# ui/cli/menu_system/vector_choice_screen.py
"""Screen for choosing between downloading pre-built vectors or building locally."""
import shutil
import sys
from pathlib import Path
from typing import Optional
from ui.cli.console_utils import print_header, format_elapsed_time
from core.utilities.setup_validator import validate_vector_cache
from config import PathConfig, DownloadConfig

def screen_vector_choice() -> str:
    """Screen for choosing vector cache setup method."""
    print_header("Set Up Vector Cache")
    
    # Check if vectors already exist
    has_vectors = (
        PathConfig.get_vector_file().exists() 
        and PathConfig.get_index_file().exists()
        and PathConfig.get_vector_file().stat().st_size > 1_000_000_000
    )
    
    if has_vectors:
        is_valid, msg = validate_vector_cache(checksum_validation=False)
        if is_valid:
            print("\n ✅ Vector cache already exists and is valid!")
            input("\n Press Enter to continue to main menu...")
            return "main_menu"
        else:
            print("\n ⚠️ Existing vector cache appears incomplete.")
    
    # Calculate requirements
    download_gb = DownloadConfig.get_required_space_gb(
        include_databases=False, include_vectors=True
    )
    
    print("\n The vector cache enables fast similarity searching.")
    print(" You have two options:\n")
    
    print(f" [1] 📥 Download Pre-built Vectors (Recommended)")
    print(f" • Download size: ~{download_gb:.1f} GB")
    print(f" • Setup time: ~30-60 minutes")
    print(f" • Ready to use: Immediately after download")
    
    print(f"\n [2] 🔨 Build Locally from Databases")
    print(f" • Processing time: 4-6 hours on fast NVMe SSD")
    print(f" • CPU usage: High (100% utilization)")
    print(f" • Ready to use: After processing completes")
    
    print(f"\n [3] ⬅️ Return to Main Menu (setup later)")
    
    # Check disk space
    try:
        stat = shutil.disk_usage(PathConfig.VECTORS)
        available_gb = stat.free / (1024**3)
        print(f"\n 💾 Available disk space: {available_gb:.1f} GB")
        if available_gb < download_gb:
            print(f" ⚠️ Warning: Insufficient space for download!")
    except Exception:
        pass
    
    choice = input("\n Choice (1-3): ").strip()
    
    if choice == "1":
        return _handle_download_option()
    elif choice == "2":
        return "preprocessing_prompt"
    else:
        return "main_menu"

def _handle_download_option() -> str:
    """Simply route to the download screen."""
    return "download_vectors"
