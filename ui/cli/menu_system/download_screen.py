# ui/cli/menu_system/download_screen.py
"""Download screen for fetching database files from HuggingFace Hub."""
import shutil
import sys
from pathlib import Path
from typing import Optional
from ui.cli.console_utils import print_header, format_elapsed_time
from core.utilities.download_manager import SpaudibleDownloader, DownloadError
from config import PathConfig, DownloadConfig

def screen_download_databases() -> str:
    """
    Screen for downloading database .zst files from HuggingFace Hub.
    
    Returns:
        Next screen identifier string
    """
    print_header("Download Required Databases")
    
    # Calculate space requirements
    download_gb = DownloadConfig.get_required_space_gb(include_databases=True, include_vectors=False)
    peak_usage_gb = DownloadConfig.get_total_extracted_space_gb()  # compressed + extracted
    
    # Check available disk space
    try:
        stat = shutil.disk_usage(PathConfig.DATABASES)
        available_gb = stat.free / (1024**3)
    except Exception:
        # Fallback if can't determine disk space
        available_gb = float('inf')
    
    print(f"\n📦 Download Size: {download_gb:.1f} GB (compressed)")
    print(f"  Peak Disk Usage: {peak_usage_gb:.1f} GB (during extraction)")
    print(f"  Available Space: {available_gb:.1f} GB")
    
    if available_gb < peak_usage_gb:
        print(f"\n ❌ Insufficient disk space!")
        print(f" You need at least {peak_usage_gb:.1f} GB free for extraction,")
        print(f" but only {available_gb:.1f} GB is available.")
        print(f"\n Options:")
        print(f" 1. Free up disk space")
        print(f" 2. Download files manually and place them in:")
        print(f"    {PathConfig.DATABASES}")
        input("\n Press Enter to return to system check...")
        return "database_check"
    
    if available_gb < peak_usage_gb * 1.2:  # 20% buffer warning
        print(f"\n ⚠️  Warning: Disk space is tight!")
        print(f" Ensure you have {peak_usage_gb:.1f} GB free before extraction.")
    
    print("\n Files to download:")
    for filename, size_gb, extracted_gb in DownloadConfig.DATABASE_FILES:
        print(f" • {filename}")
        print(f"   Download: {size_gb:.1f} GB → Extracted: {extracted_gb:.1f} GB")
    
    print("\n ⚠️  Important Notes:")
    print(" • Downloads can be resumed if interrupted")
    print(" • Do not close the program during download")
    print(" • Requires stable internet connection")
    print(" • HuggingFace Hub account is NOT required")
    
    while True:
        confirm = input("\n Start download? (yes/no): ").strip().lower()
        if confirm == 'yes':
            break
        elif confirm == 'no':
            print("\n Download cancelled.")
            print(" You can download manually from:")
            print(f" https://huggingface.co/datasets/{DownloadConfig.REPO_DB}")
            input("\n Press Enter to return...")
            return "database_check"
    
    # Initialize downloader
    downloader = SpaudibleDownloader()
    
    # Check if we have partial downloads to resume
    if downloader.state["in_progress"]:
        print("\n ⏳ Resuming previous download...")
        incomplete = list(downloader.state["in_progress"].keys())
        for key in incomplete:
            print(f"   - {key.replace('db_', '')}")
    
    try:
        print("\n" + "=" * 65)
        print("Initializing download...")
        print("=" * 65)
        print("(Progress will be shown automatically by download manager)\n")
        
        # Perform downloads
        success = downloader.download_databases()
        
        if success:
            print("\n" + "=" * 65)
            print("✅ All database files downloaded successfully!")
            print("=" * 65)
            print(f"\nDownloaded to: {PathConfig.DATABASES}")
            print("Next step: Extraction (decompressing the files)")
            
            # Ask if user wants to proceed to extraction immediately
            proceed = input("\n Proceed to extraction now? (yes/no): ").strip().lower()
            if proceed == 'yes':
                return "extraction_screen"
            else:
                return "database_check"
        else:
            print("\n ❌ Download could not be completed.")
            input("\n Press Enter to return...")
            return "database_check"
            
    except DownloadError as e:
        print(f"\n ❌ Download failed: {e}")
        
        # Check if it's a network issue or file issue
        if "404" in str(e) or "not found" in str(e).lower():
            print("\n The file may have been moved or renamed.")
            print(f" Please check: https://huggingface.co/datasets/{DownloadConfig.REPO_DB}")
        
        print("\n Options:")
        print(" 1. Retry download")
        print(" 2. Clear download state and retry")
        print(" 3. Return to system check")
        
        choice = input("\n Choice (1-3): ").strip()
        
        if choice == "1":
            return "download_screen"
        elif choice == "2":
            print(" Clearing download state...")
            downloader.clear_state()
            print(" State cleared. You can retry now.")
            input(" Press Enter to retry...")
            return "download_screen"
        else:
            return "database_check"
            
    except KeyboardInterrupt:
        print("\n\n ⚠️  Download interrupted by user!")
        print(" Progress has been saved. You can resume by restarting.")
        print(f" Resume from: {downloader.state_file}")
        input("\n Press Enter to return...")
        return "database_check"
    except Exception as e:
        print(f"\n ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\n Press Enter to return...")
        return "database_check"

def screen_download_vectors() -> str:
    """
    Alternative screen for downloading vector cache files directly.
    Used by vector_choice_screen when user selects download option.
    
    Returns:
        Next screen identifier string
    """
    print_header("Download Vector Cache")
    
    # Calculate space requirements
    vector_gb = DownloadConfig.get_required_space_gb(include_databases=False, include_vectors=True)
    
    # Check available disk space
    try:
        stat = shutil.disk_usage(PathConfig.VECTORS)
        available_gb = stat.free / (1024**3)
    except Exception:
        available_gb = float('inf')
    
    print(f"\n 📦 Download Size: {vector_gb:.1f} GB")
    print(f" 💾 Available Space: {available_gb:.1f} GB")
    
    if available_gb < vector_gb:
        print(f"\n ❌ Insufficient disk space!")
        print(f" You need at least {vector_gb:.1f} GB free.")
        input("\n Press Enter to return...")
        return "vector_choice"
    
    print("\n Files to download:")
    for filename, subdir, size_gb in DownloadConfig.VECTOR_FILES:
        path_display = f"{subdir}/{filename}" if subdir else filename
        print(f" • {path_display} ({size_gb} GB)")
    
    print("\n ⚠️  This will download pre-built search indexes.")
    print("    Skip this if you prefer to build locally (takes 4-6 hours).")
    
    confirm = input("\n Start download? (yes/no): ").strip().lower()
    if confirm != 'yes':
        return "vector_choice"
    
    downloader = SpaudibleDownloader()
    
    try:
        print("\n" + "=" * 65)
        print("Downloading vector cache...")
        print("=" * 65)
        
        success = downloader.download_vector_cache()
        
        if success:
            print("\n ✅ Vector cache downloaded successfully!")
            print(" Spaudible is ready to use.")
            input("\n Press Enter to go to main menu...")
            return "main_menu"
        else:
            print("\n ❌ Download incomplete.")
            input("\n Press Enter to return...")
            return "vector_choice"
            
    except DownloadError as e:
        print(f"\n ❌ Download failed: {e}")
        print("\n Options:")
        print(" 1. Retry")
        print(" 2. Return to choice menu")
        choice = input("\n Choice: ").strip()
        return "download_vectors" if choice == "1" else "vector_choice"
        
    except KeyboardInterrupt:
        print("\n\n ⚠️  Download interrupted!")
        print(" Progress saved. Resume by restarting.")
        input("\n Press Enter to return...")
        return "vector_choice"
