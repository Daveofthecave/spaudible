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
    """Screen for downloading database .zst files from HuggingFace Hub."""
    print_header("Download Required Databases")
    
    # Calculate space requirements
    download_gb = DownloadConfig.get_required_space_gb(include_databases=True, include_vectors=False)
    peak_usage_gb = DownloadConfig.get_total_extracted_space_gb()  # compressed + extracted
    
    # Check available disk space
    try:
        stat = shutil.disk_usage(PathConfig.DATABASES)
        available_gb = stat.free / (1e9)
    except Exception:
        # Fallback if can't determine disk space
        available_gb = float('inf')
    
    print(f"\n  Download Size: {download_gb:.1f} GB (compressed)")
    print(f"  Peak Disk Usage: {peak_usage_gb:.1f} GB (during extraction)")
    print(f"  Available Space: {available_gb:.1f} GB")
    
    if available_gb < peak_usage_gb:
        print(f"\n  ❗️ Insufficient disk space!")
        print(f"  You need at least {peak_usage_gb:.1f} GB free for extraction,")
        print(f"  but only {available_gb:.1f} GB is available.")
        print(f"\n  Options:")
        print(f"  1. Free up disk space")
        print(f"  2. Download files manually and place them in:")
        print(f"     {PathConfig.DATABASES}")
        input("\n  Press Enter to return to system check...")
        return "database_check"
    
    if available_gb < peak_usage_gb * 1.2:  # 20% buffer warning
        print(f"\n  ⚠️ Warning: Disk space is tight!")
        print(f"  Ensure you have {peak_usage_gb:.1f} GB free before extraction.")
    
    print("\n  Files to download:\n")
    for filename, size_gb, extracted_gb in DownloadConfig.DATABASE_FILES:
        print(f"  • {filename}")
        print(f"    Compressed: {size_gb:.1f} GB → Extracted: {extracted_gb:.1f} GB")
    
    print("\n  Downloads can be resumed if interrupted.")
    
    while True:
        confirm = input("\n  Start download? (yes/no): ").strip().lower()
        if confirm == 'yes':
            break
        elif confirm == 'no':
            print("\n  Download cancelled.")
            print("  You can download manually from:")
            print(f"  https://huggingface.co/datasets/{DownloadConfig.REPO_DB}")
            input("\n  Press Enter to return...")
            return "database_check"
    
    # Initialize downloader
    downloader = SpaudibleDownloader()
    
    # Check if we have partial downloads to resume
    if downloader.state["in_progress"]:
        print("\n  Resuming previous download...")
        incomplete = list(downloader.state["in_progress"].keys())
        for key in incomplete:
            print(f"  - {key.replace('db_', '')}")
    
    try:
        print_header("Initializing download...")
        print()
        
        # Perform downloads
        success = downloader.download_databases()
        
        if success:
            print_header("✅ All database files downloaded successfully!")
            print(f"\n  Downloaded to: {PathConfig.DATABASES}")
            print("  Next step: Extraction (decompressing the files)")
            
            # Ask if user wants to proceed to extraction immediately
            proceed = input("\n  Proceed to extraction now? (yes/no): ").strip().lower()
            if proceed == 'yes':
                return "extraction_screen"
            else:
                return "database_check"
        else:
            print("\n  ❗️ Download could not be completed.")
            input("\n  Press Enter to return...")
            return "database_check"
            
    except DownloadError as e:
        print(f"\n  ❗️ Download failed: {e}")
        
        # Check if it's a network issue or file issue
        if "404" in str(e) or "not found" in str(e).lower():
            print("\n  The file may have been moved or renamed.")
            print(f"  Please check: https://huggingface.co/datasets/{DownloadConfig.REPO_DB}")

            print("\n  Options:")
            options = [
                "Retry download", 
                "Clear download state and retry", 
                "Return to system check"
            ]
            print_menu(options)            
            choice = get_choice(len(options))
            
            if choice == 1:
                return "download_screen"
            elif choice == 2:
                print("  Clearing download state...")
                downloader.clear_state()
                print("  State cleared. You can retry now.")
                input("  Press Enter to retry...")
                return "download_screen"
            else:
                return "database_check"
                
    except KeyboardInterrupt:
        print("\n\n  ⚠️ Download interrupted by user!")
        print("  Progress has been saved. You can resume by restarting.")
        print(f"  Resume from: {downloader.state_file}")
        input("\n  Press Enter to return...")
        return "database_check"
        
    except Exception as e:
        print(f"\n  ❗️ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\n  Press Enter to return...")
        return "database_check"

def screen_download_vectors() -> str:
    """Screen for downloading vector cache files with partial download support."""
    print_header("Download Vector Cache")
    
    # Check which files actually need to be downloaded
    files_needed = []
    total_size_needed = 0.0
    
    for filename, subdir, size_gb in DownloadConfig.VECTOR_FILES:
        if subdir:
            file_path = PathConfig.VECTORS / subdir / filename
        else:
            file_path = PathConfig.VECTORS / filename
            
        # Check if exists and has reasonable size (>1MB to avoid empty/corrupt files)
        if not file_path.exists() or file_path.stat().st_size < 1_000_000:
            files_needed.append((filename, subdir, size_gb))
            total_size_needed += size_gb
    
    # If all files exist, we're done
    if not files_needed:
        print("\n  ✅ All vector cache files already exist!")
        input("\n  Press Enter to continue...")
        return "main_menu"
    
    # Check if this is just a query index download (vectors already exist)
    has_main_vectors = PathConfig.get_vector_file().exists() and PathConfig.get_index_file().exists()
    only_query_index = has_main_vectors and all(f[0] in ['inverted_index.bin', 'marisa_trie.bin'] for f in files_needed)
    
    if only_query_index:
        print("\n  ⚠️ Query index files are missing (required for text search).")
        print("  The main vector files exist, but the search index needs to be downloaded.\n")
    else:
        print("\n  The vector cache enables fast similarity searching.")
        print("  You can download pre-built files or build them locally from databases.\n")
    
    # Check available disk space (only for needed files)
    try:
        stat = shutil.disk_usage(PathConfig.VECTORS)
        available_gb = stat.free / (1e9)
    except Exception:
        available_gb = float('inf')
    
    print(f"  Download Size: {total_size_needed:.1f} GB")
    print(f"  Available Space: {available_gb:.1f} GB")
    
    if available_gb < total_size_needed:
        print(f"\n  ❗️ Insufficient disk space!")
        print(f"  You need at least {total_size_needed:.1f} GB free.")
        input("\n  Press Enter to return...")
        return "vector_choice" if not only_query_index else "database_check"
    
    # Show which files will be downloaded
    print("\n  The following files will be downloaded:\n")
    for filename, subdir, size_gb in files_needed:
        path_display = f"{subdir}/{filename}" if subdir else filename
        print(f"  • {path_display} ({size_gb} GB)")
    
    if len(files_needed) < len(DownloadConfig.VECTOR_FILES):
        existing_count = len(DownloadConfig.VECTOR_FILES) - len(files_needed)
    
    print("\n  This will download from HuggingFace Hub (Daveofthecave/spaudible_vectors).")
    confirm = input("\n  Start download? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        return "vector_choice" if not only_query_index else "database_check"
    
    downloader = SpaudibleDownloader()
    
    try:
        print_header("Downloading vector cache...")
        
        # Download only the files we need
        success_count = 0
        
        for filename, subdir, size_gb in files_needed:
            state_key = f"vec_{filename}"
            
            # Skip if already marked as completed in download state
            if downloader.state["completed"].get(state_key):
                print(f"\n✅ {filename} already downloaded")
                success_count += 1
                continue
            
            print(f"\n  Downloading {filename} ({size_gb} GB)...")
            
            try:
                # Construct the filename path as it appears in the repo
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
                print(f"✅ {filename} downloaded")
                success_count += 1
                
            except Exception as e:
                print(f"❗️ Failed to download {filename}: {e}")
                # Continue with other files
        
        # Determine next screen based on what we downloaded
        if success_count == len(files_needed):
            print("\n  ✅ All required files downloaded successfully!")
            
            if only_query_index:
                print("  The query index is now ready.")
                input("\n  Press Enter to continue...")
                return "database_check"
            else:
                print("  Spaudible is ready to use.")
                input("\n  Press Enter to go to main menu...")
                return "main_menu"
                
        elif success_count > 0:
            print(f"\n  ⚠️ Downloaded {success_count}/{len(files_needed)} files.")
            print("  Some files failed to download. You may need to retry.")
            input("\n  Press Enter to return...")
            return "vector_choice" if not only_query_index else "database_check"
        else:
            print("\n  ❗️ Download failed. No files were downloaded.")
            input("\n  Press Enter to return...")
            return "vector_choice" if not only_query_index else "database_check"
            
    except KeyboardInterrupt:
        print("\n\n  ⚠️ Download interrupted!")
        print("  Progress saved. Resume by restarting.")
        input("\n  Press Enter to return...")
        return "download_vectors" if not only_query_index else "database_check"
    except Exception as e:
        print(f"\n  ❗️ Unexpected error: {e}")
        input("\n  Press Enter to return...")
        return "vector_choice" if not only_query_index else "database_check"
