# ui/cli/menu_system/extraction_screen.py
"""Extraction screen for decompressing Zstandard (.zst) database files."""
import shutil
import time
import sys
from pathlib import Path
from typing import Optional
from ui.cli.console_utils import print_header, format_elapsed_time
from core.utilities.extraction_manager import ZstExtractor, ExtractionError
from config import PathConfig, DownloadConfig

def screen_extraction() -> str:
    """Screen for extracting database .zst files with progress tracking."""
    print_header("Database File Extraction")
    # Determine extraction status
    extractor = ZstExtractor()
    status = extractor.get_extraction_status()

    # If nothing to extract (no archives present)
    if not status['archives_present'] and not status['extracted']:
        print("\n❗️ No database archives found.")
        print(f"  Please download the database files first.")
        input("\n  Press Enter to return to download...")
        return "download_screen"
    
    # If all already extracted
    if not status['to_extract'] and status['extracted']:
        print("\n✅ All databases already extracted!")
        print(f"  Extracted files:")
        for db in status['extracted']:
            print(f"• {db}")
        input("\n  Press Enter to continue to vector setup...")
        return "vector_choice"
    
    # Show extraction plan
    print("\n  The following databases need to be extracted:\n")
    total_compressed = 0
    total_extracted = 0

    for filename in status['to_extract']:
        # Get sizes from config for display
        for fname, compressed_gb, extracted_gb in DownloadConfig.DATABASE_FILES:
            if fname == filename:
                print(f"• {filename} ({compressed_gb:.1f} GB → {extracted_gb:.1f} GB)")
                total_compressed += compressed_gb
                total_extracted += extracted_gb
                break
    
    # Check disk space for extraction (need compressed + extracted temporarily)
    try:
        stat = shutil.disk_usage(PathConfig.DATABASES)
        available_gb = stat.free / (1e9)
        required_gb = total_compressed + total_extracted

        print(f"\n Disk Space Required: {required_gb:.1f} GB")
        print(f" Available: {available_gb:.1f} GB")

        if available_gb < required_gb:
            print(f"\n❗️ Insufficient disk space!")
            print(f"  You need at least {required_gb:.1f} GB free for extraction.")
            print(f"  (This includes space for both compressed and extracted files)")
            input("\n  Press Enter to return...")
            return "database_check"
    
    except Exception as e:
        print(f"\n⚠️ Could not check disk space: {e}")
        confirm = input("\n  Continue anyway? (yes/no): ").strip().lower()
        if confirm != 'yes':
            return "database_check"
    
    # Extraction confirmation
    db_count = len(status['to_extract'])
    file_str = "file" if db_count == 1 else "files"

    print(f"\n  Ready to extract {db_count} {file_str}.")
    print("  This may take 5-15 minutes depending on your")
    print("  CPU and disk speed.")

    confirm = input("\n  Start extraction? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("\n  Extraction cancelled.")
        input("  Press Enter to return...")
        return "database_check"
    
    # Perform extraction
    print_header("Extracting database files...")

    overall_start = time.time()
    files_completed = 0
    failed_files = []

    for filename in status['to_extract']:
        print(f"\n Processing: {filename}")
        start_time = time.time()

        # Progress callback for this file
        def make_progress_callback(fname):
            def callback(name, bytes_processed, total_bytes):
                percent = (bytes_processed / total_bytes) * 100
                gb_processed = bytes_processed / (1e9)
                gb_total = total_bytes / (1e9)
                # Cap at 100% in case of small size discrepancies
                display_percent = min(percent, 100.0)
                bar_width = 30
                filled = int(bar_width * (percent / 100))
                # Use actual percent for bar fill
                bar = '█' * filled + '░' * (bar_width - filled)
                # Show percentage capped at 100%, but actual MBs
                sys.stdout.write(f"\r [{bar}] {display_percent:3.1f}% ({gb_processed:>5.1f}/{gb_total:>5.1f} GB)")
                sys.stdout.flush()
            return callback
        
        try:
            # Create extractor with progress callback
            file_extractor = ZstExtractor(progress_callback=make_progress_callback(filename))
            extracted_path = file_extractor.extract_database(filename)
            elapsed = time.time() - start_time
            size_gb = extracted_path.stat().st_size / (1e9)
            print(f"\r✅ Done: {size_gb:.2f} GB extracted in {format_elapsed_time(elapsed).strip()} ")
            files_completed += 1
        except ExtractionError as e:
            print(f"\n❗️ Failed: {e}")
            failed_files.append((filename, str(e)))
        except KeyboardInterrupt:
            print("\n\n⚠️ Extraction interrupted!")
            print("  Partial file may remain. It will be cleaned up on next run.")
            input("\n  Press Enter to return...")
            return "database_check"
    
    # Summary
    print_header("Extraction Summary")
    print(f"  Completed: {files_completed}/{len(status['to_extract'])} files")

    if failed_files:
        print(f"\n⚠️ {len(failed_files)} file(s) failed:")
        for fname, err in failed_files:
            print(f"• {fname}: {err}")
        input("\n  Press Enter to return...")
        return "database_check"
    
    total_elapsed = time.time() - overall_start
    print(f"\n⏱ Total time: {format_elapsed_time(total_elapsed).strip()}")
    # Cleanup option if all succeeded
    if files_completed == len(status['to_extract']):
        print("\n  Cleaning up compressed archives...")
        try:
            removed = extractor.cleanup_archives()
            freed_gb = 0
            for filename in status['to_extract']:
                for fname, compressed_gb, _ in DownloadConfig.DATABASE_FILES:
                    if fname == filename:
                        freed_gb += compressed_gb
                        break
            print(f"  Deleted {removed} archive(s), freed ~{freed_gb:.1f} GB")
        except Exception as e:
            print(f"⚠️ Could not delete archives: {e}")
            print("  You can manually delete the .zst files later to save space.")
        
        print("\n✅ Databases are ready!")
        print("  Next step: Set up vector cache for similarity search.")
        input("\n  Press Enter to continue...")

        return "vector_choice"

    return "database_check"
