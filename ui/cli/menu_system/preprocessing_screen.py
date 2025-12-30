# ui/cli/menu_system/preprocessing_screen.py
import sys
import time
from pathlib import Path
from ui.cli.console_utils import print_header
from core.preprocessing import DatabaseReader, VectorWriter, ProgressTracker
from core.vectorization.genre_mapper import load_genre_mapping
from config import PathConfig

def screen_preprocessing():
    """Screen 3: Run the actual preprocessing engine."""
    print_header("Spaudible - Setup in Progress")
    
    try:
        print("\n  Checking databases...")
        
        # Check if databases exist
        main_db = PathConfig.get_main_db()
        audio_db = PathConfig.get_audio_db()
        
        if not Path(main_db).exists():
            print("\n  ❗ Database not found:", main_db.name)
            print("\n  Press Enter to return to the main menu...")
            input()
            return "database_check"
        
        if not Path(audio_db).exists():
            print("\n  ❗ Database not found:", audio_db.name)
            print("\n  Press Enter to return to the main menu...")
            input()
            return "database_check"
        
        print("  Loading genre mappings...", end="", flush=True)
        load_genre_mapping()
        print()
        
        # Get actual track count
        with DatabaseReader(str(main_db), str(audio_db)) as db_reader:
            try:
                actual_total = db_reader.get_track_count()
                print(f"  Found {actual_total:,} tracks in database")
            except Exception as e:
                print(f"  ⚠️  Could not get exact track count: {e}")
                print("  Using estimated total: 256,000,000 tracks")
                actual_total = 256_000_000
        
        print("\n  🛠️  Processing tracks in batches of 100,000...")
        print("  This will take several hours. Press Ctrl+C to interrupt.")
        print("\n  Progress:")
        
        # Initialize progress tracker
        progress = ProgressTracker(actual_total)
        
        # Process in batches
        processed_count = 0
        last_rowid = 0

        with VectorWriter(str(PathConfig.VECTORS)) as writer:
            with DatabaseReader(str(main_db), str(audio_db)) as db_reader:
                while processed_count < actual_total:
                    try:
                        batch_processed = 0
                        for track_data in db_reader.get_batch_tracks(100000, last_rowid):
                            from core.vectorization.track_vectorizer import build_track_vector
                            vector = build_track_vector(track_data)
                            writer.write_vector(track_data['track_id'], vector)
                            last_rowid = track_data['rowid']
                            batch_processed += 1
                            progress.update(1)
                            if batch_processed >= 100000:
                                break
                        
                        if batch_processed == 0:
                            break
                        
                        processed_count += batch_processed
                        
                    except KeyboardInterrupt:
                        print("\n\n  ⏸️  Processing interrupted by user.")
                        print("  Partially processed data has been saved.")
                        print("\n  Press Enter to return to main menu...")
                        input()
                        return "database_check"
                    except Exception as e:
                        print(f"\n\n  ❗ Error during processing: {e}")
                        print("\n  Press Enter to return to main menu...")
                        input()
                        return "database_check"
        
        # Show completion
        progress.complete()
        
        # Show output statistics
        vectors_path = PathConfig.get_vector_file()
        index_path = PathConfig.get_index_file()
        
        vectors_size = vectors_path.stat().st_size if vectors_path.exists() else 0
        index_size = index_path.stat().st_size if index_path.exists() else 0
        
        vectors_gb = vectors_size / (1024**3)
        index_gb = index_size / (1024**3)
        total_gb = vectors_gb + index_gb
        
        print("\n  📊 Processing Statistics:")
        print(f"    Total tracks processed: {processed_count:,}")
        print(f"    Vector cache size: {vectors_gb:.1f} GB")
        print(f"    Vector index size: {index_gb:.1f} GB")
        print(f"    Total disk space used: {total_gb:.1f} GB")
        print(f"    Output directory: {PathConfig.VECTORS}")
        print("\n  ✅ Setup complete! Your music database is ready for searching.")
        
        print("\n  Press Enter to continue...")
        input()
        return "processing_complete"
        
    except ImportError as e:
        print(f"\n  ❗ Could not import preprocessing modules: {e}")
        print("  Make sure all required files are in place.")
        print("\n  Press Enter to return...")
        input()
        return "database_check"
    except Exception as e:
        print(f"\n  ❗ Unexpected error: {e}")
        print("\n  Press Enter to return...")
        input()
        return "database_check"
