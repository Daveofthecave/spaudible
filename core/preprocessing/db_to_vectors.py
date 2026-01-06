# core/preprocessing/db_to_vectors.py
import sqlite3
import time
from pathlib import Path
from core.vectorization.track_vectorizer import build_track_vector
from .vector_exporter import VectorWriter
from .progress import ProgressTracker
from .mask_generator import generate_mask_file

class DatabaseReader:
    """Efficient, memory-mapped reader for Spotify SQLite databases."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_conn = None
        self.audio_conn = None
        
    def __enter__(self):
        """Open database connections."""
        self.main_conn = sqlite3.connect(self.main_db_path, timeout=60)
        self.audio_conn = sqlite3.connect(self.audio_db_path, timeout=60)
        
        # Performance optimizations
        self.main_conn.execute("PRAGMA cache_size = -2000000")  # 2GB cache
        self.main_conn.execute("PRAGMA mmap_size = 30000000000")  # 30GB memory mapping
        self.main_conn.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in RAM
        self.main_conn.execute("PRAGMA synchronous = OFF")
        self.main_conn.execute("PRAGMA journal_mode = OFF")
        self.main_conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        
        self.audio_conn.execute("PRAGMA synchronous = OFF")
        self.audio_conn.execute("PRAGMA journal_mode = OFF")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close database connections."""
        if self.main_conn:
            self.main_conn.close()
        if self.audio_conn:
            self.audio_conn.close()
    
    def get_track_count(self):
        """Get total number of tracks in the database."""
        cursor = self.main_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        return cursor.fetchone()[0]

    def get_track_by_id(self, track_id):
        """Fetch a single track by its Spotify ID."""
        query = """
        SELECT 
            t.rowid,
            t.id as track_id,
            t.duration_ms,
            t.popularity,
            a.release_date,
            MAX(art.followers_total) as max_followers,
            GROUP_CONCAT(DISTINCT ag.genre) as genres
        FROM tracks t
        JOIN albums a ON t.album_rowid = a.rowid
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        JOIN artists art ON ta.artist_rowid = art.rowid
        LEFT JOIN artist_genres ag ON art.rowid = ag.artist_rowid
        WHERE t.id = ?
        GROUP BY t.rowid
        """
        
        cursor = self.main_conn.cursor()
        cursor.execute(query, (track_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            track_data = dict(zip(columns, row))
            
            if track_data['genres']:
                track_data['genres'] = track_data['genres'].split(',')
            else:
                track_data['genres'] = []
            
            # Add audio features
            track_data.update(self.get_audio_features(track_data['track_id']))
            
            return track_data
        
        return None
    
    def get_batch_tracks(self, batch_size=100000, last_rowid=0):
        """
        Efficiently fetch a batch of tracks starting after last_rowid.
        Uses keyset pagination for constant-time performance.
        """
        query = """
        SELECT 
            t.rowid,
            t.id as track_id,
            t.duration_ms,
            t.popularity,
            a.release_date,
            MAX(art.followers_total) as max_followers,
            GROUP_CONCAT(DISTINCT ag.genre) as genres
        FROM tracks t
        JOIN albums a ON t.album_rowid = a.rowid
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        JOIN artists art ON ta.artist_rowid = art.rowid
        LEFT JOIN artist_genres ag ON art.rowid = ag.artist_rowid
        WHERE t.rowid > ?
        GROUP BY t.rowid
        ORDER BY t.rowid
        LIMIT ?
        """
        
        cursor = self.main_conn.cursor()
        cursor.execute(query, (last_rowid, batch_size))
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Yield tracks one by one
        for row in cursor:
            track_data = dict(zip(columns, row))
            
            # Parse comma-separated genres into list
            if track_data['genres']:
                track_data['genres'] = track_data['genres'].split(',')
            else:
                track_data['genres'] = []
            
            # Fetch audio features for this track
            track_data.update(self.get_audio_features(track_data['track_id']))
            
            yield track_data
    
    def get_audio_features(self, track_id):
        """Get audio features for a specific track."""
        query = """
        SELECT 
            duration_ms, time_signature, tempo, key, mode,
            danceability, energy, loudness, speechiness,
            acousticness, instrumentalness, liveness, valence
        FROM track_audio_features
        WHERE track_id = ?
        """
        
        cursor = self.audio_conn.cursor()
        cursor.execute(query, (track_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        else:
            # Return empty dict if no audio features found
            return {}

class PreprocessingEngine:
    """Orchestrates the database to vector conversion process."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors"):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = output_dir
        self.batch_size = 100000
        self.estimated_total = 256_000_000
    
    def run(self):
        """Run the full preprocessing pipeline."""
        print("\n" + "═" * 65)
        print("  🚀 Starting Database Preprocessing")
        print("═" * 65)
        
        # Validate databases exist
        if not Path(self.main_db_path).exists():
            print(f"\n  ❗ Database not found: {self.main_db_path}")
            return False
        
        if not Path(self.audio_db_path).exists():
            print(f"\n  ❗ Database not found: {self.audio_db_path}")
            return False
        
        print("\n  📊 Reading database statistics...")
        
        with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
            try:
                actual_total = db_reader.get_track_count()
                print(f"  Found {actual_total:,} tracks in database")
            except Exception as e:
                print(f"  ⚠️  Could not get exact track count: {e}")
                print(f"  Using estimated total: {self.estimated_total:,} tracks")
                actual_total = self.estimated_total
        
        print("\n  🛠️  Processing tracks in batches of 100,000...")
        print("  This will take several hours. Press Ctrl+C to interrupt.")
        print("\n  Progress:")
        
        progress = ProgressTracker(actual_total)
        processed_count = 0
        last_rowid = 0

        with VectorWriter(self.output_dir) as writer:
            with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                while processed_count < actual_total:
                    try:
                        # Process one batch
                        batch_processed = 0
                        for track_data in db_reader.get_batch_tracks(self.batch_size, last_rowid):
                            # Build vector
                            vector = build_track_vector(track_data)
                            
                            # Write to storage
                            writer.write_vector(track_data['track_id'], vector)
                            
                            # Update last_rowid to the current track's rowid
                            last_rowid = track_data['rowid']
                            
                            # Update progress
                            batch_processed += 1
                            progress.update(1)
                            
                            # Early exit if we've processed enough
                            if batch_processed >= self.batch_size:
                                break
                        
                        if batch_processed == 0:
                            # No more tracks to process
                            break
                        
                        processed_count += batch_processed
                        
                    except KeyboardInterrupt:
                        print("\n\n  ⏸️  Processing interrupted by user.")
                        print("  Partially processed data has been saved.")
                        return False
                    except Exception as e:
                        print(f"\n\n  ❗ Error during processing: {e}")
                        return False
        
        # Show completion
        progress.complete()
        
        # Generate mask file as separate pass
        print("\n⚙️  Generating mask file from vectors...")
        vectors_path = Path(self.output_dir) / "track_vectors.bin"
        masks_path = Path(self.output_dir) / "track_masks.bin"
        
        mask_start = time.time()
        mask_success = generate_mask_file(vectors_path, masks_path)
        mask_time = time.time() - mask_start
        
        if mask_success:
            print(f"✅ Mask generation completed in {mask_time:.1f} seconds")
            self._show_statistics(processed_count, masks_path)
            return True
        else:
            print(f"❌ Mask generation failed after {mask_time:.1f} seconds")
            return False
    
    def _show_statistics(self, total_processed, masks_path: Path):
        """Display processing statistics including mask file."""
        vectors_path = Path(self.output_dir) / "track_vectors.bin"
        index_path = Path(self.output_dir) / "track_index.bin"
        
        vectors_size = vectors_path.stat().st_size if vectors_path.exists() else 0
        masks_size = masks_path.stat().st_size if masks_path.exists() else 0
        index_size = index_path.stat().st_size if index_path.exists() else 0
        
        vectors_gb = vectors_size / (1024**3)
        masks_gb = masks_size / (1024**3)
        index_gb = index_size / (1024**3)
        total_gb = vectors_gb + masks_gb + index_gb
        
        print("\n  📊 Processing Statistics:")
        print(f"    Total tracks processed: {total_processed:,}")
        print(f"    Vector file size: {vectors_gb:.1f} GB")
        print(f"    Mask file size: {masks_gb:.1f} GB")
        print(f"    Index file size: {index_gb:.1f} GB")
        print(f"    Total disk space used: {total_gb:.1f} GB")
        print(f"    Output directory: {self.output_dir}")
        print("\n  ✅ Preprocessing complete! Ready for similarity search.")
