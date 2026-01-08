# core/preprocessing/db_to_vectors.py
import sqlite3
import time
import numpy as np
from pathlib import Path
from core.vectorization.track_vectorizer import build_track_vectors_batch
from .vector_exporter import VectorWriter
from .progress import ProgressTracker
from .mask_generator import generate_mask_file
import mmap
import gc

class DatabaseReader:
    """Highly optimized reader for Spotify SQLite databases using memory mapping."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_conn = None
        self.audio_conn = None
        self.main_mmap = None
        self.audio_mmap = None
        self.artist_followers_cache = {}
        self.artist_genres_cache = {}
        
    def __enter__(self):
        """Open database connections with aggressive optimizations."""
        # Open connections with timeout disabled
        self.main_conn = sqlite3.connect(self.main_db_path, timeout=0)
        self.audio_conn = sqlite3.connect(self.audio_db_path, timeout=0)
        
        # Set aggressive performance settings
        self.main_conn.execute("PRAGMA journal_mode = MEMORY")
        self.main_conn.execute("PRAGMA cache_size = -200000")  # 200GB cache
        self.main_conn.execute("PRAGMA temp_store = MEMORY")
        self.main_conn.execute("PRAGMA synchronous = OFF")
        self.main_conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        self.main_conn.execute("PRAGMA mmap_size = 30000000000")  # 30GB memory mapping
        
        self.audio_conn.execute("PRAGMA journal_mode = MEMORY")
        self.audio_conn.execute("PRAGMA synchronous = OFF")
        self.audio_conn.execute("PRAGMA temp_store = MEMORY")
        
        # Memory map databases for faster access
        self._memory_map_databases()
        
        # Preload artist metadata into memory
        self._preload_artist_metadata()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close database connections."""
        if self.main_conn:
            self.main_conn.close()
        if self.audio_conn:
            self.audio_conn.close()
        if self.main_mmap:
            self.main_mmap.close()
        if self.audio_mmap:
            self.audio_mmap.close()
    
    def _memory_map_databases(self):
        """Memory-map database files for faster access."""
        try:
            # Memory-map main database
            with open(self.main_db_path, 'rb') as f:
                self.main_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Memory-map audio database
            with open(self.audio_db_path, 'rb') as f:
                self.audio_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            print(f"  ⚠️  Could not memory-map databases: {e}")
    
    def _preload_artist_metadata(self):
        """Preload all artist metadata into memory for fast access."""
        # Load artist followers
        cursor = self.main_conn.cursor()
        cursor.execute("SELECT rowid, followers_total FROM artists")
        self.artist_followers_cache = {rowid: followers for rowid, followers in cursor}
        
        # Load artist genres
        cursor.execute("SELECT artist_rowid, genre FROM artist_genres")
        self.artist_genres_cache = {}
        for artist_rowid, genre in cursor:
            if artist_rowid not in self.artist_genres_cache:
                self.artist_genres_cache[artist_rowid] = []
            self.artist_genres_cache[artist_rowid].append(genre)
        cursor.close()
    
    def get_track_count(self):
        """Get total number of tracks in the database."""
        cursor = self.main_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def stream_tracks(self, batch_size=500000, last_rowid=0):
        """
        Generator that streams tracks with optimized data fetching.
        Yields batches of track data dictionaries.
        """
        cursor = self.main_conn.cursor()
        query = """
        SELECT 
            t.rowid,
            t.id as track_id,
            t.external_id_isrc,
            t.duration_ms,
            t.popularity,
            a.release_date,
            GROUP_CONCAT(ta.artist_rowid) as artist_ids
        FROM tracks t
        JOIN albums a ON t.album_rowid = a.rowid
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        WHERE t.rowid > ?
        GROUP BY t.rowid
        ORDER BY t.rowid
        LIMIT ?
        """
        
        while True:
            cursor.execute(query, (last_rowid, batch_size))
            columns = [col[0] for col in cursor.description]
            batch = cursor.fetchall()
            
            if not batch:
                break
            
            # Convert to dictionaries and enrich with artist metadata
            enriched_batch = []
            for row in batch:
                track_data = dict(zip(columns, row))
                artist_ids = [int(id) for id in track_data['artist_ids'].split(',')] if track_data['artist_ids'] else []
                
                # Get max followers
                max_followers = 0
                for artist_id in artist_ids:
                    followers = self.artist_followers_cache.get(artist_id, 0)
                    if followers > max_followers:
                        max_followers = followers
                track_data['max_followers'] = max_followers
                
                # Get genres
                genres = set()
                for artist_id in artist_ids:
                    if artist_id in self.artist_genres_cache:
                        genres.update(self.artist_genres_cache[artist_id])
                track_data['genres'] = list(genres)
                
                enriched_batch.append(track_data)
                last_rowid = track_data['rowid']
            
            yield enriched_batch
            gc.collect()  # Prevent memory bloat
            
        cursor.close()

class PreprocessingEngine:
    """Highly optimized preprocessing engine with streaming architecture."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors"):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = output_dir
        self.batch_size = 500_000  # Optimal batch size
        self.vector_batch_size = 100_000  # Vector processing batch size
        self.total_vectors = 256_000_000
    
    def run(self):
        """Run optimized preprocessing pipeline."""
        print("\n" + "═" * 65)
        print("  🚀 Starting Optimized Database Preprocessing")
        print("═" * 65)
        
        # Validate databases
        if not Path(self.main_db_path).exists():
            print(f"\n  ❗ Database not found: {self.main_db_path}")
            return False
        if not Path(self.audio_db_path).exists():
            print(f"\n  ❗ Database not found: {self.audio_db_path}")
            return False
        
        # Initialize progress tracker
        with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
            try:
                actual_total = db_reader.get_track_count()
                print(f"  Found {actual_total:,} tracks in database")
                self.total_vectors = actual_total
            except Exception:
                print(f"  Using estimated total: {self.total_vectors:,} tracks")
        
        print("\n  🔥 Processing tracks with streaming architecture...")
        print("  This will take 30-90 minutes. Press Ctrl+C to interrupt.")
        
        # Initialize progress tracker
        progress = ProgressTracker(self.total_vectors)
        processed_count = 0
        
        # Initialize vector writer
        with VectorWriter(self.output_dir) as writer:
            with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                # Process in streaming batches
                for batch in db_reader.stream_tracks(self.batch_size):
                    # Process vector batches within the main batch
                    for i in range(0, len(batch), self.vector_batch_size):
                        vector_batch = batch[i:i+self.vector_batch_size]
                        
                        # Build vectors
                        vectors = build_track_vectors_batch(vector_batch)
                        
                        # Write vectors
                        for track_data, vector in zip(vector_batch, vectors):
                            writer.write_vector(
                                track_data['track_id'], 
                                vector,
                                track_data.get('external_id_isrc', '')
                            )
                        
                        # Update progress
                        progress.update(len(vector_batch))
                        processed_count += len(vector_batch)
                
                # Final progress update
                progress.update(self.total_vectors - processed_count)
        
        # Finalize processing
        progress.complete()
        
        # Generate mask file
        print("\n⚙️  Generating mask file from vectors...")
        vectors_path = Path(self.output_dir) / "track_vectors.bin"
        masks_path = Path(self.output_dir) / "track_masks.bin"
        
        mask_start = time.time()
        mask_success = generate_mask_file(vectors_path, masks_path)
        mask_time = time.time() - mask_start
        
        if mask_success:
            print(f"✅ Mask generation completed in {mask_time:.1f} seconds")
            self._show_statistics(self.total_vectors, masks_path)
            return True
        else:
            print(f"❌ Mask generation failed after {mask_time:.1f} seconds")
            return False
    
    def _show_statistics(self, total_processed, masks_path: Path):
        """Display processing statistics."""
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
