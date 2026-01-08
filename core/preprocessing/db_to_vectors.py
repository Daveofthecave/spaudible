# core/preprocessing/db_to_vectors.py
import sqlite3
import time
import os
import concurrent.futures
import numpy as np
from pathlib import Path
from core.vectorization.track_vectorizer import build_track_vectors_batch
from .vector_exporter import VectorWriter
from .progress import ProgressTracker
from .mask_generator import generate_mask_file
import gc
import mmap

class VectorBuilder:
    """Wrapper for vector building to enable multiprocessing."""
    @staticmethod
    def build_vectors(track_data_batch):
        return build_track_vectors_batch(track_data_batch)

class DatabaseReader:
    """Highly optimized reader for Spotify SQLite databases."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_conn = None
        self.audio_conn = None
        self.artist_followers = {}
        self.artist_genres = {}
        self.main_mmap = None
        self.audio_mmap = None
        
    def __enter__(self):
        """Open database connections with aggressive optimizations."""
        # Open connections with timeout disabled
        self.main_conn = sqlite3.connect(self.main_db_path, timeout=0)
        self.audio_conn = sqlite3.connect(self.audio_db_path, timeout=0)
        
        # Memory map databases for faster access
        self._memory_map_databases()
        
        # Preload artist metadata into memory
        self._preload_artist_metadata()
        
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
            
            print("  ✅ Memory-mapped databases for faster access")
        except Exception as e:
            print(f"  ⚠️  Could not memory-map databases: {e}")
    
    def _preload_artist_metadata(self):
        """Preload all artist metadata into memory for fast access."""
        print("  Preloading artist metadata into memory...")
        
        # Use explicit cursor management
        cursor = self.main_conn.cursor()
        try:
            # Load artist followers
            cursor.execute("SELECT rowid, followers_total FROM artists")
            self.artist_followers = {rowid: followers for rowid, followers in cursor}
            
            # Load artist genres
            cursor.execute("SELECT artist_rowid, genre FROM artist_genres")
            self.artist_genres = {}
            for artist_rowid, genre in cursor:
                if artist_rowid not in self.artist_genres:
                    self.artist_genres[artist_rowid] = []
                self.artist_genres[artist_rowid].append(genre)
        finally:
            cursor.close()
        
        print(f"  Loaded {len(self.artist_followers):,} artists and "
              f"{sum(len(g) for g in self.artist_genres.values()):,} genres")
    
    def get_track_count(self):
        """Get total number of tracks in the database."""
        cursor = self.main_conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM tracks")
            return cursor.fetchone()[0]
        finally:
            cursor.close()

    def get_audio_features_batch(self, track_ids):
        """Get audio features for multiple tracks using temporary table."""
        if not track_ids:
            return {}
        
        cursor = self.audio_conn.cursor()
        try:
            # Create temporary table for batch IDs
            cursor.execute("CREATE TEMP TABLE batch_ids (track_id TEXT)")
            
            # Insert all track IDs in bulk (allow duplicates)
            cursor.executemany("INSERT INTO batch_ids VALUES (?)", [(tid,) for tid in track_ids])
            
            # Execute single bulk query
            cursor.execute("""
                SELECT 
                    track_id,
                    duration_ms, time_signature, tempo, key, mode,
                    danceability, energy, loudness, speechiness,
                    acousticness, instrumentalness, liveness, valence
                FROM track_audio_features
                WHERE track_id IN (SELECT DISTINCT track_id FROM batch_ids)
            """)
            
            # Build features map
            features_map = {}
            for row in cursor:
                track_id = row[0]
                features = {
                    'duration_ms': row[1],
                    'time_signature': row[2],
                    'tempo': row[3],
                    'key': row[4],
                    'mode': row[5],
                    'danceability': row[6],
                    'energy': row[7],
                    'loudness': row[8],
                    'speechiness': row[9],
                    'acousticness': row[10],
                    'instrumentalness': row[11],
                    'liveness': row[12],
                    'valence': row[13]
                }
                features_map[track_id] = features
            
            # Clean up temporary table
            cursor.execute("DROP TABLE batch_ids")
            
            return features_map
        finally:
            cursor.close()

    def get_batch_tracks(self, batch_size=500000, last_rowid=0):
        """
        Optimized batch fetching using keyset pagination.
        """
        # Calculate end position
        end_idx = last_rowid + batch_size
        
        # Prepare query to get distinct tracks
        query = """
        SELECT 
            t.rowid,
            t.id as track_id,
            t.external_id_isrc,
            t.duration_ms,
            t.popularity,
            a.release_date,
            ta.artist_rowid
        FROM tracks t
        JOIN albums a ON t.album_rowid = a.rowid
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        WHERE t.rowid > ? AND t.rowid <= ?
        GROUP BY t.rowid
        """
        
        cursor = self.main_conn.cursor()
        try:
            cursor.execute(query, (last_rowid, end_idx))
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Collect tracks and track IDs
            tracks = []
            track_ids = []
            artist_map = {}
            
            for row in cursor:
                track_data = dict(zip(columns, row))
                
                # Store artist IDs for later enrichment
                artist_rowid = track_data['artist_rowid']
                if track_data['rowid'] not in artist_map:
                    artist_map[track_data['rowid']] = []
                artist_map[track_data['rowid']].append(artist_rowid)
                
                tracks.append(track_data)
                track_ids.append(track_data['track_id'])
            
            # Enrich tracks with artist metadata
            for track in tracks:
                artist_ids = artist_map.get(track['rowid'], [])
                
                # Get max followers
                max_followers = 0
                for artist_id in artist_ids:
                    followers = self.artist_followers.get(artist_id, 0)
                    if followers > max_followers:
                        max_followers = followers
                track['max_followers'] = max_followers
                
                # Get genres
                genres = set()
                for artist_id in artist_ids:
                    if artist_id in self.artist_genres:
                        genres.update(self.artist_genres[artist_id])
                track['genres'] = list(genres)
            
            # Bulk fetch audio features
            audio_features_map = self.get_audio_features_batch(track_ids)
            
            # Add audio features to each track
            for track_data in tracks:
                track_id = track_data['track_id']
                if track_id in audio_features_map:
                    track_data.update(audio_features_map[track_id])
            
            return tracks
        finally:
            cursor.close()

class PreprocessingEngine:
    """Highly optimized preprocessing engine with memory management."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors"):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = output_dir
        self.batch_size = 1_000_000  # Increased batch size
        self.estimated_total = 256_000_000
        self.workers = min(8, os.cpu_count())  # Limit workers
        self.sub_batch_size = 100_000  # Larger sub-batches
        self.memory_monitor_interval = 500_000  # Monitor memory every 500k tracks
    
    def _prewarm_cache(self, db_reader):
        """Warm up database cache to improve performance."""
        print("  Warming up database cache...")
        # Fetch first 1000 tracks to load indexes into memory
        for _ in db_reader.get_batch_tracks(1000, 0):
            pass
    
    def run(self):
        """Run optimized preprocessing pipeline with memory management."""
        print("\n" + "═" * 65)
        print("  🚀 Starting Optimized Database Preprocessing")
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
        
        print("\n  🔥 Processing tracks in batches of 1,000,000 with parallelization...")
        print(f"  Using {self.workers} workers with sub-batch size {self.sub_batch_size}")
        print("  This will take about 30-60 minutes. Press Ctrl+C to interrupt.")
        
        # Initialize progress tracker
        progress = ProgressTracker(actual_total)
        last_rowid = 0
        processed_since_last_gc = 0
        batch_count = 0

        with VectorWriter(self.output_dir) as writer:
            with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                # Pre-warm database cache
                self._prewarm_cache(db_reader)
                
                # Create process pool with limited workers
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
                    while last_rowid < actual_total:
                        try:
                            batch_count += 1
                            
                            # Fetch batch of tracks
                            batch = db_reader.get_batch_tracks(self.batch_size, last_rowid)
                            if not batch:
                                break
                            
                            # Process in sub-batches
                            for i in range(0, len(batch), self.sub_batch_size):
                                sub_batch = batch[i:i+self.sub_batch_size]
                                
                                # Build vectors in parallel
                                vectors = list(executor.map(VectorBuilder.build_vectors, [sub_batch]))[0]
                                
                                # Write vectors with ISRC
                                for track_data, vector in zip(sub_batch, vectors):
                                    isrc = track_data.get('external_id_isrc', '')
                                    writer.write_vector(
                                        track_data['track_id'], 
                                        vector,
                                        isrc
                                    )
                                    last_rowid = track_data['rowid']
                                
                                # Update progress
                                progress.update(len(sub_batch))
                                processed_since_last_gc += len(sub_batch)
                                
                                # Memory management
                                if processed_since_last_gc >= self.memory_monitor_interval:
                                    # Explicit garbage collection
                                    del vectors
                                    gc.collect()
                                    processed_since_last_gc = 0
                            
                            # End of batch cleanup
                            del batch
                            gc.collect()
                            
                            # Periodically flush write buffers
                            if batch_count % 10 == 0:
                                writer._flush_buffers()
                            
                        except KeyboardInterrupt:
                            print("\n\n  ⏸️  Processing interrupted by user.")
                            print("  Partially processed data has been saved.")
                            return False
                        except Exception as e:
                            print(f"\n\n  ❗ Error during processing: {e}")
                            return False
        
        # Show completion
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
            self._show_statistics(progress.total, masks_path)
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
