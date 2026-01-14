# core/preprocessing/db_to_vectors.py
import sqlite3
import time
import numpy as np
from pathlib import Path
import mmap
import os
import sys
import cProfile
import pstats
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
from core.utilities.region_utils import REGION_MAPPING, get_region_from_isrc

class DatabaseReader:
    """High-performance database reader with cross-database joins."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_db = None
        self.audio_db = None
        self.main_mmap = None
        self.audio_mmap = None
        
    def __enter__(self):
        """Connect to both databases and enable memory mapping."""
        # Open main database
        self.main_db = sqlite3.connect(self.main_db_path)
        self.main_db.execute("PRAGMA mmap_size=30000000000")  # 30GB memory mapping
        
        # Open audio database
        self.audio_db = sqlite3.connect(self.audio_db_path)
        self.audio_db.execute("PRAGMA mmap_size=10000000000")  # 10GB memory mapping
        
        # Memory-map databases
        self._memory_map_databases()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        if self.main_db:
            self.main_db.close()
        if self.audio_db:
            self.audio_db.close()
        if self.main_mmap:
            self.main_mmap.close()
        if self.audio_mmap:
            self.audio_mmap.close()
        return False
    
    def _memory_map_databases(self):
        """Memory-map database files for direct access."""
        try:
            # Main database
            with open(self.main_db_path, 'rb') as f:
                self.main_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Audio features database
            with open(self.audio_db_path, 'rb') as f:
                self.audio_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            print(f"⚠️ Memory mapping failed: {e}")
    
    def get_track_count(self):
        """Get total number of tracks in the database."""
        cursor = self.main_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def stream_tracks(self, batch_size=500000, last_rowid=0):
        """Ultra-fast track streaming with chunked audio feature fetching."""
        main_cursor = self.main_db.cursor()
        audio_cursor = self.audio_db.cursor()
        
        # Query for main track data
        main_query = """
        SELECT 
            t.rowid,
            t.id,
            t.external_id_isrc,
            t.duration_ms,
            t.popularity,
            a.release_date,
            GROUP_CONCAT(ta.artist_rowid) as artist_ids
        FROM tracks t
        JOIN albums a ON t.album_rowid = a.rowid
        LEFT JOIN track_artists ta ON t.rowid = ta.track_rowid
        WHERE t.rowid > ?
        GROUP BY t.rowid
        ORDER BY t.rowid
        LIMIT ?
        """
        
        while True:
            main_cursor.execute(main_query, (last_rowid, batch_size))
            batch = main_cursor.fetchall()
            
            if not batch:
                break
            
            # Collect track IDs for audio features lookup
            track_ids = [row[1] for row in batch]
            
            # Fetch audio features in chunks to avoid SQL variable limit
            audio_features = {}
            chunk_size = 500  # Well below SQLite's 999 variable limit
            for i in range(0, len(track_ids), chunk_size):
                chunk = track_ids[i:i+chunk_size]
                placeholders = ','.join(['?'] * len(chunk))
                
                audio_query = f"""
                SELECT track_id, danceability, energy, loudness, speechiness, 
                       acousticness, instrumentalness, liveness, valence, 
                       tempo, time_signature, key, mode
                FROM track_audio_features 
                WHERE track_id IN ({placeholders})
                """
                audio_cursor.execute(audio_query, chunk)
                
                for row in audio_cursor.fetchall():
                    audio_features[row[0]] = dict(zip(
                        ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                         'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 
                         'key', 'mode'],
                        row[1:]
                    ))
            
            # Process batch
            enriched_batch = []
            for row in batch:
                track_data = {
                    'rowid': row[0],
                    'track_id': row[1],
                    'external_id_isrc': row[2],
                    'duration_ms': row[3],
                    'popularity': row[4],
                    'release_date': row[5],
                    'artist_ids': row[6] if row[6] else ""
                }
                
                # Add audio features if available
                if track_data['track_id'] in audio_features:
                    track_data.update(audio_features[track_data['track_id']])
                
                # Get max followers
                max_followers = self._get_max_followers(track_data['artist_ids'])
                track_data['max_followers'] = max_followers
                
                # Get genres
                genres = self._get_genres(track_data['artist_ids'])
                track_data['genres'] = genres
                
                enriched_batch.append(track_data)
                last_rowid = track_data['rowid']
            
            yield enriched_batch
        
        main_cursor.close()
        audio_cursor.close()

    def _get_max_followers(self, artist_ids_str):
        """Get max followers efficiently."""
        if not artist_ids_str:
            return 0
        
        artist_ids = [int(id) for id in artist_ids_str.split(',')]
        if not artist_ids:
            return 0
        
        cursor = self.main_db.cursor()
        placeholders = ','.join(['?'] * len(artist_ids))
        query = f"SELECT MAX(followers_total) FROM artists WHERE rowid IN ({placeholders})"
        cursor.execute(query, artist_ids)
        result = cursor.fetchone()[0]
        cursor.close()
        
        return result or 0

    def _get_genres(self, artist_ids_str):
        """Get genres efficiently."""
        if not artist_ids_str:
            return []
        
        artist_ids = [int(id) for id in artist_ids_str.split(',')]
        if not artist_ids:
            return []
        
        cursor = self.main_db.cursor()
        placeholders = ','.join(['?'] * len(artist_ids))
        query = f"SELECT DISTINCT genre FROM artist_genres WHERE artist_rowid IN ({placeholders})"
        cursor.execute(query, artist_ids)
        genres = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        return genres


class PreprocessingEngine:
    """High-performance preprocessing engine with optimized I/O."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors",
                 enable_profiling=False,
                 profile_interval=4_000_000):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = output_dir
        self.batch_size = 1_000_000
        self.vector_batch_size = 100_000
        self.total_vectors = 256_000_000
        self.enable_profiling = enable_profiling
        self.profile_interval = profile_interval
        self.profiler = None
    
    def run(self):
        """Run high-performance preprocessing pipeline."""
        print("\n" + "═" * 65)
        print("  🚀 Starting High-Performance Database Preprocessing")
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
                print(f"  Using estimated total: {self.total_vectors:,}")
        
        print("\n  🔥 Processing tracks with optimized streaming...")
        print("  This should complete in under 60 minutes. Press Ctrl+C to interrupt.")
        
        # Initialize profiling
        if self.enable_profiling:
            print(f"  🔍 Performance profiling enabled (every {self.profile_interval:,} vectors)")
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        try:
            # Initialize progress tracker
            progress = ProgressTracker(self.total_vectors)
            processed_count = 0
            last_profile_count = 0
            
            # Initialize vector writer
            with UnifiedVectorWriter(Path(self.output_dir)) as writer:
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
                                writer.write_record(
                                    track_data['track_id'], 
                                    vector,
                                    track_data.get('external_id_isrc', ''),
                                    get_region_from_isrc(track_data.get('external_id_isrc', ''))
                                )
                            
                            # Update progress
                            progress.update(len(vector_batch))
                            processed_count += len(vector_batch)
                        
                        # Explicit profiling checkpoint
                        if self.enable_profiling:
                            if processed_count - last_profile_count >= self.profile_interval:
                                print(f"\n  📊 Saving profile after {processed_count:,} vectors...")
                                self._save_profile_stats(processed_count)
                                last_profile_count = processed_count
                    
                    # Final progress update
                    progress.update(self.total_vectors - processed_count)
                
                # Finalize processing
                writer.finalize()
                progress.complete()
            
            # Show statistics
            self._show_statistics(processed_count)
            return True
        
        except Exception as e:
            print(f"\n  ❗ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Finalize profiling
            if self.enable_profiling and self.profiler:
                self.profiler.disable()
                self._save_profile_stats(processed_count, final=True)

    def _save_profile_stats(self, processed_count, final=False):
        """Save profiling statistics at current point."""
        suffix = "final" if final else f"{processed_count//1000000}M"
        profile_path = Path(self.output_dir) / f"preprocessing_profile_{suffix}.prof"
        self.profiler.dump_stats(str(profile_path))
        
        print(f"\n  📊 Profiling data saved to: {profile_path}")
        print("  Use 'snakeviz' to visualize: snakeviz path/to/file.prof")
        
        # Generate quick text report
        stats = pstats.Stats(str(profile_path))
        print(f"\n  Top 10 Time Consumers ({suffix}):")
        stats.strip_dirs().sort_stats('cumulative').print_stats(10)
        
        # Save full stats to file
        stats_path = Path(self.output_dir) / f"preprocessing_stats_{suffix}.txt"
        with open(stats_path, 'w') as f:
            stats = pstats.Stats(str(profile_path), stream=f)
            stats.sort_stats('cumulative').print_stats()
        
        print(f"  Full stats saved to: {stats_path}")
    
    def _show_statistics(self, total_processed):
        """Display processing statistics."""
        vectors_path = Path(self.output_dir) / "track_vectors.bin"
        index_path = Path(self.output_dir) / "track_index.bin"
        
        vectors_size = vectors_path.stat().st_size if vectors_path.exists() else 0
        index_size = index_path.stat().st_size if index_path.exists() else 0
        
        vectors_gb = vectors_size / (1024**3)
        index_gb = index_size / (1024**3)
        total_gb = vectors_gb + index_gb
        
        print("\n  📊 Processing Statistics:")
        print(f"    Total tracks processed: {total_processed:,}")
        print(f"    Vector file size: {vectors_gb:.1f} GB")
        print(f"    Index file size: {index_gb:.1f} GB")
        print(f"    Total disk space used: {total_gb:.1f} GB")
        print(f"    Output directory: {self.output_dir}")
        print("\n  ✅ Preprocessing complete! Ready for similarity search.")
