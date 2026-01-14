# core/preprocessing/db_to_vectors.py
import sqlite3
import time
import numpy as np
from pathlib import Path
import gc
import os
import sys
import cProfile
import pstats
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
from core.utilities.region_utils import REGION_MAPPING, get_region_from_isrc

class DatabaseReader:
    """Ultra-optimized database reader with vectorized processing."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_db = None
        self.audio_db = None
        
    def __enter__(self):
        """Open database connections with aggressive optimizations."""
        # Open main database
        self.main_db = sqlite3.connect(self.main_db_path)
        self.main_db.execute("PRAGMA journal_mode = MEMORY")
        self.main_db.execute("PRAGMA cache_size = -200000")  # 200GB cache
        self.main_db.execute("PRAGMA temp_store = MEMORY")
        self.main_db.execute("PRAGMA synchronous = OFF")
        
        # Open audio database
        self.audio_db = sqlite3.connect(self.audio_db_path)
        self.audio_db.execute("PRAGMA journal_mode = MEMORY")
        self.audio_db.execute("PRAGMA synchronous = OFF")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close database connections."""
        if self.main_db:
            self.main_db.close()
        if self.audio_db:
            self.audio_db.close()
        return False
    
    def get_track_count(self):
        """Get total number of tracks in the database."""
        cursor = self.main_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def stream_tracks(self, batch_size=500000, last_rowid=0):
        """Vectorized track streaming with optimized SQL."""
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
            
            # Preallocate arrays for vectorization
            track_ids = []
            isrcs = []
            durations = []
            popularities = []
            release_dates = []
            artist_id_groups = []
            rowids = []
            
            for row in batch:
                rowids.append(row[0])
                track_ids.append(row[1])
                isrcs.append(row[2])
                durations.append(row[3])
                popularities.append(row[4])
                release_dates.append(row[5])
                artist_id_groups.append(row[6] if row[6] else "")
                last_rowid = row[0]
            
            # Fetch audio features in bulk
            audio_features_map = self._get_audio_features_bulk(track_ids)
            
            # Process max followers and genres in bulk
            max_followers_list = self._get_max_followers_bulk(artist_id_groups)
            genres_list = self._get_genres_bulk(artist_id_groups)
            
            # Build batch
            enriched_batch = []
            for i in range(len(track_ids)):
                track_data = {
                    'track_id': track_ids[i],
                    'external_id_isrc': isrcs[i],
                    'duration_ms': durations[i],
                    'popularity': popularities[i],
                    'release_date': release_dates[i],
                    'max_followers': max_followers_list[i],
                    'genres': genres_list[i]
                }
                
                # Add audio features if available
                if track_ids[i] in audio_features_map:
                    track_data.update(audio_features_map[track_ids[i]])
                
                enriched_batch.append(track_data)
            
            yield enriched_batch
        
        main_cursor.close()
        audio_cursor.close()

    def _get_audio_features_bulk(self, track_ids):
        """Fetch audio features using temporary tables to avoid variable limits."""
        if not track_ids:
            return {}
        
        audio_cursor = self.audio_db.cursor()
        audio_features = {}
        
        try:
            # Create temporary table
            audio_cursor.execute("CREATE TEMP TABLE tmp_track_ids (track_id TEXT PRIMARY KEY)")
            
            # Insert track IDs in batches using executemany
            chunk_size = 10000
            for i in range(0, len(track_ids), chunk_size):
                chunk = track_ids[i:i+chunk_size]
                # Convert to list of tuples [(id1,), (id2,), ...]
                values = [(track_id,) for track_id in chunk]
                audio_cursor.executemany("INSERT INTO tmp_track_ids VALUES (?)", values)
            
            # Join with audio features
            query = """
            SELECT t.track_id, af.danceability, af.energy, af.loudness, af.speechiness, 
                   af.acousticness, af.instrumentalness, af.liveness, af.valence, 
                   af.tempo, af.time_signature, af.key, af.mode
            FROM tmp_track_ids t
            JOIN track_audio_features af ON t.track_id = af.track_id
            """
            audio_cursor.execute(query)
            
            for row in audio_cursor.fetchall():
                audio_features[row[0]] = {
                    'danceability': row[1],
                    'energy': row[2],
                    'loudness': row[3],
                    'speechiness': row[4],
                    'acousticness': row[5],
                    'instrumentalness': row[6],
                    'liveness': row[7],
                    'valence': row[8],
                    'tempo': row[9],
                    'time_signature': row[10],
                    'key': row[11],
                    'mode': row[12]
                }
        
        finally:
            # Clean up temporary table
            audio_cursor.execute("DROP TABLE IF EXISTS tmp_track_ids")
            audio_cursor.close()
        
        return audio_features

    def _get_max_followers_bulk(self, artist_id_groups):
        """Vectorized processing of max followers using temp tables."""
        max_followers_list = []
        cursor = self.main_db.cursor()
        
        try:
            # Create temporary table for artist IDs
            cursor.execute("CREATE TEMP TABLE tmp_artist_ids (artist_id INTEGER PRIMARY KEY)")
            
            # Collect all artist IDs
            all_artist_ids = set()
            for artist_ids_str in artist_id_groups:
                if artist_ids_str:
                    artist_ids = [int(id) for id in artist_ids_str.split(',')]
                    all_artist_ids.update(artist_ids)
            
            # Insert artist IDs using executemany
            chunk_size = 10000
            artist_ids_list = list(all_artist_ids)
            for i in range(0, len(artist_ids_list), chunk_size):
                chunk = artist_ids_list[i:i+chunk_size]
                values = [(artist_id,) for artist_id in chunk]
                cursor.executemany("INSERT INTO tmp_artist_ids VALUES (?)", values)
            
            # Get max followers for all artists
            cursor.execute("""
            SELECT a.rowid, a.followers_total
            FROM artists a
            JOIN tmp_artist_ids t ON a.rowid = t.artist_id
            """)
            max_followers_map = dict(cursor.fetchall())
            
            # Map back to original groups
            for artist_ids_str in artist_id_groups:
                if not artist_ids_str:
                    max_followers_list.append(0)
                    continue
                
                artist_ids = [int(id) for id in artist_ids_str.split(',')]
                max_followers = max(
                    (max_followers_map.get(artist_id, 0) for artist_id in artist_ids),
                    default=0
                )
                max_followers_list.append(max_followers)
        
        finally:
            cursor.execute("DROP TABLE IF EXISTS tmp_artist_ids")
            cursor.close()
        
        return max_followers_list

    def _get_genres_bulk(self, artist_id_groups):
        """Vectorized processing of genres using temp tables."""
        genres_list = []
        cursor = self.main_db.cursor()
        
        try:
            # Create temporary table for artist IDs
            cursor.execute("CREATE TEMP TABLE tmp_artist_ids (artist_id INTEGER PRIMARY KEY)")
            
            # Collect all artist IDs
            all_artist_ids = set()
            for artist_ids_str in artist_id_groups:
                if artist_ids_str:
                    artist_ids = [int(id) for id in artist_ids_str.split(',')]
                    all_artist_ids.update(artist_ids)
            
            # Insert artist IDs using executemany
            chunk_size = 10000
            artist_ids_list = list(all_artist_ids)
            for i in range(0, len(artist_ids_list), chunk_size):
                chunk = artist_ids_list[i:i+chunk_size]
                values = [(artist_id,) for artist_id in chunk]
                cursor.executemany("INSERT INTO tmp_artist_ids VALUES (?)", values)
            
            # Get genres for all artists
            cursor.execute("""
            SELECT ag.artist_rowid, ag.genre
            FROM artist_genres ag
            JOIN tmp_artist_ids t ON ag.artist_rowid = t.artist_id
            """)
            genre_map = {}
            for row in cursor.fetchall():
                artist_id, genre = row
                if artist_id not in genre_map:
                    genre_map[artist_id] = []
                genre_map[artist_id].append(genre)
            
            # Map back to original groups
            for artist_ids_str in artist_id_groups:
                if not artist_ids_str:
                    genres_list.append([])
                    continue
                
                artist_ids = [int(id) for id in artist_ids_str.split(',')]
                genres = set()
                for artist_id in artist_ids:
                    genres.update(genre_map.get(artist_id, []))
                genres_list.append(list(genres))
        
        finally:
            cursor.execute("DROP TABLE IF EXISTS tmp_artist_ids")
            cursor.close()
        
        return genres_list
        
class PreprocessingEngine:
    """Final optimized preprocessing engine."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors",
                 enable_profiling=False,
                 profile_interval=4_000_000):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = output_dir
        self.batch_size = 500_000  # Optimized batch size
        self.vector_batch_size = 100_000
        self.total_vectors = 256_000_000
        self.enable_profiling = enable_profiling
        self.profile_interval = profile_interval
        self.profiler = None
    
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
