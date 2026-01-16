# core/preprocessing/db_to_vectors.py
import sqlite3
import time
import numpy as np
from pathlib import Path
import gc
import os
import mmap
import sys
import cProfile
import pstats
import struct
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
from core.utilities.region_utils import get_region_from_isrc

class DatabaseReader:
    """Optimized database reader using direct file access."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_db = None
        self.audio_db = None
        self.main_file = None
        self.audio_file = None
        self.main_mmap = None
        self.audio_mmap = None
        self.artist_followers_cache = {}
        self.artist_genres_cache = {}
        
    def __enter__(self):
        """Open database connections with optimizations."""
        # Open main database for metadata
        self.main_db = sqlite3.connect(self.main_db_path)
        self.main_db.execute("PRAGMA journal_mode = OFF")
        self.main_db.execute("PRAGMA locking_mode = EXCLUSIVE")
        
        # Open audio database for features
        self.audio_db = sqlite3.connect(self.audio_db_path)
        self.audio_db.execute("PRAGMA journal_mode = OFF")
        self.audio_db.execute("PRAGMA locking_mode = EXCLUSIVE")
        
        # Memory map databases
        self._memory_map_databases()
        
        # Preload artist metadata
        self._preload_artist_metadata()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        if self.main_mmap:
            self.main_mmap.close()
        if self.audio_mmap:
            self.audio_mmap.close()
        if self.main_file:
            self.main_file.close()
        if self.audio_file:
            self.audio_file.close()
        if self.main_db:
            self.main_db.close()
        if self.audio_db:
            self.audio_db.close()
        return False
    
    def _memory_map_databases(self):
        """Memory-map database files for direct access."""
        try:
            # Memory-map main database
            self.main_file = open(self.main_db_path, 'rb')
            self.main_mmap = mmap.mmap(
                self.main_file.fileno(), 
                0, 
                access=mmap.ACCESS_READ
            )
            
            # Memory-map audio database
            self.audio_file = open(self.audio_db_path, 'rb')
            self.audio_mmap = mmap.mmap(
                self.audio_file.fileno(), 
                0, 
                access=mmap.ACCESS_READ
            )
        except Exception as e:
            print(f"  ⚠️  Could not memory-map databases: {e}")
    
    def _preload_artist_metadata(self):
        """Preload artist metadata into memory."""
        # Load artist followers
        cursor = self.main_db.cursor()
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
        """Get total number of tracks."""
        cursor = self.main_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def stream_tracks(self, batch_size=500000, last_rowid=0):
        """Stream tracks using direct file access."""
        # Find the starting position for tracks
        cursor = self.main_db.cursor()
        cursor.execute("SELECT MIN(rowid) FROM tracks")
        min_rowid = cursor.fetchone()[0]
        cursor.close()
        
        # Calculate positions based on rowid
        current_rowid = max(min_rowid, last_rowid)
        
        while True:
            # Read batch directly from memory-mapped file
            batch = self._read_tracks_batch(current_rowid, batch_size)
            if not batch:
                break
                
            # Process batch
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
            
            # Fetch audio features using direct access
            audio_features_map = self._get_audio_features_bulk(track_ids)
            
            # Process max followers and genres
            max_followers_list = []
            genres_list = []
            for artist_ids_str in artist_id_groups:
                artist_ids = [int(id) for id in artist_ids_str.split(',')] if artist_ids_str else []
                
                # Get max followers
                max_followers = 0
                for artist_id in artist_ids:
                    followers = self.artist_followers_cache.get(artist_id, 0)
                    if followers > max_followers:
                        max_followers = followers
                max_followers_list.append(max_followers)
                
                # Get genres
                genres = set()
                for artist_id in artist_ids:
                    if artist_id in self.artist_genres_cache:
                        genres.update(self.artist_genres_cache[artist_id])
                genres_list.append(list(genres))
            
            # Build batch
            enriched_batch = []
            for i in range(len(track_ids)):
                track_data = {
                    'rowid': rowids[i],
                    'track_id': track_ids[i],
                    'external_id_isrc': isrcs[i],
                    'duration_ms': durations[i],
                    'popularity': popularities[i],
                    'release_date': release_dates[i],
                    'max_followers': max_followers_list[i],
                    'genres': genres_list[i]
                }
                
                # Add audio features
                if track_ids[i] in audio_features_map:
                    track_data.update(audio_features_map[track_ids[i]])
                
                enriched_batch.append(track_data)
            
            yield enriched_batch
            gc.collect()  # Prevent memory bloat
            
            # Update for next batch
            current_rowid = batch[-1][0] + 1
        
    def _read_tracks_batch(self, start_rowid, batch_size):
        """Read tracks directly from memory-mapped file."""
        cursor = self.main_db.cursor()
        cursor.execute(
            """
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
            WHERE t.rowid >= ?
            GROUP BY t.rowid
            ORDER BY t.rowid
            LIMIT ?
            """,
            (start_rowid, batch_size)
        )
        return cursor.fetchall()

    def _get_audio_features_bulk(self, track_ids: list) -> dict:
        """Direct audio feature access using memory mapping."""
        audio_features = {}
        if not track_ids:
            return audio_features
        
        # Use audio database connection
        cursor = self.audio_db.cursor()
        try:
            # Create temporary table
            cursor.execute("CREATE TEMP TABLE tmp_track_ids (track_id TEXT PRIMARY KEY)")
            
            # Insert in batches
            chunk_size = 50000
            for i in range(0, len(track_ids), chunk_size):
                chunk = track_ids[i:i+chunk_size]
                cursor.executemany(
                    "INSERT INTO tmp_track_ids VALUES (?)", 
                    [(tid,) for tid in chunk]
                )
            
            # Join with audio features
            cursor.execute("""
                SELECT t.track_id, af.danceability, af.energy, af.loudness, af.speechiness, 
                       af.acousticness, af.instrumentalness, af.liveness, af.valence, 
                       af.tempo, af.time_signature, af.key, af.mode
                FROM tmp_track_ids t
                LEFT JOIN track_audio_features af ON t.track_id = af.track_id
                WHERE af.track_id IS NOT NULL
            """)
            
            # Build results
            for row in cursor.fetchall():
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
            cursor.execute("DROP TABLE IF EXISTS tmp_track_ids")
            cursor.close()
        
        return audio_features

class PreprocessingEngine:
    """Preprocessing engine optimized for maximum throughput."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors",
                 enable_profiling=False,
                 profile_interval=4_000_000):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = Path(output_dir)  # Convert to Path object
        self.batch_size = 1000000  # Larger batch size
        self.vector_batch_size = 200000
        self.total_vectors = 256000000
        self.enable_profiling = enable_profiling
        self.profile_interval = profile_interval
        self.profiler = None
    
    def run(self):
        """Simplified preprocessing pipeline with larger batches."""
        print("\n" + "═" * 65)
        print("  🚀 Starting Resumable Database Preprocessing")
        print("═" * 65)
        
        # Validate databases
        if not Path(self.main_db_path).exists():
            print(f"\n  ❗ Database not found: {self.main_db_path}")
            return False
        if not Path(self.audio_db_path).exists():
            print(f"\n  ❗ Database not found: {self.audio_db_path}")
            return False
        
        # Setup paths
        vectors_path = self.output_dir / "track_vectors.bin"
        index_path = self.output_dir / "track_index.bin"
        checkpoint_path = self.output_dir / "preprocessing_checkpoint.txt"
        
        # Initialize progress tracker
        resume_from = 0
        progress = ProgressTracker(self.total_vectors, initial_processed=resume_from)
        progress.batch_size = self.vector_batch_size
        
        try:
            # Initialize vector writer
            with UnifiedVectorWriter(self.output_dir, resume_from) as writer:
                with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                    # Process in large batches
                    for batch in db_reader.stream_tracks(self.batch_size, resume_from):
                        # Process vector batches
                        for i in range(0, len(batch), self.vector_batch_size):
                            vector_batch = batch[i:i+self.vector_batch_size]
                            
                            # Build vectors
                            vectors = build_track_vectors_batch(vector_batch)
                            
                            # Write vectors
                            for j, track_data in enumerate(vector_batch):
                                writer.write_record(
                                    track_data['track_id'], 
                                    vectors[j],
                                    track_data.get('external_id_isrc', ''),
                                    get_region_from_isrc(track_data.get('external_id_isrc', ''))
                                )
                            
                            # Update progress
                            processed_count = len(vector_batch)
                            progress.update(processed_count)
                            resume_from += processed_count
                    
                    # Finalize processing
                    writer.finalize()
                    progress.complete()
            
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
                self._save_profile_stats(resume_from, final=True)

    def _is_vectors_file_complete(self, vectors_path):
        """Check if vectors file is complete."""
        if not vectors_path.exists():
            return False
        
        # Calculate expected size
        expected_size = UnifiedVectorWriter.HEADER_SIZE + self.total_vectors * UnifiedVectorWriter.RECORD_SIZE
        return vectors_path.stat().st_size == expected_size

    def _is_index_file_complete(self, index_path):
        """Check if index file is complete."""
        if not index_path.exists():
            return False
        
        # Calculate expected size
        expected_size = self.total_vectors * 26  # 22B track ID + 4B index
        return index_path.stat().st_size == expected_size

    def _build_index_from_vectors(self, vectors_path, index_path):
        """Build index file from completed vectors file."""
        # Initialize reader
        reader = UnifiedVectorReader(vectors_path)
        total_vectors = reader.get_total_vectors()
        
        # Create index writer
        with open(index_path, "wb") as index_file:
            # Process in chunks
            chunk_size = 1_000_000
            for start_idx in range(0, total_vectors, chunk_size):
                end_idx = min(start_idx + chunk_size, total_vectors)
                num_vectors = end_idx - start_idx
                
                # Read metadata for chunk
                metadata = reader.get_vector_metadata_batch(start_idx, num_vectors)
                
                # Write to index file
                for track_id, vector_index in metadata:
                    tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                    index_file.write(tid_bytes)
                    index_file.write(struct.pack("<I", vector_index))
        
        print(f"  ✅ Index file created with {total_vectors:,} entries")

    def _save_profile_stats(self, processed_count, final=False):
        """Save profiling statistics."""
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
