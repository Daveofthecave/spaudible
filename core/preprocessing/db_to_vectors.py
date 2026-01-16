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
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
from core.utilities.region_utils import get_region_from_isrc

class DatabaseReader:
    """Optimized database reader with memory mapping and caching."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_db = None
        self.audio_db = None
        self.artist_followers_cache = {}
        self.artist_genres_cache = {}
        
    def __enter__(self):
        """Open database connections with optimizations."""
        # Open main database
        self.main_db = sqlite3.connect(self.main_db_path)
        self.main_db.execute("PRAGMA journal_mode = WAL")
        self.main_db.execute("PRAGMA cache_size = -200000")
        self.main_db.execute("PRAGMA temp_store = MEMORY")
        self.main_db.execute("PRAGMA synchronous = OFF")
        self.main_db.execute("PRAGMA locking_mode = EXCLUSIVE")
        self.main_db.execute("PRAGMA mmap_size=268435456;")  # 256MB memory map
        self.main_db.execute("PRAGMA temp_store=MEMORY;")
        
        # Open audio database
        self.audio_db = sqlite3.connect(self.audio_db_path)
        self.audio_db.execute("PRAGMA journal_mode = WAL")
        self.audio_db.execute("PRAGMA synchronous = OFF")
        self.audio_db.execute("PRAGMA mmap_size=268435456;") # 256MB memory map
        self.audio_db.execute("PRAGMA temp_store=MEMORY;")
        
        # Memory map databases
        self._memory_map_databases()
        
        # Preload artist metadata
        self._preload_artist_metadata()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close database connections."""
        if self.main_db:
            self.main_db.close()
        if self.audio_db:
            self.audio_db.close()
        return False
    
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
        """Stream tracks with optimized data fetching."""
        cursor = self.main_db.cursor()
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
            batch = cursor.fetchall()
            
            if not batch:
                break
            
            # Preallocate arrays
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
            
            # Fetch audio features
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
        
        cursor.close()

    def _get_audio_features_bulk(self, track_ids: list) -> dict:
        """Ultra-optimized audio feature fetching with batched inserts."""
        if not track_ids:
            return {}
        
        audio_features = {}
        cursor = self.audio_db.cursor()
        
        try:
            # Create temporary table with index
            cursor.execute("CREATE TEMP TABLE tmp_track_ids (track_id TEXT PRIMARY KEY)")
            
            # Insert in large batches using executemany
            chunk_size = 50000
            insert_query = "INSERT INTO tmp_track_ids VALUES (?)"
            
            for i in range(0, len(track_ids), chunk_size):
                chunk = track_ids[i:i+chunk_size]
                cursor.executemany(insert_query, [(tid,) for tid in chunk])
            
            # Use index hint for efficient join
            cursor.execute("""
                SELECT t.track_id, af.danceability, af.energy, af.loudness, af.speechiness, 
                    af.acousticness, af.instrumentalness, af.liveness, af.valence, 
                    af.tempo, af.time_signature, af.key, af.mode
                FROM tmp_track_ids t
                LEFT JOIN track_audio_features af ON t.track_id = af.track_id
                /* Use index coverage */
                WHERE af.track_id IS NOT NULL
            """)
            
            # Directly build dictionary without intermediate steps
            audio_features = {
                row[0]: {
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
                for row in cursor.fetchall()
                if row[1] is not None  # Skip null features
            }
        finally:
            cursor.execute("DROP TABLE IF EXISTS tmp_track_ids")
            cursor.close()
        
        return audio_features

class PreprocessingEngine:
    """Preprocessing engine with profiling support."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors",
                 enable_profiling=False,
                 profile_interval=4_000_000):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = Path(output_dir)  # Convert to Path object
        self.batch_size = 500000
        self.vector_batch_size = 100000
        self.total_vectors = 256000000
        self.enable_profiling = enable_profiling
        self.profile_interval = profile_interval
        self.profiler = None
    
    def run(self):
        """Run resumable preprocessing pipeline."""
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
        
        # Check if vectors file is complete
        vectors_complete = self._is_vectors_file_complete(vectors_path)
        index_complete = self._is_index_file_complete(index_path)
        
        if vectors_complete and index_complete:
            print("\n  ✅ Preprocessing already complete!")
            return True
        
        # Initialize progress tracker
        resume_from = 0
        progress = ProgressTracker(self.total_vectors, initial_processed=resume_from)
        progress.batch_size = self.vector_batch_size
        
        # Resume from checkpoint if available
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    resume_from = int(f.read().strip())
                print(f"\n  🔍 Resuming from vector #{resume_from:,}")
                progress.update(resume_from)
            except:
                print("\n  ⚠️  Corrupted checkpoint file, starting from beginning")
        
        # Initialize profiling
        last_profile_count = resume_from
        if self.enable_profiling:
            print(f"  🔍 Performance profiling enabled (every {self.profile_interval:,} vectors)")
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        try:
            # Initialize vector writer in append mode if resuming
            with UnifiedVectorWriter(self.output_dir, resume_from) as writer:
                with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                    # Process in streaming batches
                    for batch in db_reader.stream_tracks(self.batch_size, resume_from):
                        # Process vector batches
                        for i in range(0, len(batch), self.vector_batch_size):
                            vector_batch = batch[i:i+self.vector_batch_size]
                            
                            # Build vectors for this batch
                            vectors = build_track_vectors_batch(vector_batch)
                            
                            # Write vectors - use enumerate to get proper index
                            for j, track_data in enumerate(vector_batch):
                                writer.write_record(
                                    track_data['track_id'], 
                                    vectors[j],  # Use j instead of i
                                    track_data.get('external_id_isrc', ''),
                                    get_region_from_isrc(track_data.get('external_id_isrc', ''))
                                )
                            
                            # Update progress and checkpoint
                            processed_count = len(vector_batch)
                            progress.update(processed_count)
                            resume_from += processed_count
                            
                            # Save checkpoint using last rowid
                            last_rowid = vector_batch[-1]['rowid']
                            with open(checkpoint_path, "w") as f:
                                f.write(str(last_rowid))
                        
                        # Explicit profiling checkpoint
                        if self.enable_profiling:
                            if resume_from - last_profile_count >= self.profile_interval:
                                print(f"\n  📊 Saving profile after {resume_from:,} vectors...")
                                self._save_profile_stats(resume_from)
                                last_profile_count = resume_from
                    
                    # Finalize processing
                    writer.finalize()
                    progress.complete()
            
            # Remove checkpoint file
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Show statistics
            self._show_statistics(resume_from)
            
            # Build index if vectors are complete but index is missing
            if not self._is_index_file_complete(index_path):
                print("\n  🔍 Building index from completed vectors...")
                self._build_index_from_vectors(vectors_path, index_path)
            
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
