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
import struct
import shutil
import pstats
from config import PathConfig, EXPECTED_VECTORS
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
from core.utilities.region_utils import REGION_MAPPING, get_region_from_isrc
from ui.cli.console_utils import print_header
from typing import List, Dict, Any, Generator

# The SQLite parameter limit is 999
MAX_SQL_VARS = 998

class DatabaseReader:
    """Optimized database reader with memory mapping and caching."""
    
    def __init__(self, main_db_path, audio_db_path):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.main_db = None
        self.audio_db = None
        
        # NEW: NumPy-based artist metadata caches
        self.artist_followers = None
        
        # NEW: Pre-computed artist genre sets (this is the key fix!)
        self.artist_genres_sets = None
        
        # NEW: Region lookup table (26x26 = 676 entries)
        self.region_lut = None
    
    def __enter__(self):
        """Open database connections with optimizations."""
        self.main_db = sqlite3.connect(self.main_db_path, timeout=30.0)
        self.main_db.execute("PRAGMA journal_mode = MEMORY")
        self.main_db.execute("PRAGMA cache_size = -200000")
        self.main_db.execute("PRAGMA temp_store = MEMORY")
        self.main_db.execute("PRAGMA synchronous = OFF")
        
        # Open audio database
        self.audio_db = sqlite3.connect(self.audio_db_path, timeout=30.0)
        self.audio_db.execute("PRAGMA journal_mode = MEMORY")
        self.audio_db.execute("PRAGMA synchronous = OFF")
        
        # Memory map databases
        self._memory_map_databases()
        
        # Preload optimized metadata structures
        self._preload_artist_metadata_numpy()
        
        self._preload_region_lut()
        
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
            with open(self.main_db_path, 'rb') as f:
                self.main_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            with open(self.audio_db_path, 'rb') as f:
                self.audio_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            print(f"  ⚠️ Could not memory-map databases: {e}")
    
    def _preload_artist_metadata_numpy(self):
        """
        Preload artist metadata into NumPy arrays for fast vectorized access.
        Pre-computes genre sets for each artist to avoid set building during streaming.
        """
        cursor = self.main_db.cursor()
        
        # Size arrays based on max rowid
        cursor.execute("SELECT MAX(rowid) FROM artists")
        max_rowid = cursor.fetchone()[0] + 1
        
        self.artist_followers = np.zeros(max_rowid, dtype=np.int64)
        self.artist_genres_sets = [set() for _ in range(max_rowid)]
        
        # Load followers (vectorized)
        cursor.execute("SELECT rowid, followers_total FROM artists")
        for rowid, followers in cursor:
            if rowid < max_rowid:
                self.artist_followers[rowid] = followers
        
        # Load genres into pre-built sets
        cursor.execute("SELECT artist_rowid, genre FROM artist_genres ORDER BY artist_rowid")
        for artist_rowid, genre in cursor:
            if artist_rowid < max_rowid:
                self.artist_genres_sets[artist_rowid].add(genre)
        
        cursor.close()

    def _preload_region_lut(self):
        """Precompute ISRC-to-region lookup table."""
        self.region_lut = np.full(26*26, 7, dtype=np.uint8)  # Default "Other" (7)
        
        for region_id, countries in REGION_MAPPING.items():
            for country in countries:
                if len(country) == 2:
                    idx = (ord(country[0]) - 65) * 26 + (ord(country[1]) - 65)
                    if 0 <= idx < 676:
                        self.region_lut[idx] = region_id
        
        # print(f"     Preloaded region lookup table for {len(REGION_MAPPING)} regions")
    
    def _get_region_batch(self, isrcs: List[str]) -> np.ndarray:
        """Vectorized region lookup for entire batch."""
        if not isrcs:
            return np.array([], dtype=np.uint8)
        
        n = len(isrcs)
        first_chars = np.zeros(n, dtype=np.int32)
        second_chars = np.zeros(n, dtype=np.int32)
        
        for i, isrc in enumerate(isrcs):
            if len(isrc) >= 2:
                first = isrc[0].upper()
                second = isrc[1].upper()
                if 'A' <= first <= 'Z' and 'A' <= second <= 'Z':
                    first_chars[i] = ord(first) - 65
                    second_chars[i] = ord(second) - 65
        
        indices = first_chars * 26 + second_chars
        return self.region_lut[indices]
    
    def _get_artist_data_batch(self, artist_ids_batch: List[List[int]]) -> tuple:
        """
        Batch artist data retrieval using NumPy.
        Use pre-computed genre sets instead of building them on the fly.
        """
        batch_size = len(artist_ids_batch)
        max_followers = np.zeros(batch_size, dtype=np.int64)
        genres_batch = []
        
        for i, artist_ids in enumerate(artist_ids_batch):
            if not artist_ids:
                max_followers[i] = 0
                genres_batch.append([])
                continue
            
            # Vectorized follower lookup
            followers = self.artist_followers[artist_ids]
            max_followers[i] = np.max(followers) if len(followers) > 0 else 0
            
            # NEW: Union pre-computed genre sets (much faster than building from scratch!)
            if len(artist_ids) == 1:
                # Single artist case - just copy the set
                genres = list(self.artist_genres_sets[artist_ids[0]])
            else:
                # Multi-artist case - union the sets
                combined_genres = set()
                for artist_id in artist_ids:
                    combined_genres.update(self.artist_genres_sets[artist_id])
                genres = list(combined_genres)
            
            genres_batch.append(genres)
        
        return max_followers, genres_batch
    
    def get_track_count(self):
        """Get total number of tracks."""
        cursor = self.main_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    
    def stream_tracks(self, batch_size=500000, last_rowid=0) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream tracks with optimized data fetching."""
        cursor = self.main_db.cursor()
        
        track_query = """
            SELECT 
                t.rowid,
                t.id as track_id,
                t.external_id_isrc,
                t.duration_ms,
                t.popularity,
                a.release_date
            FROM tracks t
            JOIN albums a ON t.album_rowid = a.rowid
            WHERE t.rowid > ?
            ORDER BY t.rowid
            LIMIT ?
        """
        
        batch_num = 0
        while True:
            
            try:
                cursor.execute(track_query, (last_rowid, batch_size))
                self.main_db.execute("PRAGMA busy_timeout = 5000")
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    print(f"  ❗️ Database locked error: {e}")
                    print("  🔄 Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    raise
            
            if not rows:
                break
            
            batch_num += 1
            
            # Process rows efficiently
            batch = []
            track_ids_for_audio = []
            track_rowids = []
            isrcs = []
            
            for row in rows:
                rowid = row[0]
                track_rowids.append(rowid)
                
                isrc = row[2] if row[2] else ""
                isrcs.append(isrc)
                
                track_batch_data = {
                    'rowid': rowid,
                    'track_id': row[1],
                    'external_id_isrc': isrc,
                    'duration_ms': row[3],
                    'popularity': row[4],
                    'release_date': str(row[5])
                }
                batch.append(track_batch_data)
                track_ids_for_audio.append(row[1])
                last_rowid = rowid
            
            # Vectorized region lookup
            regions = self._get_region_batch(isrcs)
            
            # Batch artist fetch
            artist_map = self._get_artist_ids_batch(cursor, track_rowids)

            max_followers, genres_list = self._get_artist_data_batch(list(artist_map.values()))
            
            # Batch audio features
            audio_features_map = self._get_audio_features_bulk(track_ids_for_audio)
            
            # Merge all data
            enriched_batch = []
            for i, track_data in enumerate(batch):
                artists = artist_map.get(track_data['rowid'], [])
                track_data['max_followers'] = max_followers[i]
                track_data['genres'] = genres_list[i]
                track_data['region'] = regions[i]
                track_data.update(audio_features_map.get(track_data['track_id'], {}))
                enriched_batch.append(track_data)
            
            yield enriched_batch
            gc.collect()
    
    def _get_artist_ids_batch(self, cursor, track_rowids: List[int]) -> Dict[int, List[int]]:
        """Batch-fetch artist IDs for all tracks."""
        artist_map = {}
        
        for chunk_start in range(0, len(track_rowids), MAX_SQL_VARS):
            chunk_end = min(chunk_start + MAX_SQL_VARS, len(track_rowids))
            chunk_trackids = track_rowids[chunk_start:chunk_end]
            
            placeholders = ','.join(['?'] * len(chunk_trackids))
            artist_query = f"""
                SELECT track_rowid, artist_rowid
                FROM track_artists
                WHERE track_rowid IN ({placeholders})
            """
            cursor.execute(artist_query, chunk_trackids)
            
            for track_rowid, artist_rowid in cursor:
                if track_rowid not in artist_map:
                    artist_map[track_rowid] = []
                artist_map[track_rowid].append(artist_rowid)
        
        return artist_map
    
    def _get_audio_features_bulk(self, track_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch audio features using chunked direct IN query."""
        if not track_ids:
            return {}
        
        audio_features = {}
        audio_cursor = self.audio_db.cursor()
        
        for chunk_start in range(0, len(track_ids), MAX_SQL_VARS):
            chunk_end = min(chunk_start + MAX_SQL_VARS, len(track_ids))
            chunk_trackids = track_ids[chunk_start:chunk_end]
            
            placeholders = ','.join(['?'] * len(chunk_trackids))
            query = f"""
                SELECT track_id, danceability, energy, loudness, speechiness,
                       acousticness, instrumentalness, liveness, valence,
                       tempo, time_signature, key, mode
                FROM track_audio_features
                WHERE track_id IN ({placeholders})
            """
            audio_cursor.execute(query, chunk_trackids)
            
            for row in audio_cursor:
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
        
        audio_cursor.close()
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
        self.output_dir = Path(output_dir)
        self.batch_size = 500000
        self.vector_batch_size = 100000
        self.total_vectors = 256_000_000
        self.enable_profiling = enable_profiling
        self.profile_interval = profile_interval
        self.profiler = None
    
    def run(self):
        """Run complete preprocessing pipeline (vectors + sorted index)."""
        print_header("Starting Database Preprocessing")
        
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
        if self._is_vectors_file_complete(vectors_path) and index_path.exists():
            print("\n  ✅ Preprocessing already complete!")
            return True
        
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
                print("\n  ⚠️ Corrupted checkpoint file. Starting from beginning...")
        
        # Initialize profiling
        last_profile_count = resume_from
        if self.enable_profiling:
            print(f"  🔍 Performance profiling enabled (every {self.profile_interval:,} vectors)")
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        try:
            # Pass 1: Write vectors and unsorted temp index
            with UnifiedVectorWriter(self.output_dir, resume_from) as writer:
                with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                    for batch in db_reader.stream_tracks(self.batch_size, resume_from):
                        # Process in vector batches
                        for i in range(0, len(batch), self.vector_batch_size):
                            vector_batch = batch[i:i+self.vector_batch_size]
                            
                            # Build and write vectors
                            vectors = build_track_vectors_batch(vector_batch)
                            
                            for j, track_data in enumerate(vector_batch):
                                writer.write_record(
                                    track_data['track_id'],
                                    vectors[j],
                                    track_data.get('external_id_isrc', ''),
                                    track_data['region']
                                )
                            
                            # Update progress
                            processed_count = len(vector_batch)
                            progress.update(processed_count)
                            resume_from += processed_count
                            
                            # Save checkpoint
                            last_rowid = vector_batch[-1]['rowid']
                            with open(checkpoint_path, "w") as f:
                                f.write(str(last_rowid))
                        
                        # Profiling checkpoint
                        if self.enable_profiling:
                            if resume_from - last_profile_count >= self.profile_interval:
                                print(f"\n  📊 Saving profile after {resume_from:,} vectors...")
                                self._save_profile_stats(resume_from)
                                last_profile_count = resume_from
            
            # Pass 2: Sort temp index into final sorted index
            print("\n  🔧 Building final sorted index...")
            self._sort_temp_index(writer.get_temp_index_dir(), index_path)
            
            # Update checksum in vectors file
            writer.update_header_checksum()
            
            # Cleanup temp directory
            shutil.rmtree(writer.get_temp_index_dir())
            
            # Remove checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Show statistics
            self._show_statistics(resume_from)
            
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
        """Check if vectors file exists and has correct size."""
        if not vectors_path.exists():
            return False
        
        expected_size = self.HEADER_SIZE + self.total_vectors * self.RECORD_SIZE
        actual_size = vectors_path.stat().st_size
        
        # Allow 1% margin
        return abs(actual_size - expected_size) <= expected_size * 0.01
    
    def _sort_temp_index(self, temp_index_dir: Path, output_path: Path):
        """
        External merge sort for index files.
        Phase 1: Sort 2M-entry chunks in memory.
        Phase 2: Merge sorted chunks with heap sort.
        """
        print("\n  Sorting index with external merge sort...")
        
        temp_files = sorted(temp_index_dir.glob("temp_index_*.bin"),
                           key=lambda f: int(f.stem.split('_')[-1]))
        
        if not temp_files:
            raise FileNotFoundError(f"No temp index files found in {temp_index_dir}")
        
        print(f"  Found {len(temp_files)} temp files to sort")
        
        # Phase 1: Sort chunks
        chunk_size = 2_000_000  # 2M entries × 26B = ~52MB per chunk
        temp_sorted_dir = temp_index_dir / "sorted_chunks"
        temp_sorted_dir.mkdir(exist_ok=True)
        
        chunk_files = []
        current_chunk = []
        chunk_idx = 0
        
        for temp_file in temp_files:
            with open(temp_file, "rb") as f:
                while True:
                    data = f.read(26)
                    if not data:
                        break
                    
                    track_id = data[:22].decode('ascii', 'ignore').rstrip('\0')
                    vector_index = struct.unpack("<I", data[22:26])[0]
                    current_chunk.append((track_id, vector_index))
                    
                    if len(current_chunk) >= chunk_size:
                        current_chunk.sort(key=lambda x: x[0])
                        
                        chunk_file = temp_sorted_dir / f"sorted_{chunk_idx:04d}.bin"
                        with open(chunk_file, "wb") as cf:
                            for tid, idx in current_chunk:
                                tid_bytes = tid.encode('ascii', 'ignore').ljust(22, b'\0')
                                cf.write(tid_bytes)
                                cf.write(struct.pack("<I", idx))
                        
                        chunk_files.append(chunk_file)
                        chunk_idx += 1
                        current_chunk = []
        
        # Final chunk
        if current_chunk:
            current_chunk.sort(key=lambda x: x[0])
            chunk_file = temp_sorted_dir / f"sorted_{chunk_idx:04d}.bin"
            with open(chunk_file, "wb") as cf:
                for tid, idx in current_chunk:
                    tid_bytes = tid.encode('ascii', 'ignore').ljust(22, b'\0')
                    cf.write(tid_bytes)
                    cf.write(struct.pack("<I", idx))
            chunk_files.append(chunk_file)
        
        print(f"  Created {len(chunk_files)} sorted chunks")
        
        # Phase 2: Merge with heap sort
        print("  Merging sorted chunks...")
        self._merge_chunks(chunk_files, output_path)
        
        shutil.rmtree(temp_sorted_dir)
        print(f"  ✅ Sorted index written to {output_path}")
    
    def _merge_chunks(self, chunk_files, output_path):
        """Merge sorted chunks using heap sort."""
        import heapq
        import struct
        
        files = [open(f, "rb") for f in chunk_files]
        records = []
        
        for i, f in enumerate(files):
            data = f.read(26)
            if data:
                track_id = data[:22].decode('ascii', 'ignore').rstrip('\0')
                vector_index = struct.unpack("<I", data[22:26])[0]
                records.append((track_id, vector_index, i))
        
        heapq.heapify(records)
        
        with open(output_path, "wb") as out_file:
            processed = 0
            while records:
                track_id, vector_index, file_idx = heapq.heappop(records)
                
                tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                out_file.write(tid_bytes)
                out_file.write(struct.pack("<I", vector_index))
                
                next_data = files[file_idx].read(26)
                if next_data:
                    next_track_id = next_data[:22].decode('ascii', 'ignore').rstrip('\0')
                    next_vector_index = struct.unpack("<I", next_data[22:26])[0]
                    heapq.heappush(records, (next_track_id, next_vector_index, file_idx))
                
                processed += 1
                if processed % 10_000_000 == 0:
                    print(f"  Merged {processed:,} records...")
        
        for f in files:
            f.close()

    def _save_profile_stats(self, processed_count, final=False):
        """Save profiling statistics."""
        suffix = "final" if final else f"{processed_count//1000000}M"
        profile_path = Path(self.output_dir) / f"preprocessing_profile_{suffix}.prof"
        self.profiler.dump_stats(str(profile_path))
        
        print(f"\n  📊 Profiling data saved to: {profile_path}")
        print("  Use 'snakeviz' to visualize: snakeviz path/to/file.prof")
        
        stats = pstats.Stats(str(profile_path))
        print(f"\n  Top 10 Time Consumers ({suffix}):")
        stats.strip_dirs().sort_stats('cumulative').print_stats(10)
        
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
