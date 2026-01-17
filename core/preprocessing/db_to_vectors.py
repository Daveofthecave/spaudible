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
from config import PathConfig, EXPECTED_VECTORS
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
from core.utilities.region_utils import REGION_MAPPING, get_region_from_isrc
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
        print("  🔍 Debug: Starting artist metadata preload...")
        start = time.time()
        self._preload_artist_metadata_numpy()
        print(f"  ✅ Debug: Artist metadata loaded in {time.time() - start:.2f}s")
        
        print("  🔍 Debug: Starting region LUT preload...")
        start = time.time()
        self._preload_region_lut()
        print(f"  ✅ Debug: Region LUT loaded in {time.time() - start:.2f}s")
        
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
            print(f"  ⚠️  Could not memory-map databases: {e}")
    
    def _preload_artist_metadata_numpy(self):
        """
        OPTIMIZATION 2: Load artist metadata into NumPy arrays.
        KEY FIX: Pre-compute genre SETS for each artist to avoid set building during streaming.
        """
        cursor = self.main_db.cursor()
        
        # Get max rowid for array sizing
        cursor.execute("SELECT MAX(rowid) FROM artists")
        max_rowid = cursor.fetchone()[0] + 1
        
        # Initialize arrays
        self.artist_followers = np.zeros(max_rowid, dtype=np.int64)
        
        # NEW: Pre-computed genre sets for each artist
        self.artist_genres_sets = [set() for _ in range(max_rowid)]
        
        # Load followers (vectorized)
        print(f"  🔍 Debug: Loading {max_rowid:,} artist followers...")
        followers_start = time.time()
        cursor.execute("SELECT rowid, followers_total FROM artists")
        for rowid, followers in cursor:
            if rowid < max_rowid:
                self.artist_followers[rowid] = followers
        print(f"  ✅ Debug: Followers loaded in {time.time() - followers_start:.2f}s")
        
        # Load genres into pre-built sets (this is the key optimization!)
        print(f"  🔍 Debug: Loading artist genres into pre-computed sets...")
        genres_start = time.time()
        cursor.execute("SELECT artist_rowid, genre FROM artist_genres ORDER BY artist_rowid")
        genre_count = 0
        for artist_rowid, genre in cursor:
            if artist_rowid < max_rowid:
                self.artist_genres_sets[artist_rowid].add(genre)
                genre_count += 1
        
        print(f"  ✅ Debug: {genre_count:,} genres loaded into sets in {time.time() - genres_start:.2f}s")
        cursor.close()
        
        print(f"  📊 Loaded {max_rowid:,} artists with pre-computed genre sets")
    
    def _preload_region_lut(self):
        """OPTIMIZATION 1: Precompute ISRC-to-region lookup table."""
        self.region_lut = np.full(26*26, 7, dtype=np.uint8)  # Default "Other" (7)
        
        for region_id, countries in REGION_MAPPING.items():
            for country in countries:
                if len(country) == 2:
                    idx = (ord(country[0]) - 65) * 26 + (ord(country[1]) - 65)
                    if 0 <= idx < 676:
                        self.region_lut[idx] = region_id
        
        print(f"  📊 Preloaded region lookup table for {len(REGION_MAPPING)} regions")
    
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
        OPTIMIZATION 2: Batch artist data retrieval using NumPy.
        KEY FIX: Use pre-computed genre sets instead of building them on the fly.
        """
        batch_size = len(artist_ids_batch)
        max_followers = np.zeros(batch_size, dtype=np.int64)
        genres_batch = []
        
        print(f"  🔍 Debug: Processing {batch_size:,} tracks with artist data...")
        start_time = time.time()
        
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
        
        elapsed = time.time() - start_time
        print(f"  ✅ Debug: Artist data processed in {elapsed:.2f}s ({batch_size/elapsed:.0f} tracks/sec)")
        
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
            print(f"  🔍 Debug: Fetching batch {batch_num} starting from rowid {last_rowid}")
            start_time = time.time()
            
            try:
                cursor.execute(track_query, (last_rowid, batch_size))
                self.main_db.execute("PRAGMA busy_timeout = 5000")
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    print(f"  ❌ Database locked error: {e}")
                    print("  🔄 Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    raise
            
            elapsed = time.time() - start_time
            print(f"  ✅ Debug: Batch {batch_num} fetched {len(rows)} rows in {elapsed:.2f}s")
            
            if not rows:
                print(f"  🔍 Debug: No more rows, exiting stream")
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
            print(f"  🔍 Debug: Looking up regions for {len(isrcs)} ISRCs...")
            regions = self._get_region_batch(isrcs)
            print(f"  ✅ Debug: Regions retrieved")
            
            # Batch artist fetch
            print(f"  🔍 Debug: Fetching artist IDs...")
            artist_map = self._get_artist_ids_batch(cursor, track_rowids)
            print(f"  ✅ Debug: Artist IDs fetched")
            
            print(f"  🔍 Debug: Getting artist data...")
            max_followers, genres_list = self._get_artist_data_batch(list(artist_map.values()))
            print(f"  ✅ Debug: Artist data retrieved")
            
            # Batch audio features
            print(f"  🔍 Debug: Fetching audio features...")
            audio_features_map = self._get_audio_features_bulk(track_ids_for_audio)
            print(f"  ✅ Debug: Audio features retrieved")
            
            # Merge all data
            enriched_batch = []
            for i, track_data in enumerate(batch):
                artists = artist_map.get(track_data['rowid'], [])
                track_data['max_followers'] = max_followers[i]
                track_data['genres'] = genres_list[i]
                track_data['region'] = regions[i]
                track_data.update(audio_features_map.get(track_data['track_id'], {}))
                enriched_batch.append(track_data)
            
            print(f"  ✅ Debug: Yielding batch of {len(enriched_batch)} tracks")
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
            # Initialize vector writer
            with UnifiedVectorWriter(self.output_dir, resume_from) as writer:
                with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                    # Process in streaming batches
                    first_batch = True
                    
                    for batch in db_reader.stream_tracks(self.batch_size, resume_from):
                        # Process vector batches
                        for i in range(0, len(batch), self.vector_batch_size):
                            vector_batch = batch[i:i+self.vector_batch_size]
                            
                            # Build vectors for this batch
                            vectors = build_track_vectors_batch(vector_batch)
                            
                            # Write vectors
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
                            
                            # Force progress bar after first batch
                            if first_batch:
                                print(f"  ✅ First batch processed! Progress bar should now appear.")
                                first_batch = False
                        
                        # Profiling checkpoint
                        if self.enable_profiling:
                            if resume_from - last_profile_count >= self.profile_interval:
                                print(f"\n  📊 Saving profile after {resume_from:,} vectors...")
                                self._save_profile_stats(resume_from)
                                last_profile_count = resume_from
                    
                    # Finalize
                    writer.finalize()
                    progress.complete()
            
            # Remove checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Show statistics
            self._show_statistics(resume_from)
            
            # Build index if needed
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
        from core.preprocessing.unified_vector_reader import UnifiedVectorReader
        reader = UnifiedVectorReader(vectors_path)
        total_vectors = reader.get_total_vectors()
        
        if total_vectors != self.total_vectors:
            print(f"  ❗ Vector file has {total_vectors:,} vectors, expected {self.total_vectors:,}")
            return
        
        temp_dir = self.output_dir / "temp_index"
        temp_dir.mkdir(exist_ok=True)
        
        # Process in chunks
        chunk_size = 1_000_000
        chunk_files = []
        num_chunks = (total_vectors + chunk_size - 1) // chunk_size
        
        print(f"  Processing {total_vectors:,} vectors in {num_chunks} chunks")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_vectors)
            num_vectors = end_idx - start_idx
            
            print(f"  Chunk {chunk_idx+1}/{num_chunks}: vectors {start_idx:,}-{end_idx-1:,}")
            
            metadata = reader.get_vector_metadata_batch(start_idx, num_vectors)
            metadata.sort(key=lambda x: x[0])
            
            chunk_file = temp_dir / f"chunk_{chunk_idx}.bin"
            with open(chunk_file, "wb") as f:
                for track_id, vector_index in metadata:
                    tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                    f.write(tid_bytes)
                    f.write(int.to_bytes(vector_index, 4, 'little'))
            
            chunk_files.append(chunk_file)
        
        # Merge sorted chunks
        print("  Merging sorted chunks...")
        self._merge_chunks(chunk_files, index_path)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"  ✅ Index file created with {total_vectors:,} entries")

    def _merge_chunks(self, chunk_files, output_path):
        """Merge sorted chunk files into final index."""
        import struct
        import heapq
        
        files = [open(f, "rb") for f in chunk_files]
        records = [None] * len(files)
        
        for i, f in enumerate(files):
            tid_bytes = f.read(22)
            if tid_bytes:
                index_bytes = f.read(4)
                track_id = tid_bytes.decode('ascii', 'ignore').rstrip('\0')
                vector_index = struct.unpack("<I", index_bytes)[0]
                records[i] = (track_id, vector_index, i)
        
        heap = []
        for i, rec in enumerate(records):
            if rec is not None:
                heapq.heappush(heap, (rec[0], rec[1], rec[2]))
        
        with open(output_path, "wb") as out_file:
            processed = 0
            while heap:
                track_id, vector_index, file_idx = heapq.heappop(heap)
                
                tid_bytes = track_id.encode('ascii', 'ignore').ljust(22, b'\0')
                out_file.write(tid_bytes)
                out_file.write(struct.pack("<I", vector_index))
                
                processed += 1
                if processed % 5_000_000 == 0:
                    print(f"  Merged {processed:,} records...")
                
                tid_bytes = files[file_idx].read(22)
                if tid_bytes:
                    index_bytes = files[file_idx].read(4)
                    track_id = tid_bytes.decode('ascii', 'ignore').rstrip('\0')
                    vector_index = struct.unpack("<I", index_bytes)[0]
                    heapq.heappush(heap, (track_id, vector_index, file_idx))
        
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
