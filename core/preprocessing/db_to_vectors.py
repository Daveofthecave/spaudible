# core/preprocessing/db_to_vectors.py
import sqlite3
import time
import numpy as np
from pathlib import Path
from config import EXPECTED_VECTORS
import gc
import os
import mmap
import sys
import cProfile
import pstats
import duckdb
import pyarrow as pa
from typing import Iterator, Dict, Any, List, Tuple
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
from core.utilities.region_utils import get_region_from_isrc

def connect_duckdb(main_db_path: str, audio_db_path: str):
    temp_dir = Path(__file__).parent.parent.parent / "data" / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL sqlite; LOAD sqlite;")
    
    # Memory limits
    con.execute("SET memory_limit='4GB';")             # RAM limit
    con.execute("SET max_memory='3GB';")               # Per-query limit 
    con.execute(f"SET temp_directory='{temp_dir}';")
    con.execute("SET max_temp_directory_size='50GB';")
    con.execute("SET preserve_insertion_order=false;") # Allow reordering
    con.execute("SET threads=8;")                      # Single thread = minimal spill
    con.execute("SET sqlite_all_varchar=true;")

    con.execute(f"ATTACH '{main_db_path}' AS main_db (TYPE sqlite);")
    con.execute(f"ATTACH '{audio_db_path}' AS audio_db (TYPE sqlite);")
    return con

class DatabaseReader:
    """DuckDB-based streaming reader for track data with PyArrow integration."""
    
    def __init__(self, main_db_path: str, audio_db_path: str, batch_rows: int = 200_000):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.batch_rows = batch_rows
        self.con = None
        
    def __enter__(self):
        self.con = connect_duckdb(self.main_db_path, self.audio_db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up DuckDB connection."""
        if self.con:
            self.con.close()
        return False

    def get_track_count(self) -> int:
        """How many tracks are in the databases."""
        return EXPECTED_VECTORS
        
    def stream_tracks(self, batch_size: int = 25_000, last_rowid: int = 0):
        offset = last_rowid
        limit = batch_size
        offset_end = offset + limit
        
        self.con.execute("""
            CREATE OR REPLACE TEMP TABLE batch_artist_cache AS 
            SELECT DISTINCT 
                ta.artist_rowid::BIGINT as artist_rowid,
                MAX(ar.followers_total::BIGINT) as followers_total,
                list_distinct(list(ag.genre)) as genres
            FROM main_db.track_artists ta
            LEFT JOIN main_db.artists ar ON ta.artist_rowid::BIGINT = ar.rowid::BIGINT
            LEFT JOIN main_db.artist_genres ag ON ta.artist_rowid::BIGINT = ag.artist_rowid::BIGINT
            WHERE ta.track_rowid::BIGINT >= {offset} AND ta.track_rowid::BIGINT <= {offset_end}
            GROUP BY ta.artist_rowid
        """.format(offset=offset, offset_end=offset_end))
        
        where_clause = f"WHERE t.rowid::BIGINT >= {offset} AND t.rowid::BIGINT <= {offset_end}"
        
        query = f"""
        SELECT
            t.rowid::BIGINT as rowid,
            t.id as track_id,
            t.external_id_isrc,
            t.duration_ms::BIGINT as duration_ms,
            t.popularity::INTEGER as popularity,
            a.release_date,
            COALESCE(MAX(bac.followers_total), 0) as max_followers,
            COALESCE(bac.genres, []) as genres,
            COALESCE(af.danceability::FLOAT, 0.0) as danceability,
            COALESCE(af.energy::FLOAT, 0.0) as energy,
            COALESCE(af.loudness::FLOAT, -60.0) as loudness,
            COALESCE(af.speechiness::FLOAT, 0.0) as speechiness,
            COALESCE(af.acousticness::FLOAT, 0.0) as acousticness,
            COALESCE(af.instrumentalness::FLOAT, 0.0) as instrumentalness,
            COALESCE(af.liveness::FLOAT, 0.0) as liveness,
            COALESCE(af.valence::FLOAT, 0.0) as valence,
            COALESCE(af.tempo::FLOAT, 0.0) as tempo,
            COALESCE(af.time_signature::INTEGER, 4) as time_signature,
            COALESCE(af.key::INTEGER, 0) as key,
            COALESCE(af.mode::INTEGER, 0) as mode
        FROM main_db.tracks t
        JOIN main_db.albums a ON t.album_rowid = a.rowid 
        LEFT JOIN main_db.track_artists ta ON t.rowid::BIGINT = ta.track_rowid::BIGINT
        LEFT JOIN TEMP.batch_artist_cache bac ON ta.artist_rowid::BIGINT = bac.artist_rowid
        LEFT JOIN audio_db.track_audio_features af ON t.id = af.track_id
        {where_clause}
        GROUP BY t.rowid, t.id, t.external_id_isrc, t.duration_ms, t.popularity, 
                a.release_date, bac.followers_total, bac.genres,
                af.danceability, af.energy, af.loudness, af.speechiness, 
                af.acousticness, af.instrumentalness, af.liveness, af.valence, 
                af.tempo, af.time_signature, af.key, af.mode
        ORDER BY t.rowid
        LIMIT {limit}
        """
        
        rel = self.con.sql(query)
        reader = rel.fetch_record_batch(rows_per_batch=batch_size)
        
        for batch in reader:
            records = self._arrow_batch_to_records(batch)
            yield records

    def _arrow_batch_to_records(self, batch: pa.RecordBatch) -> List[Dict[str, Any]]:
        """Convert Arrow batch to exact dict format expected by vectorizer."""
        table = pa.Table.from_batches([batch])
        cols = {name: table[name].to_pylist() for name in table.schema.names}
        n = len(table)
        
        records = []
        for i in range(n):
            rec = {
                'rowid': int(cols['rowid'][i]),
                'track_id': cols['track_id'][i],
                'external_id_isrc': cols['external_id_isrc'][i] or '',
                'duration_ms': int(cols['duration_ms'][i] or 0),
                'popularity': int(cols['popularity'][i] or 0),
                'release_date': cols['release_date'][i],
                'max_followers': int(cols['max_followers'][i] or 0),
                'genres': cols['genres'][i] or [],
            }
            # Add audio features matching your original order
            audio_fields = [
                'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'key', 'mode'
            ]
            for field in audio_fields:
                rec[field] = float(cols[field][i])
            
            records.append(rec)
        return records


class PreprocessingEngine:
    """Preprocessing engine with DuckDB streaming and full resume support."""
    
    def __init__(self, 
                 main_db_path="data/databases/spotify_clean.sqlite3",
                 audio_db_path="data/databases/spotify_clean_audio_features.sqlite3",
                 output_dir="data/vectors",
                 enable_profiling=False,
                 profile_interval=4_000_000):
        self.main_db_path = main_db_path
        self.audio_db_path = audio_db_path
        self.output_dir = Path(output_dir)
        self.batch_size = 50_000  # Arrow batch size (was 200k)
        self.vector_batch_size = 25_000 # (was 100k)
        self.total_vectors = EXPECTED_VECTORS
        self.enable_profiling = enable_profiling
        self.profile_interval = profile_interval
        self.profiler = None
    
    def run(self):
        """Run resumable preprocessing pipeline with DuckDB streaming."""
        print("\n" + "═" * 65)
        print("  🚀 Starting DuckDB Streaming Preprocessing Pipeline")
        print("═" * 65)
        
        # Validate databases
        if not Path(self.main_db_path).exists():
            print(f"\n  ❗ Main database not found: {self.main_db_path}")
            return False
        if not Path(self.audio_db_path).exists():
            print(f"\n  ❗ Audio database not found: {self.audio_db_path}")
            return False
        
        # Setup paths
        vectors_path = self.output_dir / "track_vectors.bin"
        index_path = self.output_dir / "track_index.bin"
        checkpoint_path = self.output_dir / "preprocessing_checkpoint.txt"
        
        # Check if already complete
        vectors_complete = self._is_vectors_file_complete(vectors_path)
        index_complete = self._is_index_file_complete(index_path)
        
        if vectors_complete and index_complete:
            print("\n  ✅ Preprocessing already complete!")
            return True
        
        # Initialize progress and resume
        resume_rowid = 0
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    resume_rowid = int(f.read().strip())
                print(f"\n  🔍 Resuming from rowid {resume_rowid:,}")
            except:
                print("\n  ⚠️  Invalid checkpoint, starting from beginning")
        
        # Get total count for progress
        with DatabaseReader(self.main_db_path, self.audio_db_path) as reader:
            total_tracks = reader.get_track_count()
            print(f"\n  📊 Total tracks to process: {total_tracks:,}")
        
        progress = ProgressTracker(total_tracks, initial_processed=0)
        progress.batch_size = self.vector_batch_size
        
        # Initialize profiling
        last_profile_count = 0
        if self.enable_profiling:
            print(f"  🔍 Profiling enabled (every {self.profile_interval:,} vectors)")
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        try:
            with UnifiedVectorWriter(self.output_dir, 0) as writer:  # Let writer handle resume
                with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                    processed_count = 0
                    
                    for batch in db_reader.stream_tracks(self.batch_size, resume_rowid):
                        # Process vector sub-batches
                        for i in range(0, len(batch), self.vector_batch_size):
                            vector_batch = batch[i:i + self.vector_batch_size]
                            
                            if not vector_batch:
                                continue
                            
                            # Build vectors (existing function)
                            vectors = build_track_vectors_batch(vector_batch)
                            
                            # Write vectors with metadata
                            for j, track_data in enumerate(vector_batch):
                                writer.write_record(
                                    track_data['track_id'], 
                                    vectors[j],
                                    track_data.get('external_id_isrc', ''),
                                    get_region_from_isrc(track_data.get('external_id_isrc', ''))
                                )
                            
                            # Update progress
                            batch_processed = len(vector_batch)
                            processed_count += batch_processed
                            progress.update(batch_processed)
                        
                        # Save checkpoint using last rowid
                        if batch:
                            last_rowid = batch[-1]['rowid']
                            with open(checkpoint_path, "w") as f:
                                f.write(str(last_rowid))
                        
                        # Periodic profiling
                        if self.enable_profiling and processed_count - last_profile_count >= self.profile_interval:
                            print(f"\n  📊 Profiling checkpoint at {processed_count:,} vectors")
                            self._save_profile_stats(processed_count)
                            last_profile_count = processed_count
                    
                    # Finalize writer
                    progress.complete()
            
            # Clean up checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Show stats
            self._show_statistics(processed_count)
            
            return True
        
        except Exception as e:
            print(f"\n  ❗ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            if self.enable_profiling and self.profiler:
                self.profiler.disable()
                self._save_profile_stats(processed_count, final=True)
    
    def _is_vectors_file_complete(self, vectors_path: Path) -> bool:
        """Check if vectors file is complete."""
        if not vectors_path.exists():
            return False
        expected_size = UnifiedVectorWriter.HEADER_SIZE + self.total_vectors * UnifiedVectorWriter.RECORD_SIZE
        return vectors_path.stat().st_size == expected_size
    
    def _is_index_file_complete(self, index_path: Path) -> bool:
        """Check if index file is complete."""
        if not index_path.exists():
            return False
        expected_size = self.total_vectors * 26  # 22B track ID + 4B index
        return index_path.stat().st_size == expected_size
    
    def _save_profile_stats(self, processed_count: int, final: bool = False):
        """Save profiling statistics."""
        suffix = "final" if final else f"{processed_count//1000000}M"
        profile_path = self.output_dir / f"duckdb_preprocessing_profile_{suffix}.prof"
        self.profiler.dump_stats(str(profile_path))
        
        print(f"\n  📊 Profile saved: {profile_path}")
        
        # Quick text report
        stats = pstats.Stats(str(profile_path))
        print(f"\n  Top 5 bottlenecks ({suffix}):")
        stats.strip_dirs().sort_stats('cumulative').print_stats(5)
    
    def _show_statistics(self, total_processed: int):
        """Display final statistics."""
        vectors_path = self.output_dir / "track_vectors.bin"
        index_path = self.output_dir / "track_index.bin"
        
        vectors_size = vectors_path.stat().st_size if vectors_path.exists() else 0
        index_size = index_path.stat().st_size if index_path.exists() else 0
        
        vectors_gb = vectors_size / (1024**3)
        index_gb = index_size / (1024**3)
        
        print("\n" + "═" * 65)
        print("  ✅ DuckDB Streaming Preprocessing Complete!")
        print("═" * 65)
        print(f"    Total tracks processed: {total_processed:,}")
        print(f"    Vector file: {vectors_gb:.1f} GB")
        print(f"    Index file: {index_gb:.1f} GB")
        print(f"    Output directory: {self.output_dir}")
        print("\n  🚀 Ready for similarity search!")


if __name__ == "__main__":
    engine = PreprocessingEngine(
        enable_profiling=True
    )
    success = engine.run()
    sys.exit(0 if success else 1)
