# core/preprocessing/db_to_vectors.py
import sqlite3
import time
import numpy as np
from pathlib import Path
import mmap
import gc
from core.vectorization.track_vectorizer import build_track_vectors_batch
from core.preprocessing.unified_vector_writer import UnifiedVectorWriter
from core.preprocessing.progress import ProgressTracker
import os

class DatabaseReader:
    """Optimized database reader with memory mapping and efficient audio feature handling."""
    
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
        """Open database connections with optimizations."""
        # Open connections
        self.main_conn = sqlite3.connect(self.main_db_path, timeout=0)
        self.audio_conn = sqlite3.connect(self.audio_db_path, timeout=0)
        
        # Set performance settings
        self.main_conn.execute("PRAGMA journal_mode = MEMORY")
        self.main_conn.execute("PRAGMA cache_size = -200000")  # 200GB cache
        self.main_conn.execute("PRAGMA temp_store = MEMORY")
        self.main_conn.execute("PRAGMA synchronous = OFF")
        self.main_conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        self.main_conn.execute("PRAGMA mmap_size = 30000000000")  # 30GB memory mapping
        
        self.audio_conn.execute("PRAGMA journal_mode = MEMORY")
        self.audio_conn.execute("PRAGMA synchronous = OFF")
        self.audio_conn.execute("PRAGMA temp_store = MEMORY")
        
        # Memory map databases
        self._memory_map_databases()
        
        # Preload artist metadata only
        self._preload_artist_metadata()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
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
            with open(self.main_db_path, 'rb') as f:
                self.main_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            with open(self.audio_db_path, 'rb') as f:
                self.audio_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            print(f"  ⚠️  Could not memory-map databases: {e}")
    
    def _preload_artist_metadata(self):
        """Preload artist metadata into memory."""
        cursor = self.main_conn.cursor()
        cursor.execute("SELECT rowid, followers_total FROM artists")
        self.artist_followers_cache = {rowid: followers for rowid, followers in cursor}
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
        Optimized track streaming with batched audio feature loading.
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
            track_ids = []
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
                track_ids.append(track_data['track_id'])
                last_rowid = track_data['rowid']
            
            # Fetch audio features in bulk for this batch
            self._fetch_audio_features(enriched_batch, track_ids)
            
            yield enriched_batch
            gc.collect()
            
        cursor.close()

    def _fetch_audio_features(self, track_batch, track_ids):
        """Fetch audio features for a batch of tracks efficiently."""
        if not track_ids:
            return
        
        # Create lookup dictionary
        features_map = {}
        audio_cursor = self.audio_conn.cursor()
        
        # Process in chunks to avoid SQLite parameter limits
        chunk_size = 10000
        for i in range(0, len(track_ids), chunk_size):
            chunk_ids = track_ids[i:i+chunk_size]
            placeholders = ','.join(['?'] * len(chunk_ids))
            
            query = f"""
            SELECT track_id, danceability, energy, loudness, speechiness,
                acousticness, instrumentalness, liveness, valence,
                tempo, time_signature, key, mode
            FROM track_audio_features
            WHERE track_id IN ({placeholders})
            """
            audio_cursor.execute(query, chunk_ids)
            
            for row in audio_cursor:
                track_id = row[0]
                features_map[track_id] = {
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
        
        # Merge audio features into track data
        for track_data in track_batch:
            track_id = track_data['track_id']
            if track_id in features_map:
                track_data.update(features_map[track_id])
            else:
                # Set default values for missing audio features
                track_data.update({
                    'danceability': -1.0,
                    'energy': -1.0,
                    'loudness': -1.0,
                    'speechiness': -1.0,
                    'acousticness': -1.0,
                    'instrumentalness': -1.0,
                    'liveness': -1.0,
                    'valence': -1.0,
                    'tempo': -1.0,
                    'time_signature': -1.0,
                    'key': -1.0,
                    'mode': -1.0
                })

class PreprocessingEngine:
    """Optimized preprocessing engine with unified vector format."""
    
    # Region dictionary: 8 geographical/cultural/linguistic regions
    REGION_MAPPING = {
        0: ["AU", "CA", "CB", "GB", "GG", "GX", "IE", "IM", "JE", "NZ", 
            "QM", "QT", "QZ", "UK", "US"],  # Anglo
        1: ["AD", "AT", "BE", "CH", "DE", "DK", "EE", "FI", "FO", "FR", 
            "FX", "GI", "GL", "IS", "IT", "LI", "LU", "MC", "MT", "NL", 
            "NO", "PT", "SE", "SM"],  # Western European
        2: ["AL", "BA", "BG", "BY", "CS", "CY", "CZ", "GR", "HR", "HU", 
            "LT", "LV", "MD", "ME", "MK", "PL", "RO", "RS", "RU", "SI", 
            "SK", "UA", "XK", "YU"],  # Eastern European
        3: ["AR", "BC", "BK", "BO", "BP", "BR", "BX", "BZ", "CL", "CO", 
            "CR", "CU", "DO", "EC", "ES", "GT", "HN", "MX", "NI", "PA", 
            "PE", "PR", "PY", "SV", "UY", "VE"],  # Hispanic
        4: ["BN", "CN", "HK", "ID", "JP", "KG", "KH", "KR", "KS", "KZ", 
            "LA", "MM", "MN", "MO", "MY", "PG", "PH", "SG", "TH", "TL", 
            "TW", "UZ", "VN"],  # Asian
        5: ["BD", "BT", "IN", "LK", "MV", "NP", "PK"],  # Indian
        6: ["AE", "AF", "AM", "AZ", "BH", "DZ", "EG", "GE", "IL", "IQ", 
            "IR", "JO", "KW", "LB", "MA", "OM", "PS", "QA", "SA", "SY", 
            "TN", "TR", "YE"],  # Middle Eastern
        7: ["AG", "AI", "AO", "AW", "BB", "BF", "BI", "BJ", "BM", "BS", 
            "BW", "CD", "CF", "CG", "CI", "CM", "CP", "CV", "CW", "DG", 
            "DM", "ET", "FJ", "GA", "GD", "GH", "GM", "GN", "GQ", "GY", 
            "HT", "JM", "KE", "KM", "KN", "KY", "LC", "LR", "LS", "MF", 
            "MG", "ML", "MP", "MR", "MS", "MU", "MW", "MZ", "NA", "NE", 
            "NG", "PF", "QN", "RW", "SB", "SC", "SD", "SL", "SN", "SO", 
            "SR", "SS", "ST", "SX", "SZ", "TC", "TD", "TG", "TO", "TT", 
            "TZ", "UG", "VC", "VG", "VU", "VV", "ZA", "ZB", "ZM", "ZW", 
            "ZZ"]  # Other
    }
    
    # Reverse lookup for country codes
    COUNTRY_TO_REGION = {}
    for region_id, countries in REGION_MAPPING.items():
        for country in countries:
            COUNTRY_TO_REGION[country] = region_id

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
    
    def get_region_from_isrc(self, isrc: str) -> int:
        """
        Convert ISRC to region index (0-7).
        
        Args:
            isrc: ISRC code (first 2 characters are country code)
            
        Returns:
            Region index (0-7), defaults to 7 (Other)
        """
        if not isrc or not isrc.strip():
            return 7
        if len(isrc) < 2:
            return 7
        country_code = isrc[:2].upper()
        return self.COUNTRY_TO_REGION.get(country_code, 7)
    
    def run(self):
        """Run optimized preprocessing pipeline."""
        print("\n" + "═" * 65)
        print("  🚀 Starting High-Performance Preprocessing")
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
        
        print("\n  🔥 Processing tracks with optimized pipeline...")
        print("  This will take 30-90 minutes. Press Ctrl+C to interrupt.")
        
        # Initialize progress tracker
        progress = ProgressTracker(self.total_vectors)
        processed_count = 0
        
        # Initialize unified vector writer
        with UnifiedVectorWriter(Path(self.output_dir)) as writer:
            with DatabaseReader(self.main_db_path, self.audio_db_path) as db_reader:
                # Process in streaming batches
                for batch in db_reader.stream_tracks(self.batch_size):
                    # Process vector batches within the main batch
                    for i in range(0, len(batch), self.vector_batch_size):
                        vector_batch = batch[i:i+self.vector_batch_size]
                        
                        # Build vectors - THIS IS WHERE AUDIO FEATURES ARE HANDLED
                        vectors = build_track_vectors_batch(vector_batch)
                        
                        # Write vectors
                        for j, track_data in enumerate(vector_batch):
                            # Get ISRC and resolve region
                            isrc = track_data.get('external_id_isrc', '') or ''
                            region = self.get_region_from_isrc(isrc)
                            
                            writer.write_record(
                                track_data['track_id'], 
                                vectors[j],
                                isrc,
                                region
                            )
                        
                        # Update progress
                        progress.update(len(vector_batch))
                        processed_count += len(vector_batch)
                
                # Finalize processing
                writer.finalize()
                progress.complete()
        
        # Show statistics
        self._show_statistics(processed_count)
        return True
    
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
