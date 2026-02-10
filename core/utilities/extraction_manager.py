# core/utilities/extraction_manager.py
"""Extraction manager for decompressing Zstandard (.zst) files."""
import shutil
import zstandard as zstd
from pathlib import Path
from typing import Optional, Callable
from config import PathConfig, DownloadConfig

class ExtractionError(Exception):
    """Raised when extraction fails irrecoverably."""
    pass

class ZstExtractor:
    """Stream-decompresses Zstandard files with progress tracking."""
    
    def __init__(self, progress_callback: Optional[Callable[[str, int, int], None]] = None):
        """Initialize extractor with optional progress callback.
        
        Args:
            progress_callback: Called with (filename, bytes_processed, total_bytes)
        """
        self.progress_callback = progress_callback
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks for efficient streaming

    def extract_database(self, zst_filename: str) -> Path:
        """Extract a single database .zst file with validation.
        
        Args:
            zst_filename: Name of the .zst file (e.g., "spotify_clean.sqlite3.zst")
            
        Returns:
            Path to the extracted file
            
        Raises:
            ExtractionError: If extraction fails or file is corrupted
        """
        input_path = PathConfig.DATABASES / zst_filename
        output_filename = zst_filename.replace('.zst', '')
        output_path = PathConfig.DATABASES / output_filename
        
        if not input_path.exists():
            raise ExtractionError(f"Archive not found: {input_path}")
        
        # Get expected size from config for validation
        expected_gb = 0
        for fname, _, extracted_gb in DownloadConfig.DATABASE_FILES:
            if fname == zst_filename:
                expected_gb = extracted_gb
                break
        
        # Check if already extracted and valid
        if output_path.exists():
            actual_gb = output_path.stat().st_size / (1e9)
            min_expected = expected_gb * 0.95  # 5% tolerance
            if actual_gb >= min_expected:
                return output_path
            else:
                print(f"⚠️ Existing file too small ({actual_gb:.1f}GB vs {expected_gb}GB expected), re-extracting...")
                try:
                    output_path.unlink()
                except OSError as e:
                    raise ExtractionError(f"Cannot remove corrupt existing file: {e}")
        
        # Perform extraction
        self._extract_file(input_path, output_path, expected_gb)
        
        # Verify extraction succeeded
        if not output_path.exists():
            raise ExtractionError("Extraction completed but output file missing")
        
        actual_size = output_path.stat().st_size
        expected_bytes = expected_gb * 1e9
        
        if actual_size < expected_bytes * 0.95:
            # Cleanup failed extraction
            try:
                output_path.unlink()
            except:
                pass
            raise ExtractionError(
                f"Extracted file too small: {actual_size / (1e9):.1f} GB "
                f"(expected {expected_gb} GB). File may be corrupted."
            )
        
        return output_path

    def extract_all_databases(self) -> bool:
        """Extract all database .zst files if present.
        
        Returns:
            True if all extractions completed successfully
        """
        for filename, _, _ in DownloadConfig.DATABASE_FILES:
            zst_path = PathConfig.DATABASES / filename
            extracted_path = PathConfig.DATABASES / filename.replace('.zst', '')
            
            if zst_path.exists() and not extracted_path.exists():
                print(f"  Extracting {filename}...")
                try:
                    self.extract_database(filename)
                    print(f"  Done: {filename.replace('.zst', '')}")
                except Exception as e:
                    raise ExtractionError(f"Failed to extract {filename}: {e}")
        
        return True

    def _extract_file(self, input_path: Path, output_path: Path, expected_gb: float):
        """Core extraction logic using streaming decompression.
        
        Writes to temp file first, then moves atomically on success.
        """
        temp_path = output_path.with_suffix('.tmp')
        total_bytes_expected = int(expected_gb * 1e9)
        bytes_written = 0
        
        try:
            with open(input_path, 'rb') as compressed, open(temp_path, 'wb') as output:
                # Create decompressor
                dctx = zstd.ZstdDecompressor()
                
                # Stream decompress
                with dctx.stream_reader(compressed) as reader:
                    while True:
                        chunk = reader.read(self.chunk_size)
                        if not chunk:
                            break
                        output.write(chunk)
                        bytes_written += len(chunk)
                        
                        if self.progress_callback:
                            self.progress_callback(input_path.name, bytes_written, total_bytes_expected)
            
            # Validate extraction succeeded (check non-empty)
            if temp_path.stat().st_size == 0:
                raise ExtractionError("Extracted file is empty")
            
            # Atomic move to final destination
            shutil.move(str(temp_path), str(output_path))
            
        except Exception as e:
            # Cleanup temp file on failure
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise ExtractionError(f"Extraction failed: {e}")

    def cleanup_archives(self) -> int:
        """Remove .zst archives after successful extraction to save disk space.
        
        Call this only after verifying extractions are valid.
        
        Returns:
            Number of archives removed
        """
        removed_count = 0
        for filename, _, _ in DownloadConfig.DATABASE_FILES:
            zst_path = PathConfig.DATABASES / filename
            extracted_path = PathConfig.DATABASES / filename.replace('.zst', '')
            
            if zst_path.exists() and extracted_path.exists():
                # Verify extracted file is reasonable size before deleting archive
                expected_gb = next(
                    (extracted for fname, _, extracted in DownloadConfig.DATABASE_FILES if fname == filename),
                    0
                )
                actual_gb = extracted_path.stat().st_size / (1e9)
                
                if actual_gb >= expected_gb * 0.95:  # Verify extraction was complete
                    zst_path.unlink()
                    removed_count += 1
        
        return removed_count

    def get_extraction_status(self) -> dict:
        """Check which databases need extraction.
        
        Returns:
            Dict with keys: 'extracted', 'to_extract', 'archives_present'
        """
        status = {
            'extracted': [],
            'to_extract': [],
            'archives_present': [],
            'missing': []
        }
        
        for filename, _, _ in DownloadConfig.DATABASE_FILES:
            zst_path = PathConfig.DATABASES / filename
            extracted_path = PathConfig.DATABASES / filename.replace('.zst', '')
            
            if extracted_path.exists():
                status['extracted'].append(filename.replace('.zst', ''))
            elif zst_path.exists():
                status['to_extract'].append(filename)
                status['archives_present'].append(filename)
            else:
                status['missing'].append(filename)
        
        return status
