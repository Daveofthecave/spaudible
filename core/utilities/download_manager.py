# core/utilities/download_manager.py
"""Download manager for Spaudible data files from HuggingFace Hub."""
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from config import PathConfig, DownloadConfig

class DownloadError(Exception):
    """Raised when a download fails irrecoverably."""
    pass

class SpaudibleDownloader:
    """Manages downloads from HuggingFace Hub with resume capability."""
    
    def __init__(self, progress_callback: Optional[Callable[[str, int, int], None]] = None):
        self.progress_callback = progress_callback
        self.state_file = DownloadConfig.get_download_state_file()
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load download state from disk or return fresh state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"completed": {}, "in_progress": {}, "failed": {}}
    
    def _save_state(self):
        """Persist current state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def check_existing_files(self) -> Dict[str, bool]:
        status = {}
        expected_sizes = {
            "spotify_clean.sqlite3.zst": int(36.7 * 1024**3),
            "spotify_clean_audio_features.sqlite3.zst": int(17.7 * 1024**3)
        }
        for filename, _, _ in DownloadConfig.DATABASE_FILES:
            path = PathConfig.DATABASES / filename
            if path.exists():
                actual_size = path.stat().st_size
                expected_size = expected_sizes.get(filename, 1_000_000)
                # Allow 5% tolerance for partial downloads/resumes
                status[filename] = actual_size > expected_size * 0.95
            else:
                status[filename] = False
        return status
    
    def download_databases(self) -> bool:
        """Download both database .zst files."""
        print(f"Downloading database files from {DownloadConfig.REPO_DB}...")
        
        for filename, size_gb, _ in DownloadConfig.DATABASE_FILES:
            if self.state["completed"].get(f"db_{filename}"):
                print(f"  {filename} already downloaded")
                continue
                
            print(f"  Downloading {filename} ({size_gb} GB)...")
            try:
                self._download_file(
                    repo_id=DownloadConfig.REPO_DB,
                    filename=filename,
                    local_dir=PathConfig.DATABASES,
                    state_key=f"db_{filename}"
                )
                print(f"  {filename} complete")
            except Exception as e:
                raise DownloadError(f"Failed to download {filename}: {e}")
        
        return True
    
    def download_vector_cache(self) -> bool:
        """Download pre-built vector cache files."""
        print(f"Downloading vector cache from {DownloadConfig.REPO_VECTORS}...")
        
        for filename, subdir, size_gb in DownloadConfig.VECTOR_FILES:
            state_key = f"vec_{filename}"
            if self.state["completed"].get(state_key):
                print(f"  {filename} already downloaded")
                continue
                
            print(f"  Downloading {filename} ({size_gb} GB)...")
            local_dir = PathConfig.VECTORS / subdir if subdir else PathConfig.VECTORS
            
            try:
                self._download_file(
                    repo_id=DownloadConfig.REPO_VECTORS,
                    filename=filename,
                    local_dir=local_dir,
                    state_key=state_key
                )
                print(f"  {filename} complete")
            except Exception as e:
                raise DownloadError(f"Failed to download {filename}: {e}")
        
        return True
    
    def _download_file(self, repo_id: str, filename: str, local_dir: Path, state_key: str):
        """Download single file with resume support via huggingface_hub."""
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists and valid
        target_path = local_dir / filename
        if target_path.exists():
            # Verify size matches expected (from config)
            expected_bytes = self._get_expected_size(state_key)
            if target_path.stat().st_size >= expected_bytes * 0.99:  # 1% tolerance
                self.state["completed"][state_key] = str(target_path)
                self._save_state()
                return
        
        self.state["in_progress"][state_key] = {
            "repo_id": repo_id,
            "filename": filename,
            "local_dir": str(local_dir)
        }
        self._save_state()
        
        try:
            # hf_hub_download handles resume automatically via ETags
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=False
                # Note: tqdm progress bar will show automatically
            )
            self.state["completed"][state_key] = str(downloaded_path)
            self.state["in_progress"].pop(state_key, None)
            self._save_state()
        except Exception as e:
            self.state["failed"][state_key] = str(e)
            self._save_state()
            raise DownloadError(f"Failed to download {filename}: {e}")

    def _get_expected_size(self, state_key: str) -> int:
        """Get expected file size in bytes from config."""
        if state_key.startswith("db_"):
            filename = state_key.replace("db_", "")
            for fname, size_gb, _ in DownloadConfig.DATABASE_FILES:
                if fname == filename:
                    return int(size_gb * 1024**3)
        elif state_key.startswith("vec_"):
            filename = state_key.replace("vec_", "")
            for fname, _, size_gb in DownloadConfig.VECTOR_FILES:
                if fname == filename:
                    return int(size_gb * 1024**3)
        return 0

    def is_database_download_complete(self) -> bool:
        """Check if all database files are marked complete."""
        return all(
            self.state["completed"].get(f"db_{filename}")
            for filename, _, _ in DownloadConfig.DATABASE_FILES
        )
    
    def is_vector_download_complete(self) -> bool:
        """Check if all vector files are marked complete."""
        return all(
            self.state["completed"].get(f"vec_{filename}")
            for filename, _, _ in DownloadConfig.VECTOR_FILES
        )
    
    def clear_state(self):
        """Clear download state (useful for retrying failed downloads)."""
        self.state = {"completed": {}, "in_progress": {}, "failed": {}}
        self._save_state()
