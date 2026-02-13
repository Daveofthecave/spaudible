# core/utilities/update_manager.py
"""Spaudible Updater- checks the main branch on GitHub.
Works whether or not the user has git installed."""
import os
import sys
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from config import VERSION, PathConfig, FRAME_WIDTH

class UpdateManager:
    """Handles Spaudible updates from GitHub main branch."""
    
    REPO_OWNER = "Daveofthecave"
    REPO_NAME = "spaudible"
    
    # Directories/files to never touch during the update
    PROTECTED_PATHS = {
        'data', '.venv', 'venv', '__pycache__', '.git', 
        'playlists', 'backups', 'update.log',
        'config.json', '.env', '*.pyc', '*.pyo'
    }
    
    def __init__(self):
        self.base_dir = PathConfig.BASE_DIR
        self.is_git_repo = self._check_git_available() and (self.base_dir / ".git").exists()
        self.temp_dir = None
        self.target_branch = "dev"  # Change to "main" for production
        
    def _check_git_available(self) -> bool:
        """Check if git command is available. """
        try:
            result = subprocess.run(
                ["git", "--version"], 
                capture_output=True, 
                check=False,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_local_version_info(self) -> Dict:
        """Get current version and commit info."""
        info = {
            'version': VERSION,
            'commit': None,
            'date': None,
            'is_git': self.is_git_repo
        }
        
        if self.is_git_repo:
            try:
                # Get current commit hash
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=self.base_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5
                )
                info['commit'] = result.stdout.strip()
                
                # Get commit date
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%cd", "--date=iso"],
                    cwd=self.base_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5
                )
                info['date'] = result.stdout.strip()
            except Exception:
                pass
                
        return info
    
    def get_remote_version_info(self) -> Optional[Dict]:
        """Fetch latest commit info from GitHub API."""
        try:
            # Try GitHub API first (works for both git and ZIP users)
            api_url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/commits/{self.target_branch}"
            
            req = urllib.request.Request(
                api_url,
                headers={
                    'Accept': 'application/vnd.github.v3+json',
                    'User-Agent': f'Spaudible-{VERSION}'
                }
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                return {
                    'commit': data['sha'][:7],
                    'full_commit': data['sha'],
                    'date': data['commit']['committer']['date'],
                    'message': data['commit']['message'].split('\n')[0],
                    'url': data['html_url']
                }
        except Exception as e:
            print(f"Debug: API error {e}")
            return None
    
    def check_for_update(self) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """
        Check if update is available.
        Returns: (update_available, local_info, remote_info)
        """
        local = self.get_local_version_info()
        remote = self.get_remote_version_info()
        
        if not remote:
            return False, local, None
            
        # If we have git, compare commit hashes
        if local['commit'] and remote['commit']:
            update_available = local['commit'] != remote['commit']
        else:
            # Fallback: compare version strings (if version changed)
            # Or assume update available if we can't determine
            update_available = True
            
        return update_available, local, remote
    
    def update_via_git(self) -> bool:
        """Update using git pull (cleanest method)."""
        if not self.is_git_repo:
            raise UpdateError("Not a git repository")
            
        try:
            # Fetch latest
            result = subprocess.run(
                ["git", "fetch", "origin", self.target_branch],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            # Check if there are local uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            has_changes = bool(result.stdout.strip())
            
            if has_changes:
                # Stash local changes
                subprocess.run(
                    ["git", "stash", "push", "-m", "Auto-stash before update"],
                    cwd=self.base_dir,
                    check=True,
                    timeout=10
                )
            
            # Hard reset to origin/main (cleanest update)
            result = subprocess.run(
                ["git", "reset", "--hard", f"origin/{self.target_branch}"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            raise UpdateError(f"Git update failed: {e.stderr}")
    
    def update_via_zip(self, progress_callback=None) -> bool:
        """
        Update by downloading ZIP of main branch.
        Used when git is not available.
        """
        zip_url = f"https://github.com/{self.REPO_OWNER}/{self.REPO_NAME}/archive/refs/heads/{self.target_branch}.zip"
        
        self.temp_dir = tempfile.mkdtemp(prefix='spaudible_update_')
        zip_path = Path(self.temp_dir) / "spaudible_update.zip"
        extract_dir = Path(self.temp_dir) / "extracted"
        
        try:
            # Download
            if progress_callback:
                progress_callback("Downloading update...", 0, 100)
                
            req = urllib.request.Request(zip_url, headers={'User-Agent': f'Spaudible-{VERSION}'})
            
            with urllib.request.urlopen(req, timeout=120) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size:
                            pct = int((downloaded / total_size) * 50)  # First 50% is download
                            progress_callback("Downloading...", pct, 100)
            
            # Extract
            if progress_callback:
                progress_callback("Extracting...", 50, 100)
                
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Find source directory (usually spaudible-main/)
            subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if len(subdirs) != 1:
                raise UpdateError("Unexpected archive structure")
            
            source_dir = subdirs[0]
            
            # Backup current installation
            backup_dir = self._create_backup()
            
            if progress_callback:
                progress_callback("Installing update...", 60, 100)
            
            # Copy files, respecting protected paths
            updated = self._copy_update_files(source_dir, progress_callback)
            
            if progress_callback:
                progress_callback("Cleaning up...", 90, 100)
            
            # Write update log
            self._log_update(f"ZIP update successful, {updated} files updated")
            
            return True
            
        finally:
            self._cleanup_temp()
    
    def _create_backup(self) -> Path:
        """Create timestamped backup of current code."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.base_dir / "backups" / f"backup_{VERSION}_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy Python files and config
        for item in self.base_dir.iterdir():
            if item.name in self.PROTECTED_PATHS:
                continue
                
            try:
                if item.is_file() and item.suffix in ['.py', '.toml', '.txt', '.md', '.bat', '.command']:
                    shutil.copy2(item, backup_dir)
                elif item.is_dir() and item.name not in ['core', 'ui', 'data', '.venv', '__pycache__']:
                    shutil.copytree(item, backup_dir / item.name, dirs_exist_ok=True)
            except Exception:
                pass  # Best effort backup
        
        # Copy core and ui directories
        for code_dir in ['core', 'ui']:
            src = self.base_dir / code_dir
            if src.exists():
                dst = backup_dir / code_dir
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        return backup_dir
    
    def _copy_update_files(self, source_dir: Path, progress_callback=None) -> int:
        """Copy files from source to installation, preserving protected paths."""
        updated = 0
        total = sum(1 for _ in source_dir.rglob('*') if _.is_file())
        current = 0
        
        for src_file in source_dir.rglob('*'):
            if not src_file.is_file():
                continue
            
            current += 1
            
            # Calculate relative path
            try:
                rel_path = src_file.relative_to(source_dir)
            except ValueError:
                continue
            
            # Skip protected paths
            if any(part in self.PROTECTED_PATHS for part in rel_path.parts):
                continue
            
            dest_path = self.base_dir / rel_path
            
            # Create parent directories
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            try:
                shutil.copy2(src_file, dest_path)
                updated += 1
                
                if progress_callback and current % 10 == 0:
                    pct = 60 + int((current / total) * 30)  # 60-90% range
                    progress_callback(f"Installing ({current}/{total})...", pct, 100)
                    
            except Exception as e:
                print(f"Warning: Could not update {rel_path}: {e}")
        
        return updated
    
    def _log_update(self, message: str):
        """Log update activity."""
        log_path = self.base_dir / "update.log"
        timestamp = datetime.now().isoformat()
        try:
            with open(log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass
    
    def _cleanup_temp(self):
        """Remove temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
    
    def verify_installation(self) -> Tuple[bool, str]:
        """Verify that update didn't break anything critical."""
        critical_files = [
            PathConfig.BASE_DIR / "main.py",
            PathConfig.BASE_DIR / "config.py",
            PathConfig.BASE_DIR / "data" / "genre_intensity_mapping.csv",
            PathConfig.BASE_DIR / "core" / "similarity_engine" / "__init__.py",
            PathConfig.BASE_DIR / "ui" / "cli" / "menu_system" / "main_menu_handlers" / "core_search.py"
        ]
        
        for file_path in critical_files:
            if not file_path.exists():
                return False, f"Critical file missing after update: {file_path.name}"
        
        return True, "OK"

class UpdateError(Exception):
    """Custom exception for update failures."""
    pass
