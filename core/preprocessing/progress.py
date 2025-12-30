# core/preprocessing/progress.py
import time
import sys

class ProgressTracker:
    """Handles progress display with ETA during preprocessing."""
    
    def __init__(self, total_items, bar_width=50):
        self.total = total_items
        self.processed = 0
        self.start_time = time.time()
        self.bar_width = bar_width
        self.last_update = 0
    
    def update(self, count=1):
        """Update progress and display if needed."""
        self.processed += count
        current_time = time.time()
        
        # Only update display every 0.5 seconds
        if current_time - self.last_update >= 0.5:
            self._display()
            self.last_update = current_time
    
    def _display(self):
        """Display progress bar with ETA."""
        percent = self.processed / self.total
        filled = int(self.bar_width * percent)
        bar = '█' * filled + '░' * (self.bar_width - filled)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.processed > 0:
            items_per_second = self.processed / elapsed
            eta_seconds = (self.total - self.processed) / items_per_second
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Format numbers with commas
        processed_str = f"{self.processed:,}"
        total_str = f"{self.total:,}"
        
        # Clear screen and reprint both lines
        print("\033[2K\033[1G", end='')  # Clear line, move to beginning
        print(f"  [{bar}] {percent:.1%}")
        print(f"  Tracks processed: {processed_str} / {total_str} - ETA: {eta_str}")
        print("\033[2A", end='')  # Move cursor up 2 lines for next update
    
        sys.stdout.flush()
    
    def _format_time(self, seconds):
        """Format seconds into HH:MM:SS or MM:SS."""
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            return f"{int(seconds)}s"
    
    def complete(self):
        """Display completion message."""
        self._display()
        print()  # New line after progress bar
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        if hours > 0:
            print(f"\n\n  ✅ Processing completed in {hours}h {minutes}m {seconds}s")
        else:
            print(f"\n\n  ✅ Processing completed in {minutes}m {seconds}s")
