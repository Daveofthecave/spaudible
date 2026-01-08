# core/preprocessing/progress.py
# core/preprocessing/progress.py
import time
import sys
import math
from collections import deque

class ProgressTracker:
    """Lightweight progress tracker with accurate ETA calculation."""
    
    def __init__(self, total_items, bar_width=50):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            bar_width: Width of progress bar in characters
        """
        self.total = total_items
        self.processed = 0
        self.start_time = time.time()
        self.bar_width = bar_width
        self.last_update = self.start_time
        self.last_processed = 0
        self.speed_history = deque(maxlen=20)  # Last 20 speed measurements
        self.eta_history = deque(maxlen=10)    # Last 10 ETA calculations
    
    def update(self, count=1):
        """Update progress and display if needed."""
        self.processed += count
        
        # Only update display periodically
        current_time = time.time()
        elapsed_since_update = current_time - self.last_update
        
        # Update at least every 0.5 seconds or 50k items
        if elapsed_since_update >= 0.5 or (self.processed - self.last_processed) >= 50000:
            self._display()
            self.last_update = current_time
            self.last_processed = self.processed
    
    def _display(self):
        """Display progress bar with accurate ETA."""
        # Calculate current progress
        percent = self.processed / self.total
        filled = int(self.bar_width * percent)
        bar = '█' * filled + '░' * (self.bar_width - filled)
        
        # Calculate current speed
        elapsed = time.time() - self.start_time
        current_speed = (self.processed - self.last_processed) / (time.time() - self.last_update)
        self.speed_history.append(current_speed)
        
        # Calculate rolling average speed (last 10 measurements)
        if self.speed_history:
            avg_speed = sum(self.speed_history) / len(self.speed_history)
        else:
            avg_speed = current_speed
        
        # Format speed
        if avg_speed > 1000000:
            speed_str = f"{avg_speed/1000000:.2f}M vec/s"
        elif avg_speed > 1000:
            speed_str = f"{avg_speed/1000:.1f}K vec/s"
        else:
            speed_str = f"{int(avg_speed)} vec/s"
        
        # Calculate ETA (only after 1% progress)
        if percent > 0.01 and avg_speed > 0:
            remaining = self.total - self.processed
            eta_seconds = remaining / avg_speed
            self.eta_history.append(eta_seconds)
            
            # Use median of last 10 ETAs for stability
            sorted_etas = sorted(self.eta_history)
            median_eta = sorted_etas[len(sorted_etas) // 2]
            eta_str = self._format_time(median_eta)
        else:
            eta_str = "--:--:--"
        
        # Format processed count
        processed_str = f"{self.processed:,}"
        total_str = f"{self.total:,}"
        
        # Update display
        sys.stdout.write(
            f"\r  [{bar}] {percent:.1%} | "
            f"Processed: {processed_str}/{total_str} | "
            f"Speed: {speed_str} | ETA: {eta_str}"
        )
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def complete(self):
        """Display completion message."""
        self._display()
        print()  # Move to new line
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # Calculate average speed
        avg_speed = self.total / total_time
        if avg_speed > 1000000:
            speed_str = f"{avg_speed/1000000:.2f}M vec/s"
        elif avg_speed > 1000:
            speed_str = f"{avg_speed/1000:.1f}K vec/s"
        else:
            speed_str = f"{int(avg_speed)} vec/s"
        
        print(f"\n  ✅ Processing completed in {self._format_time(total_time)}")
        print(f"  Average speed: {speed_str}")
