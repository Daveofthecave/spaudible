# core/preprocessing/progress.py
import time
import sys
import math
from collections import deque

class ProgressTracker:
    """Progress tracker with reliable two-line display."""
    
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
        self.last_speed = 0
        self.last_eta = ""
        self.display_started = False
    
    def update(self, count=1):
        """Update progress and display if needed."""
        self.processed += count
        
        # Only update display periodically
        current_time = time.time()
        elapsed_since_update = current_time - self.last_update
        
        # Update at least every 0.5 seconds or 50k items
        if elapsed_since_update >= 0.5 or (self.processed - self.last_processed) >= 50000:
            if not self.display_started:
                # Start the display for the first time
                print()  # Add extra newline before progress bar
                self.display_started = True
            
            # Calculate current progress
            percent = self.processed / self.total
            filled = int(self.bar_width * percent)
            bar = '█' * filled + '░' * (self.bar_width - filled)
            
            # Calculate current speed
            current_speed = (self.processed - self.last_processed) / (time.time() - self.last_update)
            self.speed_history.append(current_speed)
            
            # Calculate rolling average speed
            if self.speed_history:
                avg_speed = sum(self.speed_history) / len(self.speed_history)
            else:
                avg_speed = current_speed
            
            # Format speed
            if avg_speed > 1000000:
                speed_str = f"{avg_speed/1000000:.1f}M vec/s"
            elif avg_speed > 1000:
                speed_str = f"{avg_speed/1000:.1f}K vec/s"
            else:
                speed_str = f"{int(avg_speed)} vec/s"
            
            # Calculate ETA
            if percent > 0.01 and avg_speed > 0:
                remaining = self.total - self.processed
                eta_seconds = remaining / avg_speed
                self.eta_history.append(eta_seconds)
                
                # Use median of last 10 ETAs
                sorted_etas = sorted(self.eta_history)
                median_eta = sorted_etas[len(sorted_etas) // 2]
                eta_str = self._format_hours_minutes(median_eta)
            else:
                eta_str = "--"
            
            # Format processed count
            processed_m = self._format_millions(self.processed)
            total_m = int(self.total / 1_000_000)
            
            # Print progress display
            print(f"  [{bar}] {percent:.1%}")  # Progress bar line
            print(f"  Progress: {processed_m}M/{total_m}M tracks | Speed: {speed_str} | ETA: {eta_str}")
            
            # Move cursor up 2 lines for next update
            sys.stdout.write("\033[2A")
            
            self.last_update = current_time
            self.last_processed = self.processed
    
    def _format_millions(self, number):
        """Format number in millions with one decimal place."""
        return f"{number/1000000:.1f}"
    
    def _format_hours_minutes(self, seconds: float) -> str:
        """Format seconds into HHh MMm."""
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}h {minutes:02d}m"
        else:
            return f"{minutes}m"
    
    def complete(self):
        """Display completion message."""
        if self.display_started:
            # Move down past the progress display
            sys.stdout.write("\033[2B")
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # Format average speed
        avg_speed = self.total / total_time
        if avg_speed > 1000000:
            speed_str = f"{avg_speed/1000000:.1f}M vec/s"
        elif avg_speed > 1000:
            speed_str = f"{avg_speed/1000:.1f}K vec/s"
        else:
            speed_str = f"{int(avg_speed)} vec/s"
        
        # Print completion message
        print(f"\n  ✅ Processing completed in {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"  Average speed: {speed_str}")
        print(f"  Total tracks processed: {self.total:,}")
