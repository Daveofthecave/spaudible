# core/preprocessing/progress.py
import time
import sys
import math
from collections import deque

class ProgressTracker:
    """Ultra-stable progress tracker with delayed ETA display."""
    
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
        self.last_update = 0
        self.speed_history = deque(maxlen=100)  # Keep last 100 speeds
        self.eta_history = deque(maxlen=10)     # Keep last 10 ETAs
        self.last_processed = 0
        self.last_time = self.start_time
        self.last_eta = float('inf')
        self.stable_speed = None
        self.min_speed = float('inf')
        self.min_samples = 100  # Minimum samples before showing ETA
    
    def update(self, count=1):
        """Update progress and display if needed."""
        self.processed += count
        current_time = time.time()
        
        # Update speed history
        if current_time > self.last_time:
            elapsed = current_time - self.last_time
            speed = count / elapsed
            self.speed_history.append(speed)
            
            # Track minimum speed to avoid overestimating ETA
            if speed < self.min_speed:
                self.min_speed = speed
        
        self.last_time = current_time
        self.last_processed = self.processed
        
        # Only update display every 0.5 seconds
        if current_time - self.last_update >= 0.5:
            self._display()
            self.last_update = current_time
    
    def _display(self):
        """Display progress bar with ultra-stable ETA."""
        percent = self.processed / self.total
        filled = int(self.bar_width * percent)
        bar = '█' * filled + '░' * (self.bar_width - filled)
        
        # Format numbers with commas
        processed_str = f"{self.processed:,}"
        total_str = f"{self.total:,}"
        
        # Always show current speed
        if self.speed_history:
            # Use median speed for display
            sorted_speeds = sorted(self.speed_history)
            median_speed = sorted_speeds[len(sorted_speeds) // 2]
            
            # Format speed
            if median_speed > 1000:
                speed_str = f"{median_speed/1000:.1f}K vec/s"
            else:
                speed_str = f"{int(median_speed)} vec/s"
        else:
            speed_str = "Calculating..."
        
        # Delay ETA display until sufficient samples
        if len(self.speed_history) < self.min_samples:
            eta_str = "--:--"
        else:
            # Use 90th percentile speed for stability
            sorted_speeds = sorted(self.speed_history)
            idx = min(int(len(sorted_speeds) * 0.9), len(sorted_speeds) - 1)
            stable_speed = sorted_speeds[idx]
            
            # Track stable speed
            if self.stable_speed is None:
                self.stable_speed = stable_speed
            else:
                # Smooth transition: 95% old speed, 5% new measurement
                self.stable_speed = 0.95 * self.stable_speed + 0.05 * stable_speed
            
            # Calculate ETA
            remaining = self.total - self.processed
            eta_seconds = remaining / self.stable_speed
            
            # Apply heavy smoothing to ETA
            self.eta_history.append(eta_seconds)
            if len(self.eta_history) > 1:
                # Use median of last 10 ETAs
                sorted_etas = sorted(self.eta_history)
                median_eta = sorted_etas[len(sorted_etas) // 2]
                
                # Blend with previous ETA (90/10 ratio)
                if self.last_eta < float('inf'):
                    smoothed_eta = 0.9 * self.last_eta + 0.1 * median_eta
                else:
                    smoothed_eta = median_eta
            else:
                smoothed_eta = eta_seconds
                
            self.last_eta = smoothed_eta
            eta_str = self._format_time(smoothed_eta)
        
        # Clear screen and reprint both lines
        print("\033[2K\033[1G", end='')  # Clear line, move to beginning
        print(f"  [{bar}] {percent:.1%}")
        print(f"  Tracks: {processed_str}/{total_str} | Speed: {speed_str} | ETA: {eta_str}")
        print("\033[2A", end='')  # Move cursor up 2 lines for next update
    
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into clean HH:MM format."""
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m    "
        elif seconds >= 60:
            minutes = max(1, int(seconds // 60))  # Minimum 1 minute
            return f"{minutes}m    "
        else:
            return "<1m    "
    
    def complete(self):
        """Display completion message."""
        self._display()
        print()  # New line after progress bar
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # Calculate average speed
        avg_speed = self.total / total_time
        if avg_speed > 1000:
            speed_str = f"{avg_speed/1000:.1f}K vec/s"
        else:
            speed_str = f"{int(avg_speed)} vec/s"
        
        if hours > 0:
            print(f"\n\n  ✅ Processing completed in {hours}h {minutes}m | Avg speed: {speed_str}")
        else:
            print(f"\n\n  ✅ Processing completed in {minutes}m {seconds}s | Avg speed: {speed_str}")
