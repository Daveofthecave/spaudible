# core/preprocessing/progress.py
import time
import sys
from collections import deque

class ProgressTracker:
    """Stable progress tracker with adaptive smoothing"""
    
    def __init__(self, total_items, bar_width=50, initial_processed=0):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            bar_width: Width of progress bar in characters
            initial_processed: Initial count of processed items
        """
        self.total = total_items
        self.processed = initial_processed
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_processed = initial_processed
        self.bar_width = bar_width
        self.display_started = False
        self.initial_processed = initial_processed
        
        # Time-windowed speed calculation
        self.time_window = 30.0  # seconds
        self.speed_data = deque()
        self.min_data_time = 10.0  # minimum seconds of data to show speed
        self.last_valid_speed = None
        self.smoothed_speed = None
        self.last_speed_time = time.time()
    
    def update(self, count=1):
        """Update progress with accurate timing"""
        self.processed += count
        
        current_time = time.time()
        
        # Always record data point for speed calculation
        self.speed_data.append((current_time, self.processed))
        
        # Remove data points outside our time window
        while self.speed_data and current_time - self.speed_data[0][0] > self.time_window:
            self.speed_data.popleft()
        
        elapsed_since_update = current_time - self.last_update_time
        
        # Update display at least every 0.5 seconds or 50k items
        if elapsed_since_update >= 0.5 or (self.processed - self.last_processed) >= 50000:
            if not self.display_started:
                # Start the display for the first time
                print()  # Add extra newline before progress bar
                self.display_started = True
            
            # Calculate current progress
            percent = self.processed / self.total
            filled = int(self.bar_width * percent)
            bar = '█' * filled + '░' * (self.bar_width - filled)
            
            # Initialize speed and ETA display
            speed_str = "--"
            eta_str = "--"
            current_speed = None
            
            # Calculate current speed if we have sufficient data
            if len(self.speed_data) > 1:
                oldest_time, oldest_count = self.speed_data[0]
                newest_time, newest_count = self.speed_data[-1]
                time_delta = newest_time - oldest_time
                
                if time_delta >= self.min_data_time:
                    vectors_delta = newest_count - oldest_count
                    current_speed = vectors_delta / time_delta
                    self.last_valid_speed = current_speed
            
            # Apply exponential smoothing
            if current_speed is not None:
                # Calculate time since last speed update
                time_since_last = current_time - self.last_speed_time
                
                # Adaptive smoothing factor based on time elapsed
                # More smoothing for rapid updates, less for slower updates
                smoothing_factor = min(0.9, 0.7 * (1 + time_since_last))
                
                if self.smoothed_speed is None:
                    self.smoothed_speed = current_speed
                else:
                    # Apply exponential smoothing
                    self.smoothed_speed = (smoothing_factor * self.smoothed_speed + 
                                          (1 - smoothing_factor) * current_speed)
                
                self.last_speed_time = current_time
                display_speed = self.smoothed_speed
            elif self.last_valid_speed is not None:
                display_speed = self.last_valid_speed
            else:
                display_speed = None
            
            # Format speed if available
            if display_speed is not None:
                if display_speed > 1000000:
                    speed_str = f"{display_speed/1000000:.2f}M vec/s"
                elif display_speed > 1000:
                    speed_str = f"{display_speed/1000:.1f}K vec/s"
                elif display_speed > 0:
                    speed_str = f"{int(display_speed)} vec/s"
                
                # Calculate ETA based on current speed
                if display_speed > 0 and percent > 0.01:
                    remaining_vectors = self.total - self.processed
                    eta_seconds = remaining_vectors / display_speed
                    eta_str = self._format_hours_minutes(eta_seconds)
            
            # Format processed count
            processed_m = self._format_millions(self.processed)
            total_m = int(self.total / 1_000_000)
            
            # Print progress display
            print(f"  [{bar}] {percent:.1%}")  # Progress bar line
            print(f"  Progress: {processed_m}M/{total_m}M tracks | Speed: {speed_str} | ETA: {eta_str}")
            
            # Move cursor up 2 lines for next update
            sys.stdout.write("\033[2A")
            
            # Reset counters
            self.last_update_time = current_time
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
            return f"{hours}h {minutes:02d}m    "
        else:
            return f"{minutes}m    "
    
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
            speed_str = f"{avg_speed/1000000:.2f}M vec/s"
        elif avg_speed > 1000:
            speed_str = f"{avg_speed/1000:.1f}K vec/s"
        else:
            speed_str = f"{int(avg_speed)} vec/s"
        
        # Print completion message
        print(f"\n  ✅ Processing completed in {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"  Average speed: {speed_str}")
        print(f"  Total tracks processed: {self.total:,}")
