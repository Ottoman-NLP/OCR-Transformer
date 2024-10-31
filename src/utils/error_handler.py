import logging
from collections import defaultdict
from typing import Dict, Set
import time
import os

class ErrorHandler:
    def __init__(self, log_file: str = 'training_errors.log'):
        self.error_counts = defaultdict(int)
        self.seen_errors = set()
        self.last_error_time = 0
        self.error_cooldown = 30  # Increased cooldown to 30 seconds
        self.log_file = log_file
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        # Setup file handler for errors
        self.logger = logging.getLogger('error_handler')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.WARNING)
    
    def handle_error(self, error: Exception, batch_idx: int, **context) -> bool:
        """Handle training errors silently unless critical."""
        error_key = f"{type(error).__name__}:{str(error)}"
        self.error_counts[error_key] += 1
        current_time = time.time()
        
        # Define critical errors
        critical_errors = {
            "CUDA out of memory",
            "device-side assert triggered",
            "no CUDA GPUs are available"
        }
        
        is_critical = any(ce in str(error) for ce in critical_errors)
        
        # Log to file
        self.logger.warning(
            f"Batch {batch_idx}: {error} (Count: {self.error_counts[error_key]})"
        )
        
        # Only print to console if critical
        if is_critical and (error_key not in self.seen_errors or 
                          current_time - self.last_error_time > self.error_cooldown):
            print(f"\nCritical error in batch {batch_idx}: {error}")
            self.seen_errors.add(error_key)
            self.last_error_time = current_time
            return False
        
        return True
    
    def save_summary(self):
        """Save error summary to log file."""
        with open(self.log_file, 'a') as f:
            f.write("\n\nError Summary\n")
            f.write("=" * 50 + "\n")
            for error, count in self.error_counts.items():
                f.write(f"\nError: {error}\nCount: {count}\n")
                f.write("-" * 50 + "\n")