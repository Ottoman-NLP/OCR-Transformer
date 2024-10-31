import torch
import gc

def optimize_memory():
    """Optimize GPU memory usage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def log_memory_usage():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        return f"GPU Memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached"
    return "GPU not available" 