from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

class TrainingProgressTracker:
    def __init__(self):
        self._console = Console()
        self._metrics = {
            'epoch': 0,
            'total_epochs': 0,
            'batch': 0,
            'total_batches': 0,
            'loss': 0.0,
            'best_loss': float('inf'),
            'accuracy': 0.0,
            'best_accuracy': 0.0,
            'learning_rate': 0.0,
            'cuda_memory': 0.0,
            'elapsed_time': 0,
            'samples_processed': 0,
            'samples_per_second': 0.0
        }
        
        self._start_time = time.time()
        
        # Create progress columns with detailed information
        self._progress_columns = [
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green"),
            TaskProgressColumn(),
            TextColumn("•"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn()
        ]
        
        self._progress = Progress(
            *self._progress_columns,
            console=self._console,
            refresh_per_second=1,
            expand=True
        )
        
        self._live = None
        self._tasks = {}
    
    def __enter__(self):
        """Context manager entry."""
        self._start_time = time.time()
        self._live = Live(
            self._get_layout(),
            console=self._console,
            refresh_per_second=1,
            transient=False
        )
        self._live.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._live:
            self._live.stop()
    
    def _get_layout(self):
        """Create a layout with metrics and progress."""
        layout = Layout()
        
        # Calculate elapsed time and other metrics
        elapsed = time.time() - self._start_time
        samples_per_sec = self._metrics['samples_processed'] / elapsed if elapsed > 0 else 0
        
        # Create metrics panel
        metrics_text = "\n".join([
            f"[cyan]Training Progress[/cyan]",
            f"├─ Epoch: {self._metrics['epoch']}/{self._metrics['total_epochs']}",
            f"├─ Batch: {self._metrics['batch']}/{self._metrics['total_batches']}",
            f"├─ Samples: {self._metrics['samples_processed']:,}",
            f"└─ Speed: {samples_per_sec:.2f} samples/sec\n",
            f"[yellow]Performance Metrics[/yellow]",
            f"├─ Current Loss: {self._metrics['loss']:.4f}",
            f"├─ Best Loss: {self._metrics['best_loss']:.4f}",
            f"├─ Accuracy: {self._metrics['accuracy']:.2f}%",
            f"└─ Best Accuracy: {self._metrics['best_accuracy']:.2f}%\n",
            f"[green]Resource Usage[/green]",
            f"├─ CUDA Memory: {self._metrics['cuda_memory']:.2f} MB",
            f"├─ Learning Rate: {self._metrics['learning_rate']:.6f}",
            f"└─ Runtime: {datetime.fromtimestamp(elapsed).strftime('%H:%M:%S')}"
        ])
        
        metrics_panel = Panel(
            metrics_text,
            title="Training Status",
            border_style="bold white",
            expand=False
        )
        
        # Add components to layout
        layout.split(
            Layout(metrics_panel),
            Layout(self._progress)
        )
        
        return layout
    
    def create_epoch_task(self, total_epochs: int) -> int:
        """Create epoch progress task."""
        self._metrics['total_epochs'] = total_epochs
        task_id = self._progress.add_task(
            "Epochs",
            total=total_epochs
        )
        self._tasks['epoch'] = task_id
        return task_id
    
    def create_batch_task(self, total_batches: int) -> int:
        """Create batch progress task."""
        self._metrics['total_batches'] = total_batches
        task_id = self._progress.add_task(
            "Batches",
            total=total_batches
        )
        self._tasks['batch'] = task_id
        return task_id
    
    def update_metrics(self, **kwargs):
        """Update training metrics."""
        self._metrics.update(kwargs)
        
        # Update best metrics
        if self._metrics['loss'] < self._metrics['best_loss']:
            self._metrics['best_loss'] = self._metrics['loss']
        if self._metrics['accuracy'] > self._metrics['best_accuracy']:
            self._metrics['best_accuracy'] = self._metrics['accuracy']
        
        # Update elapsed time
        self._metrics['elapsed_time'] = time.time() - self._start_time
        
        if self._live:
            self._live.update(self._get_layout())
    
    def update_task(self, task_id: int, advance: int = 1, **kwargs):
        """Update task progress and metrics."""
        self._progress.update(task_id, advance=advance)
        if kwargs:
            self.update_metrics(**kwargs)
    
    def reset_batch_progress(self):
        """Reset batch progress without flickering."""
        if 'batch' in self._tasks:
            self._progress.reset(self._tasks['batch'], visible=True)