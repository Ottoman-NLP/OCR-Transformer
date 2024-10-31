from .cpp.viz_engine import VizEngine
import numpy as np
from pathlib import Path
import time

class TrainingVisualizer:
    def __init__(self):
        try:
            self.engine = VizEngine()
        except ImportError as e:
            print(f"Warning: Failed to load C++ visualization engine: {e}")
            print("Falling back to Python implementation")
            from .qt_viz import TrainingVisualizer as PyViz
            self.engine = PyViz()
        
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.learning_rates = []
    
    def update(self, epoch, train_loss, val_loss, lr, accuracy):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.accuracies.append(float(accuracy))
        self.learning_rates.append(float(lr))
        self.engine.update(train_loss, val_loss, accuracy, lr)
    
    def save(self, prefix='training'):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_dir = Path('outputs/visualization')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            save_dir / f'{prefix}_data_{timestamp}.npz',
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            accuracies=self.accuracies,
            learning_rates=self.learning_rates
        )