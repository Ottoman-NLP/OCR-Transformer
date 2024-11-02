from .rust_viz import VizEngine
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

class TrainingVisualizer:
    def __init__(self):
        try:
            self.engine = VizEngine()
            print("Using Rust visualization engine")
            
            # Setup matplotlib figure
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.show(block=False)
            
        except Exception as e:
            print(f"Warning: Failed to load Rust visualization engine: {e}")
            print(f"Error details: {str(e)}")
            raise e
        
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.learning_rates = []
    
    def update(self, epoch, train_loss, val_loss, lr, accuracy):
        """Update visualization with new data"""
        try:
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.accuracies.append(accuracy)
            self.learning_rates.append(lr)
            
            # Update Rust visualization
            self.engine.update(
                train_loss,
                val_loss,
                accuracy,
                lr
            )
            
            # Update matplotlib plot
            self.ax.clear()
            epochs = range(len(self.train_losses))
            
            # Plot training loss
            self.ax.plot(epochs, self.train_losses, 'r-', label='Training Loss')
            # Plot validation loss
            self.ax.plot(epochs, self.val_losses, 'b-', label='Validation Loss')
            # Plot accuracy
            self.ax.plot(epochs, self.accuracies, 'g-', label='Accuracy')
            
            self.ax.set_title('Training Progress')
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Value')
            self.ax.legend()
            self.ax.grid(True)
            
            # Refresh the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
            raise e
    
    def save(self, prefix='training'):
        """Save visualization data"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_dir = Path('outputs/visualization')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics data
        np.savez(
            save_dir / f'{prefix}_data_{timestamp}.npz',
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            accuracies=self.accuracies,
            learning_rates=self.learning_rates
        )
        
        # Save final plot
        plt.savefig(save_dir / f'{prefix}_plot_{timestamp}.png')