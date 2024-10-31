import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

class StatisticsVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.canvas.manager.set_window_title('Training Statistics')
        Path('outputs/visualization').mkdir(parents=True, exist_ok=True)
        
    def update(self, train_losses, val_losses, accuracies, learning_rates):
        """Update statistical visualizations."""
        self.axes[0,0].clear()
        self.axes[0,1].clear()
        self.axes[1,0].clear()
        self.axes[1,1].clear()
        
        # Loss distribution
        if len(train_losses) > 1:
            sns.histplot(train_losses, kde=True, ax=self.axes[0,0], label='Train')
            sns.histplot(val_losses, kde=True, ax=self.axes[0,0], label='Val')
        self.axes[0,0].set_title('Loss Distribution')
        self.axes[0,0].legend()
        
        # Learning curve analysis
        epochs = range(1, len(train_losses) + 1)
        self.axes[0,1].plot(epochs, train_losses, label='Train')
        self.axes[0,1].plot(epochs, val_losses, label='Val')
        self.axes[0,1].set_title('Learning Curves')
        self.axes[0,1].legend()
        
        # Accuracy progression
        if len(accuracies) > 0:
            sns.regplot(x=epochs, y=accuracies, ax=self.axes[1,0])
        self.axes[1,0].set_title('Accuracy Trend')
        
        # Learning rate decay
        self.axes[1,1].plot(epochs, learning_rates)
        self.axes[1,1].set_title('Learning Rate Decay')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save(self, prefix='statistics'):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.fig.savefig(f'outputs/visualization/{prefix}_analysis_{timestamp}.png',
                        bbox_inches='tight', dpi=300)
        self.fig.savefig(f'outputs/visualization/{prefix}_analysis_{timestamp}.pdf',
                        bbox_inches='tight') 