import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime

class ConfusionVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('Character-Level Confusion Matrix')
        Path('outputs/visualization').mkdir(parents=True, exist_ok=True)
        
    def update(self, true_chars, pred_chars, char_to_idx):
        """Update confusion matrix visualization."""
        self.ax.clear()
        
        # Create confusion matrix
        n_classes = len(char_to_idx)
        matrix = np.zeros((n_classes, n_classes))
        
        for t, p in zip(true_chars, pred_chars):
            matrix[char_to_idx[t], char_to_idx[p]] += 1
            
        # Normalize
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        
        # Plot
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='YlOrRd',
                   xticklabels=list(char_to_idx.keys()),
                   yticklabels=list(char_to_idx.keys()),
                   ax=self.ax)
        
        self.ax.set_title('Character Confusion Matrix')
        self.ax.set_xlabel('Predicted')
        self.ax.set_ylabel('True')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save(self, prefix='confusion'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.fig.savefig(f'outputs/visualization/{prefix}_matrix_{timestamp}.png',
                        bbox_inches='tight', dpi=300)
        self.fig.savefig(f'outputs/visualization/{prefix}_matrix_{timestamp}.pdf',
                        bbox_inches='tight') 