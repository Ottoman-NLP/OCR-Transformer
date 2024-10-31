import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from matplotlib.patches import Patch

class ErrorAnalysisVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Detailed Error Analysis', fontsize=16, y=0.95)
        self.gs = self.fig.add_gridspec(2, 3)
        
        # Create subplots
        self.error_dist_ax = self.fig.add_subplot(self.gs[0, 0])
        self.error_pattern_ax = self.fig.add_subplot(self.gs[0, 1])
        self.error_position_ax = self.fig.add_subplot(self.gs[0, 2])
        self.error_heatmap_ax = self.fig.add_subplot(self.gs[1, :2])
        self.error_stats_ax = self.fig.add_subplot(self.gs[1, 2])
        
        # Style configuration
        plt.style.use('default')
        self.colors = sns.color_palette("husl", 8)
        
        self.fig.canvas.manager.set_window_title('Error Analysis Dashboard')
        Path('outputs/visualization/error_analysis').mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.error_types = defaultdict(list)
        self.position_errors = defaultdict(int)
        self.error_patterns = defaultdict(int)
        
    def update(self, true_words, pred_words, char_to_idx):
        """Update error analysis visualization."""
        self._analyze_errors(true_words, pred_words)
        self._plot_error_distribution()
        self._plot_error_patterns()
        self._plot_position_analysis()
        self._plot_error_heatmap(true_words, pred_words)
        self._plot_error_statistics()
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _analyze_errors(self, true_words, pred_words):
        """Analyze different types of errors."""
        self.error_types.clear()
        self.position_errors.clear()
        self.error_patterns.clear()
        
        for true, pred in zip(true_words, pred_words):
            if true != pred:
                # Analyze error type
                if len(true) != len(pred):
                    self.error_types['length_mismatch'].append((true, pred))
                elif any(t != p for t, p in zip(true, pred)):
                    self.error_types['character_substitution'].append((true, pred))
                
                # Analyze position errors
                for i, (t, p) in enumerate(zip(true, pred)):
                    if t != p:
                        self.position_errors[i] += 1
                
                # Analyze error patterns
                pattern = f"{true}->{pred}"
                self.error_patterns[pattern] += 1
    
    def _plot_error_distribution(self):
        """Plot distribution of error types."""
        self.error_dist_ax.clear()
        error_counts = {k: len(v) for k, v in self.error_types.items()}
        
        sns.barplot(x=list(error_counts.keys()), 
                   y=list(error_counts.values()),
                   palette=self.colors,
                   ax=self.error_dist_ax)
        
        self.error_dist_ax.set_title('Error Type Distribution')
        self.error_dist_ax.set_xticklabels(self.error_dist_ax.get_xticklabels(), 
                                         rotation=45, ha='right')
    
    def _plot_error_patterns(self):
        """Plot most common error patterns."""
        self.error_pattern_ax.clear()
        
        # Get top 10 error patterns
        top_patterns = dict(sorted(self.error_patterns.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:10])
        
        sns.barplot(x=list(top_patterns.values()),
                   y=list(top_patterns.keys()),
                   palette=self.colors,
                   ax=self.error_pattern_ax)
        
        self.error_pattern_ax.set_title('Top 10 Error Patterns')
    
    def _plot_position_analysis(self):
        """Plot error frequency by position."""
        self.error_position_ax.clear()
        
        positions = list(self.position_errors.keys())
        counts = list(self.position_errors.values())
        
        sns.lineplot(x=positions, y=counts, 
                    marker='o', 
                    ax=self.error_position_ax)
        
        self.error_position_ax.set_title('Error Frequency by Position')
        self.error_position_ax.set_xlabel('Character Position')
        self.error_position_ax.set_ylabel('Error Count')
    
    def _plot_error_heatmap(self, true_words, pred_words):
        """Plot error correlation heatmap."""
        self.error_heatmap_ax.clear()
        
        # Create correlation matrix of errors
        error_pairs = defaultdict(int)
        for true, pred in zip(true_words, pred_words):
            if true != pred:
                for i, (t, p) in enumerate(zip(true, pred)):
                    if t != p:
                        error_pairs[f"{t}->{p}"] += 1
        
        # Convert to matrix
        labels = sorted(error_pairs.keys())
        matrix = np.zeros((len(labels), len(labels)))
        for i, l1 in enumerate(labels):
            for j, l2 in enumerate(labels):
                if i != j:
                    matrix[i,j] = len(set(error_pairs[l1]) & set(error_pairs[l2]))
        
        sns.heatmap(matrix, 
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap='YlOrRd',
                   ax=self.error_heatmap_ax)
        
        self.error_heatmap_ax.set_title('Error Correlation Heatmap')
        plt.setp(self.error_heatmap_ax.get_xticklabels(), 
                rotation=45, ha='right')
    
    def _plot_error_statistics(self):
        """Plot error statistics."""
        self.error_stats_ax.clear()
        self.error_stats_ax.axis('off')
        
        stats_text = "Error Statistics\n\n"
        
        # Calculate statistics
        total_errors = sum(len(v) for v in self.error_types.values())
        stats_text += f"Total Errors: {total_errors}\n\n"
        
        for error_type, errors in self.error_types.items():
            rate = len(errors) / total_errors * 100
            stats_text += f"{error_type}:\n"
            stats_text += f"  Count: {len(errors)}\n"
            stats_text += f"  Rate: {rate:.2f}%\n\n"
        
        self.error_stats_ax.text(0.05, 0.95, stats_text,
                                transform=self.error_stats_ax.transAxes,
                                verticalalignment='top',
                                fontsize=10)
    
    def save(self, prefix='error_analysis'):
        """Save visualizations."""
        from datetime import datetime
        import json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save high-resolution images
        self.fig.savefig(f'outputs/visualization/error_analysis/{prefix}_{timestamp}.png',
                        bbox_inches='tight', dpi=300)
        self.fig.savefig(f'outputs/visualization/error_analysis/{prefix}_{timestamp}.pdf',
                        bbox_inches='tight')
        
        # Save error statistics
        stats = {
            'error_types': {k: len(v) for k, v in self.error_types.items()},
            'position_errors': dict(self.position_errors),
            'error_patterns': dict(self.error_patterns)
        }
        
        with open(f'outputs/visualization/error_analysis/{prefix}_stats_{timestamp}.json', 'w') as f:
            json.dump(stats, f, indent=2) 