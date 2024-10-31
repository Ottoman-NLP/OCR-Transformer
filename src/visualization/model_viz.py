import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle, FancyArrowPatch
import networkx as nx

class ModelArchitectureVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(20, 15))
        self.fig.suptitle('Model Architecture Visualization', fontsize=16, y=0.95)
        
        # Style configuration
        plt.style.use('default')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        self.fig.canvas.manager.set_window_title('Model Architecture')
        Path('outputs/visualization/architecture').mkdir(parents=True, exist_ok=True)
    
    def visualize_architecture(self, model):
        """Create detailed visualization of model architecture."""
        self.fig.clear()
        
        # Create graph
        G = nx.DiGraph()
        pos = {}
        layer_nodes = self._analyze_model(model)
        
        # Position nodes
        if layer_nodes:  # Check if we have any nodes
            max_layer_size = max(len(nodes) for nodes in layer_nodes.values())
            num_layers = len(layer_nodes)
            
            for layer_idx, nodes in layer_nodes.items():
                layer_x = layer_idx / max(1, num_layers - 1)  # Prevent division by zero
                for node_idx, node in enumerate(nodes):
                    node_y = node_idx / max(1, max_layer_size - 1)  # Prevent division by zero
                    G.add_node(node['name'], **node)
                    pos[node['name']] = (layer_x, node_y)
        
            # Add edges
            self._add_edges(G, layer_nodes)
            
            # Draw
            ax = self.fig.add_subplot(111)
            self._draw_architecture(G, pos, ax)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _analyze_model(self, model):
        """Analyze model structure."""
        from collections import defaultdict
        layer_nodes = defaultdict(list)
        
        def add_module(module, name, layer_idx):
            params = sum(p.numel() for p in module.parameters())
            layer_nodes[layer_idx].append({
                'name': name,
                'type': module.__class__.__name__,
                'params': params,
                'shape': self._get_layer_shape(module)
            })
        
        # Analyze model structure
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_idx = len(name.split('.'))
                add_module(module, name, layer_idx)
        
        return layer_nodes
    
    def _get_layer_shape(self, module):
        """Get layer output shape if available."""
        if hasattr(module, 'out_features'):
            return f"({module.out_features})"
        elif hasattr(module, 'out_channels'):
            return f"({module.out_channels})"
        return ""
    
    def _add_edges(self, G, layer_nodes):
        """Add edges between layers."""
        layers = sorted(layer_nodes.keys())
        for i in range(len(layers)-1):
            current_layer = layers[i]
            next_layer = layers[i+1]
            
            for node1 in layer_nodes[current_layer]:
                for node2 in layer_nodes[next_layer]:
                    if node2['name'].startswith(node1['name']):
                        G.add_edge(node1['name'], node2['name'])
    
    def _draw_architecture(self, G, pos, ax):
        """Draw the architecture graph."""
        # Draw nodes
        for node in G.nodes():
            node_data = G.nodes[node]
            color = self.colors[hash(node_data['type']) % len(self.colors)]
            
            # Create node label
            label = f"{node_data['type']}\n{node_data['shape']}\n{node_data['params']} params"
            
            # Draw node
            bbox = dict(facecolor=color, edgecolor='black', alpha=0.7)
            ax.text(pos[node][0], pos[node][1], label,
                   bbox=bbox, ha='center', va='center')
        
        # Draw edges
        for edge in G.edges():
            start = pos[edge[0]]
            end = pos[edge[1]]
            arrow = FancyArrowPatch(start, end,
                                  arrowstyle='->',
                                  color='gray',
                                  connectionstyle='arc3,rad=0.2')
            ax.add_patch(arrow)
        
        # Set limits and remove axes
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
    
    def save(self, prefix='architecture'):
        """Save visualizations."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save high-resolution images
        self.fig.savefig(f'outputs/visualization/architecture/{prefix}_{timestamp}.png',
                        bbox_inches='tight', dpi=300)
        self.fig.savefig(f'outputs/visualization/architecture/{prefix}_{timestamp}.pdf',
                        bbox_inches='tight') 