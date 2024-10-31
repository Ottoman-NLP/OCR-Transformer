from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer
import pyqtgraph as pg
import sys
import numpy as np
from pathlib import Path
import threading
import time

class TrainingVisualizer(QMainWindow):
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()
        
        # Setup main window
        self.setWindowTitle('Training Progress')
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create plots with dark background
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        
        self.loss_plot = pg.PlotWidget(title='Loss')
        self.acc_plot = pg.PlotWidget(title='Accuracy')
        self.lr_plot = pg.PlotWidget(title='Learning Rate')
        
        # Add plots to layout
        layout.addWidget(self.loss_plot)
        layout.addWidget(self.acc_plot)
        layout.addWidget(self.lr_plot)
        
        # Initialize data storage
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.learning_rates = []
        self.epochs = []
        
        # Setup plot items with better colors
        self.train_curve = self.loss_plot.plot(pen=('b', 2), name='Train')
        self.val_curve = self.loss_plot.plot(pen=('r', 2), name='Validation')
        self.acc_curve = self.acc_plot.plot(pen=('g', 2))
        self.lr_curve = self.lr_plot.plot(pen=('y', 2))
        
        # Add legends and grid
        self.loss_plot.addLegend()
        for plot in [self.loss_plot, self.acc_plot, self.lr_plot]:
            plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1000)  # Update every second
        
        # Show window
        self.show()
        
        # Start event loop in a separate thread
        self.thread = threading.Thread(target=self.app.exec)
        self.thread.daemon = True
        self.thread.start()
    
    def update(self, epoch, train_loss, val_loss, lr, accuracy):
        """Update data"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.accuracies.append(accuracy)
        self.learning_rates.append(lr)
    
    def update_plots(self):
        """Update plot curves"""
        if self.epochs:
            self.train_curve.setData(self.epochs, self.train_losses)
            self.val_curve.setData(self.epochs, self.val_losses)
            self.acc_curve.setData(self.epochs, self.accuracies)
            self.lr_curve.setData(self.epochs, self.learning_rates)
    
    def save(self, prefix='training'):
        """Save plots and data"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_dir = Path('outputs/visualization')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        np.savez(
            save_dir / f'{prefix}_data_{timestamp}.npz',
            epochs=self.epochs,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            accuracies=self.accuracies,
            learning_rates=self.learning_rates
        )
        
        # Export plots as images
        for plot, name in [
            (self.loss_plot, 'loss'),
            (self.acc_plot, 'accuracy'),
            (self.lr_plot, 'lr')
        ]:
            exporter = pg.exporters.ImageExporter(plot)
            exporter.export(str(save_dir / f'{prefix}_{name}_{timestamp}.png')) 