from .viz_engine import VizEngine
import time
import math

def test_visualization():
    viz = VizEngine()
    
    # Simulate some training data
    for i in range(100):
        train_loss = math.exp(-i/50) + 0.1  # Decreasing exponential
        val_loss = train_loss + 0.05 * math.sin(i/10)  # Slightly higher with oscillation
        accuracy = 100 * (1 - math.exp(-i/30))  # Increasing accuracy
        lr = 0.001 * math.exp(-i/200)  # Decreasing learning rate
        
        viz.update(train_loss, val_loss, accuracy, lr)
        time.sleep(0.1)  # Update every 100ms

if __name__ == "__main__":
    test_visualization()