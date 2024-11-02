from rust_viz import VizEngine
import time
import math

def test_visualization():
    viz = VizEngine()
    
    for i in range(100):
        train_loss = math.exp(-i/50) + 0.1
        val_loss = train_loss + 0.05 * math.sin(i/10)
        accuracy = 100 * (1 - math.exp(-i/30))
        lr = 0.001 * math.exp(-i/200)
        
        viz.update(train_loss, val_loss, accuracy, lr)
        time.sleep(0.1)  # Wait a bit to see the plot update
        print(f"Step {i}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, accuracy={accuracy:.2f}%")

if __name__ == "__main__":
    test_visualization()