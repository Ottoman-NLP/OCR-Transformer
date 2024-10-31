import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

try:
    # Import the module directly
    from src.visualization.cpp.viz_engine import VizEngine
    print("Successfully imported VizEngine!")
    
    # Create an instance
    viz = VizEngine()
    print("Successfully created VizEngine instance!")
    
    # Test visualization with some dummy data
    for i in range(5):
        train_loss = 1.0 - (i * 0.1)  # Decreasing train loss
        val_loss = 0.9 - (i * 0.1)    # Decreasing validation loss
        accuracy = 0.5 + (i * 0.1)     # Increasing accuracy
        lr = 0.001 * (0.9 ** i)        # Decreasing learning rate
        
        viz.update(train_loss, val_loss, accuracy, lr)
        print(f"Step {i}: Updated visualization with:")
        print(f"  Train Loss: {train_loss:.3f}")
        print(f"  Val Loss:   {val_loss:.3f}")
        print(f"  Accuracy:   {accuracy:.3f}")
        print(f"  LR:         {lr:.6f}")
        
    print("\nAll tests passed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    
    # Print more debugging info
    print("\nChecking file existence:")
    print(f"viz_engine.pyd exists: {os.path.exists('viz_engine.pyd')}")
    print(f"viz_engine.cp311-win_amd64.pyd exists: {os.path.exists('viz_engine.cp311-win_amd64.pyd')}")
    print("\nDirectory contents:")
    print(os.listdir('.'))