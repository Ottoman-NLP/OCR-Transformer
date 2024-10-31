import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    try:
        from src.preprocessor.data_cleaner import DataCleaner
        from src.models.dataset import NoisyWordDataset
        from src.models.transformer import Seq2SeqLSTM
        from src.visualization.training_viz import TrainingVisualizer
        logger.info("All imports successful!")
        return True
    except Exception as e:
        logger.error(f"Import error: {e}")
        return False

def test_directories():
    """Test required directories exist"""
    required_dirs = [
        'checkpoints',
        'outputs/visualization',
        'logs',
        'src/models',
        'src/preprocessor',
        'src/utils',
        'src/visualization'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    if missing:
        logger.warning(f"Created missing directories: {missing}")
    else:
        logger.info("All required directories present")
    
    return len(missing) == 0

def test_files():
    """Test required files exist"""
    required_files = [
        'src/models/dataset.py',
        'src/models/transformer.py',
        'src/preprocessor/data_cleaner.py',
        'src/visualization/training_viz.py',
        'main.py',
        'setup.py'
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        logger.error(f"Missing required files: {missing}")
        return False
    
    logger.info("All required files present")
    return True

def main():
    logger.info("Testing project setup...")
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Test", test_directories),
        ("File Test", test_files)
    ]
    
    all_passed = True
    for name, test_func in tests:
        logger.info(f"\nRunning {name}...")
        if not test_func():
            all_passed = False
    
    if all_passed:
        logger.info("\nAll tests passed! Project setup is correct.")
    else:
        logger.error("\nSome tests failed. Please fix the issues above.")

if __name__ == "__main__":
    main() 