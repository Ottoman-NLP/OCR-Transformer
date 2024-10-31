import logging
from pathlib import Path
from src.preprocessor.data_cleaner import DataCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_data():
    data_path = Path('outputs/noisy_clean_pairs.txt')
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return False
        
    cleaner = DataCleaner()
    total_pairs = 0
    valid_pairs = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_pairs += 1
                try:
                    noisy, clean = line.strip().split('\t')
                    if cleaner.is_valid_word(clean):
                        valid_pairs += 1
                except ValueError:
                    logger.warning(f"Invalid line format: {line.strip()}")
    
    logger.info(f"Total pairs: {total_pairs}")
    logger.info(f"Valid pairs: {valid_pairs}")
    logger.info(f"Validation rate: {valid_pairs/total_pairs:.2%}")
    
    return valid_pairs > 0

if __name__ == "__main__":
    verify_data() 