import logging
from src.utils.data_handler import DataHandler
from src.preprocessor.data_cleaner import DataCleaner
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Preprocess Ottoman text data')
    parser.add_argument('--txt', type=str, default='outputs/noisy_clean_pairs.txt',
                      help='Path to txt pairs file')
    parser.add_argument('--json', type=str, default='goldset.json',
                      help='Path to goldset json file')
    parser.add_argument('--output', type=str, default='outputs/processed_pairs.txt',
                      help='Output path for processed data')
    args = parser.parse_args()
    
    # Initialize handlers
    cleaner = DataCleaner()
    handler = DataHandler(cleaner)
    
    # Load and process data
    pairs = handler.load_all_data(args.txt, args.json)
    
    # Save processed data
    handler.save_processed_data(pairs, args.output)

if __name__ == "__main__":
    main() 