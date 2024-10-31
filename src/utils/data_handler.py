import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Union
from src.preprocessor.data_cleaner import DataCleaner

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, cleaner: DataCleaner = None):
        self.cleaner = cleaner or DataCleaner()
        self.data_pairs: List[Tuple[str, str]] = []
        
    def load_txt_pairs(self, filepath: str) -> List[Tuple[str, str]]:
        """Load noisy-clean pairs from txt file."""
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        noisy, clean = line.strip().split('\t')
                        if self.cleaner.is_valid_word(clean):
                            pairs.append((noisy, clean))
                    except ValueError:
                        logger.warning(f"Invalid line format: {line.strip()}")
        return pairs
    
    def load_json_goldset(self, filepath: str) -> List[Tuple[str, str]]:
        """Load pairs from goldset JSON."""
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                noisy = item.get('noisy', '').strip()
                clean = item.get('clean', '').strip()
                if noisy and clean and self.cleaner.is_valid_word(clean):
                    pairs.append((noisy, clean))
        return pairs
    
    def load_all_data(self, txt_path: str = None, json_path: str = None) -> List[Tuple[str, str]]:
        """Load data from all available sources."""
        all_pairs = []
        
        if txt_path and Path(txt_path).exists():
            txt_pairs = self.load_txt_pairs(txt_path)
            logger.info(f"Loaded {len(txt_pairs)} pairs from txt file")
            all_pairs.extend(txt_pairs)
            
        if json_path and Path(json_path).exists():
            json_pairs = self.load_json_goldset(json_path)
            logger.info(f"Loaded {len(json_pairs)} pairs from goldset")
            all_pairs.extend(json_pairs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in all_pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)
        
        logger.info(f"Total unique pairs: {len(unique_pairs)}")
        return unique_pairs
    
    def save_processed_data(self, pairs: List[Tuple[str, str]], output_path: str):
        """Save processed pairs to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for noisy, clean in pairs:
                f.write(f"{noisy}\t{clean}\n")
        
        logger.info(f"Saved {len(pairs)} pairs to {output_path}") 