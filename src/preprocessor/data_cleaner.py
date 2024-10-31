import re
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, 
                 max_word_length: int = 15,
                 min_word_length: int = 3,
                 stop_words: Set[str] = None):
        
        self.max_word_length = max_word_length
        self.min_word_length = min_word_length
        self.stop_words = stop_words or self._default_stop_words()
        
        # Compile regex patterns
        self.number_pattern = re.compile(r'\d')
        self.symbol_pattern = re.compile(r'[^\w\s\']')  # Allow apostrophes
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')  # 3+ repeated chars
        
        # Add Ottoman-specific patterns
        self.cojoined_pattern = re.compile(r'([a-zçğıöşü])\1{3,}', re.IGNORECASE)  # Detect repeated chars
        self.invalid_pattern = re.compile(r'[^a-zçğıöşü\s\'\-]', re.IGNORECASE)  # Only allow valid Ottoman chars
        
        # Add validation for Ottoman-specific rules
        self.max_consonants = 4  # Max consecutive consonants
        self.consonants = set('bcçdfgğhjklmnprsştvyz')
        
        # Statistics
        self.total_words = 0
        self.filtered_words = 0
        self.reasons = {
            'length': 0,
            'numbers': 0,
            'symbols': 0,
            'stop_words': 0,
            'repeated_chars': 0,
            'encoding': 0,
            'invalid_pattern': 0
        }
    
    def _default_stop_words(self) -> Set[str]:
            
        """Default Ottoman Turkish and historical Turkish stop words (up to 3 letters, excluding suffixes and particles)."""
        return {
            'ad', 'ak', 'al', 'ama', 'an', 'at', 'az', 'ben', 'bir', 'biz', 'bu',
            'çık', 'çok', 'el', 'en', 'er', 'ev', 'gel', 'git', 'gör',
            'ha', 'hem', 'hep', 'her', 'iç', 'iki', 'il', 'ile', 'ki', 'kim', 'koy', 'kıl',
            'ne', 'o', 'on', 'ot', 'sen', 'su', 'şu', 'üç', 'var', 've', 'ya', 'yak', 'yaz', 'yok',
        }

    
    def is_valid_word(self, word: str) -> bool:
        """Check if word meets all criteria."""
        self.total_words += 1
        
        # Check length
        if len(word) < self.min_word_length or len(word) > self.max_word_length:
            self.reasons['length'] += 1
            return False
        
        # Check for numbers
        if self.number_pattern.search(word):
            self.reasons['numbers'] += 1
            return False
        
        # Check for symbols (except apostrophes)
        if self.symbol_pattern.search(word):
            self.reasons['symbols'] += 1
            return False
        
        # Check for stop words
        if word.lower() in self.stop_words:
            self.reasons['stop_words'] += 1
            return False
        
        # Check for repeated characters
        if self.repeated_char_pattern.search(word):
            self.reasons['repeated_chars'] += 1
            return False
        
        # Check for potential encoding issues
        try:
            word.encode('utf-8').decode('utf-8')
        except UnicodeError:
            self.reasons['encoding'] += 1
            return False
        
        # Add Ottoman-specific validation
        word = word.lower().strip()
        
        # Check for cojoined words
        if self.cojoined_pattern.search(word):
            self.reasons['repeated_chars'] += 1
            return False
            
        # Check for invalid characters
        if self.invalid_pattern.search(word):
            self.reasons['symbols'] += 1
            return False
            
        # Check consecutive consonants
        consonant_count = 0
        for char in word:
            if char in self.consonants:
                consonant_count += 1
                if consonant_count > self.max_consonants:
                    self.reasons['invalid_pattern'] += 1
                    return False
            else:
                consonant_count = 0
        
        return True
    
    def clean_text(self, text: str) -> List[str]:
        """Clean text and return valid words."""
        words = text.strip().split()
        valid_words = [word for word in words if self.is_valid_word(word)]
        self.filtered_words += len(words) - len(valid_words)
        return valid_words
    
    def get_statistics(self) -> dict:
        """Return cleaning statistics."""
        return {
            'total_words': self.total_words,
            'filtered_words': self.filtered_words,
            'accepted_words': self.total_words - self.filtered_words,
            'acceptance_rate': (self.total_words - self.filtered_words) / self.total_words if self.total_words > 0 else 0,
            'rejection_reasons': self.reasons
        } 