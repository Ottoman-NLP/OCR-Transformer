import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class NoisyWordDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], max_length: int = 30):
        self.pairs = pairs
        self.max_length = max_length
        self.char_to_idx = {'<pad>': 0}  # Start with padding token
        self.build_vocab()
        
    def build_vocab(self):
        """Build vocabulary from all words in pairs."""
        # Add characters from both noisy and clean words
        for noisy, clean in self.pairs:
            for char in noisy + clean:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = len(self.char_to_idx)
        
        # Create reverse mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        logger.info(f"Created vocabulary with {self.vocab_size - 1} characters")
        logger.info(f"Vocabulary size: {self.vocab_size}")
    
    def _encode_word(self, word: str) -> torch.Tensor:
        """Convert word to tensor of character indices."""
        # Ensure word length is within bounds
        word = word[:self.max_length]
        # Convert characters to indices
        indices = [self.char_to_idx.get(c, 0) for c in word]
        # Pad sequence
        padding_length = self.max_length - len(indices)
        indices = indices + [0] * padding_length
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        return self._encode_word(noisy), self._encode_word(clean) 