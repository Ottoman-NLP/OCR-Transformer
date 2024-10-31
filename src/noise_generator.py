import random
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class NoiseGenerator:
    """Generates synthetic noise for Ottoman Turkish words."""
    
    def add_space_noise(self, word: str) -> str:
        """
        Add random spaces while preserving apostrophes.
        """
        if len(word) < 4:
            return word
            
        parts = word.split("'")
        noisy_parts = []
        
        for part in parts:
            if len(part) < 3:  # Don't add spaces to short segments
                noisy_parts.append(part)
                continue
                
            chars = list(part)
            num_spaces = random.randint(1, len(chars) // 3)
            for _ in range(num_spaces):
                pos = random.randint(1, len(chars) - 1)
                chars.insert(pos, ' ')
            noisy_parts.append(''.join(chars))
        
        # Rejoin with apostrophes
        return "'".join(noisy_parts)
    
    def character_swap(self, word: str) -> str:
        """
        Swap adjacent characters while preserving apostrophes.
        """
        if len(word) < 3:
            return word
            
        parts = word.split("'")
        noisy_parts = []
        
        for part in parts:
            if len(part) < 3:
                noisy_parts.append(part)
                continue
                
            chars = list(part)
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            noisy_parts.append(''.join(chars))
        
        return "'".join(noisy_parts)
    
    def character_deletion(self, word: str) -> str:
        """
        Delete a random character while preserving apostrophes.
        """
        if len(word) < 4:
            return word
            
        parts = word.split("'")
        noisy_parts = []
        
        for part in parts:
            if len(part) < 3:
                noisy_parts.append(part)
                continue
                
            chars = list(part)
            pos = random.randint(0, len(chars) - 1)
            del chars[pos]
            noisy_parts.append(''.join(chars))
        
        return "'".join(noisy_parts)
    
    def generate_noisy_sample(self, word: str) -> str:
        """Apply random noise transformation."""
        noise_funcs = [
            self.add_space_noise,
            self.character_swap,
            self.character_deletion
        ]
        func = random.choice(noise_funcs)
        return func(word)

    def generate_pairs(self, words: List[str], num_samples: int = 1) -> List[Tuple[str, str]]:
        """
        Generate pairs of (noisy, clean) words.
        """
        pairs = []
        for word in words:
            for _ in range(num_samples):
                noisy = self.generate_noisy_sample(word)
                if noisy != word:  # Only keep if noise was actually added
                    pairs.append((noisy, word))
        
        return pairs 