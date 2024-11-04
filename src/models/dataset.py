import torch
from torch.utils.data import Dataset
from collections import Counter
import random

class TextAugmenter:
    def __init__(self, swap_prob=0.1, delete_prob=0.1, substitute_prob=0.1):
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        self.substitute_prob = substitute_prob
        
        # Common Ottoman Turkish character substitutions
        self.substitutions = {
            'a': 'eâ', 'e': 'aê', 'i': 'ıî', 'ı': 'iî',
            'o': 'öô', 'ö': 'oô', 'u': 'üû', 'ü': 'uû'
        }
    
    def augment(self, text):
        chars = list(text)
        
        # Character swapping
        for i in range(len(chars)-1):
            if random.random() < self.swap_prob:
                chars[i], chars[i+1] = chars[i+1], chars[i]
        
        # Character deletion
        chars = [c for c in chars if random.random() > self.delete_prob]
        
        # Character substitution
        for i in range(len(chars)):
            if random.random() < self.substitute_prob:
                if chars[i] in self.substitutions:
                    chars[i] = random.choice(self.substitutions[chars[i]])
        
        return ''.join(chars)

class OttomanDataset(Dataset):
    def __init__(self, data_pairs, max_length=128):
        self.data = data_pairs
        self.max_length = max_length
        
        # Special tokens
        self.special_tokens = {
            'PAD': 0,
            'SOS': 1,
            'EOS': 2,
            'UNK': 3
        }
        
        # Build vocabulary
        self.build_vocab()
        
        # Initialize augmenter
        self.augmenter = TextAugmenter()
    
    def build_vocab(self):
        # Character frequency analysis
        char_freq = Counter()
        for noisy, clean in self.data:
            char_freq.update(noisy + clean)
        
        # Create character mappings
        self.char_to_idx = {char: idx + len(self.special_tokens) 
                           for idx, (char, freq) in enumerate(char_freq.items()) 
                           if freq >= 5}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Add special tokens to mappings
        self.char_to_idx.update(self.special_tokens)
        for token, idx in self.special_tokens.items():
            self.idx_to_char[idx] = token
        
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        return [self.char_to_idx.get(c, self.special_tokens['UNK']) for c in text]
    
    def decode(self, indices):
        return ''.join(self.idx_to_char.get(idx, '<UNK>') for idx in indices)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        noisy, clean = self.data[idx]
        
        # Apply augmentation during training
        if random.random() < 0.5:
            noisy = self.augmenter.augment(noisy)
        
        # Encode sequences
        src_ids = self.encode(noisy)
        tgt_ids = self.encode(clean)
        
        # Add SOS and EOS tokens
        src_ids = [self.special_tokens['SOS']] + src_ids + [self.special_tokens['EOS']]
        tgt_ids = [self.special_tokens['SOS']] + tgt_ids + [self.special_tokens['EOS']]
        
        # Convert to tensors (without padding - let collate_fn handle it)
        return {
            'input': torch.tensor(src_ids, dtype=torch.long),
            'target': torch.tensor(tgt_ids, dtype=torch.long)
        }

def collate_batch(batch):
    """Custom collate function to handle variable length sequences"""
    # Find max length in the batch
    max_src_len = max(len(item['input']) for item in batch)
    max_tgt_len = max(len(item['target']) for item in batch)
    
    # Add 1 for end token
    max_src_len += 1
    max_tgt_len += 1
    
    # Pad sequences
    padded_src = []
    padded_tgt = []
    
    for item in batch:
        src = item['input']
        tgt = item['target']
        
        # Add end token
        src = torch.cat([src, torch.tensor([2])])  # 2 is end token
        tgt = torch.cat([tgt, torch.tensor([2])])
        
        # Pad
        src_padding = torch.zeros(max_src_len - len(src))
        tgt_padding = torch.zeros(max_tgt_len - len(tgt))
        
        src = torch.cat([src, src_padding])
        tgt = torch.cat([tgt, tgt_padding])
        
        padded_src.append(src)
        padded_tgt.append(tgt)
    
    return {
        'input': torch.stack(padded_src).long(),
        'target': torch.stack(padded_tgt).long()
    }