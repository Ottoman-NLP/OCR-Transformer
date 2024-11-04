import torch
import torch.nn as nn

class TurkishAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Define character groups for Turkish
        self.char_groups = {
            'vowels': set('aeiouöüıâîû'),
            'similar_chars': {
                'i': 'ıi',
                'ı': 'ıi',
                'o': 'oö',
                'ö': 'oö',
                'u': 'uü',
                'ü': 'uü',
                'g': 'gğ',
                'ğ': 'gğ',
            }
        }
    
    def forward(self, pred, target):
        base_loss = self.base_criterion(pred, target)
        
        # Apply lower penalty for similar character substitutions
        pred_chars = pred.argmax(dim=-1)
        for i, (p, t) in enumerate(zip(pred_chars, target)):
            p_char = chr(p)
            t_char = chr(t)
            
            # Check if characters belong to same group
            if (p_char in self.char_groups['similar_chars'] and 
                t_char in self.char_groups['similar_chars'][p_char]):
                base_loss[i] *= 0.8
        
        return base_loss.mean() 