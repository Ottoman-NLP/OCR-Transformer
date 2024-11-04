import torch
import torch.nn as nn
import random

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, vocab_size)  # *2 for bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is PAD token

    def forward(self, batch, teacher_forcing_ratio=0.5):
        # Handle both training and inference modes
        if isinstance(batch, dict):
            src = batch['input']
            target = batch.get('target', None)  # Use get() to handle missing target
        else:
            src = batch
            target = None
            
        batch_size = src.shape[0]
        max_len = src.shape[1]
        
        # Embedding and LSTM
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(outputs)
        
        # During training (when target is provided)
        if target is not None:
            predictions_flat = predictions.view(-1, self.vocab_size)
            target_flat = target.view(-1)
            return self.criterion(predictions_flat, target_flat)
            
        return predictions