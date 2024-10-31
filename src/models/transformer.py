import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size).to(self.device)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        ).to(self.device)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, vocab_size).to(self.device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Log model architecture
        logger.info(f"Model initialized on device: {self.device}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Embedding
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_len, hidden_size)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        
        # Output layer
        output = self.fc(lstm_out)
        # output shape: (batch_size, seq_len, vocab_size)
        
        return output
    
    def predict(self, x: torch.Tensor) -> str:
        """Make a prediction for a single input."""
        self.eval()
        with torch.no_grad():
            x = x.to(next(self.parameters()).device)
            output = self(x)
            predictions = torch.argmax(output, dim=-1)
            return predictions.cpu()  # Return predictions to CPU