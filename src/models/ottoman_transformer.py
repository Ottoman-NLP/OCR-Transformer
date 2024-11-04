import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import logging
from warnings import filterwarnings
filterwarnings('ignore')

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from src.visualization.research_viz import ResearchVisualizer
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from ..visualization.live_plotter import LiveTrainingVisualizer
import seaborn as sns
from rich.console import Console
from rich.layout import Layout
from rich import box
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from torch.cuda.amp import GradScaler, autocast
from collections import deque
import torch.profiler
import GPUtil

plt.style.use('bmh')  # Using a built-in style, not seaborn

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class CharacterCNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x_conv = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        conv1_out = F.relu(self.conv1(x_conv))
        conv2_out = F.relu(self.conv2(x_conv))
        conv_out = (conv1_out + conv2_out).transpose(1, 2)  # [batch_size, seq_len, d_model]
        return self.layer_norm(conv_out)

class ContextualAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.vocab_size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for PAD and true label
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())

class OttomanTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_len=512):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # position embeddings
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, #TODO: check if this is correct
            nhead=nhead, #TODO: using 8 as default but check if this is correct
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_padding_mask(self, seq):
        return (seq == 0).to(seq.device)
    
    def forward(self, batch):
        try:
            src = batch['input']
            tgt = batch['target'][:, :-1]  # Remove last token for input BECAUSE OF Causal Mask

            if src.size(1) > self.max_seq_len:
                src = src[:, :self.max_seq_len]
            if tgt.size(1) > self.max_seq_len:
                tgt = tgt[:, :self.max_seq_len]
            
            # Create padding masks
            src_padding_mask = (src == 0).to(src.device)
            tgt_padding_mask = (tgt == 0).to(tgt.device)

            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Embeddings and positional encoding
            src = self.embedding(src) * math.sqrt(self.d_model)
            tgt = self.embedding(tgt) * math.sqrt(self.d_model)
            
            src = self.pos_encoder(src)
            tgt = self.pos_encoder(tgt)
            
            # Forward passes for encoder and decoder
            memory = self.encoder(
                src,
                src_key_padding_mask=src_padding_mask
            )
            
            output = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            logits = self.fc_out(output)
            
            # Calculate loss using Cross Entropy
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch['target'][:, 1:].reshape(-1),
                ignore_index=0  # Ignore padding index
            )
            
            return {
                'loss': loss,
                'logits': logits,
                'predictions': logits.argmax(-1)
            }
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Shapes - src: {src.shape}, tgt: {tgt.shape}")
            logger.error(f"Device - src: {src.device}, tgt: {tgt.device}")
            raise

def get_training_config():
    return {
        'model': {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.2,  # Increased dropout
            'max_seq_len': 512
        },
        'training': {
            'batch_size': 64,  # Significantly reduced
            'epochs': 20,
            'learning_rate': 1e-4,  # Significantly reduced
            'weight_decay': 0.1,  # Increased regularization
            'warmup_steps': 4000,  # Increased warmup
            'max_grad_norm': 0.1,  # Aggressive gradient clipping
            'gradient_accumulation_steps': 4,  # Increased to simulate larger batch
            'fp16_precision': False,  # Disable mixed precision temporarily
            'empty_cache_freq': 500
        }
    }

class TrainingMetricsTracker:
    def __init__(self, save_dir='outputs/metrics', total_steps=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_times = deque(maxlen=100)  # Store last 100 batch times for moving average
        self.start_time = time.time()
        self.last_time = self.start_time
        self.progress = Progress( #TODO: SHOULD BE CHANGED TO RICH PROGRESS AND BETTER INTEGRATION WITH RICH VISUALIZATION
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[red]Loss: {task.fields[loss]:.4f}"),
            TextColumn("[yellow]LR: {task.fields[lr]:.2e}"),
            TextColumn("[green]{task.fields[speed]:.1f} samples/s"),
            TextColumn("[cyan]GPU: {task.fields[gpu_util]}%"),
            TextColumn("[magenta]{task.fields[gpu_mem]:.1f}GB"),
            TimeRemainingColumn(),
            console=Console(),
            transient=True,
            expand=True
        )
        
        self.task = self.progress.add_task(
            "Training",
            total=total_steps,
            loss=0.0,
            lr=0.0,
            speed=0.0,
            gpu_util=0,
            gpu_mem=0
        )

    def __enter__(self):
        """Start the progress display"""
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the progress display"""
        self.progress.stop()
        if exc_type is not None:
            logger.error(f"Training error: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

    def update(self, **kwargs):
        """Update metrics with accurate timing"""
        torch.cuda.synchronize()
        current_time = time.time()
        batch_time = current_time - self.last_time
        self.batch_times.append(batch_time)
        self.last_time = current_time
        
        # Calculate average speed over last N batches (100)
        avg_time = sum(self.batch_times) / len(self.batch_times)
        samples_per_sec = kwargs.get('batch_size', 32) / avg_time if avg_time > 0 else 0

        try:
            gpu = GPUtil.getGPUs()[0]
            gpu_util = gpu.load * 100
            gpu_mem = gpu.memoryUsed / 1024 # GB
        except:
            gpu_util = 0
            gpu_mem = 0
        
        self.progress.update(
            self.task,
            advance=1,
            description=f"Epoch {kwargs.get('epoch', '0/0')}",
            loss=kwargs.get('train_loss', 0),
            lr=kwargs.get('learning_rate', 0),
            speed=samples_per_sec,
            gpu_util=gpu_util,
            gpu_mem=gpu_mem
        )

def train_model(args, dataset, train_loader, val_loader, device):
    config = get_training_config()
    logger.info("Initializing model and training components...")
    
    # Enable TF32 for better numerical stability
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize metrics tracker first
    metrics_dir = Path(args.output_dir) / 'metrics'
    total_steps = len(train_loader) * config['training']['epochs']
    
    try:
        model = OttomanTransformer(vocab_size=dataset.vocab_size, **config['model']).to(device)
        model = model.float()  # Ensure float32 precision
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Initialize optimizer for adaptive learning rate --> AdamW
        optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            eps=1e-8
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps
        )

        logger.info("Starting training loop...")
        with TrainingMetricsTracker(save_dir=metrics_dir, total_steps=total_steps) as metrics:
            for epoch in range(config['training']['epochs']):
                model.train()
                running_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # Move batch to device
                        batch = {k: v.to(device) for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = model(batch)
                        loss = outputs['loss']
                        
                        # Check for NaN loss
                        if torch.isnan(loss):
                            logger.error(f"NaN loss detected at batch {batch_idx}, skipping...")
                            continue
                        
                        # Scale loss for gradient accumulation
                        loss = loss / config['training']['gradient_accumulation_steps']
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient accumulation
                        if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                config['training']['max_grad_norm']
                            )
                            
                            # Check for NaN gradients
                            valid_gradients = True
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any():
                                        logger.error(f"NaN gradient in {name}, skipping update...")
                                        valid_gradients = False
                                        break
                            
                            if valid_gradients:
                                optimizer.step()
                                scheduler.step()
                            
                            optimizer.zero_grad(set_to_none=True)
                        
                        running_loss += loss.item()
                        num_batches += 1
                        
                        if batch_idx % args.log_interval == 0:
                            avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
                            metrics.update(
                                epoch=f"{epoch}/{config['training']['epochs']}",
                                batch_size=args.batch_size,
                                train_loss=avg_loss,
                                learning_rate=scheduler.get_last_lr()[0]
                            )
                    
                    except RuntimeError as e:
                        logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            continue
                        raise
                    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

def validate_step(model, val_loader, idx_to_char, device):
    """Compute all validation metrics in one pass"""
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_char_errors = 0
    total_chars = 0
    total_word_errors = 0
    total_words = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            output = model(batch)
            logits = output['logits']
            
            # Ensure predictions and targets have the same shape
            predictions = logits.argmax(dim=-1)
            targets = batch['target']
            
            # Trim predictions or targets if necessary to match shapes
            min_len = min(predictions.size(1), targets.size(1))
            predictions = predictions[:, :min_len]
            targets = targets[:, :min_len]
            
            # Create mask for non-padding tokens
            mask = (targets != 0) & (targets != 1) & (targets != 2)
            
            # Compute accuracy
            total_correct += (predictions[mask] == targets[mask]).sum().item()
            total_tokens += mask.sum().item()
            
            # Convert to text and compute CER/WER
            for pred, tgt in zip(predictions, targets):
                pred_text = ''.join([idx_to_char[idx.item()] for idx in pred if idx.item() not in [0, 1, 2]])
                tgt_text = ''.join([idx_to_char[idx.item()] for idx in tgt if idx.item() not in [0, 1, 2]])
                
                if len(tgt_text) > 0:
                    # Character-level metrics
                    total_chars += len(tgt_text)
                    total_char_errors += levenshtein_distance(pred_text, tgt_text)
                    
                    # Word-level metrics
                    pred_words = pred_text.split()
                    tgt_words = tgt_text.split()
                    total_words += len(tgt_words)
                    total_word_errors += levenshtein_distance(pred_words, tgt_words)
    
    return {
        'accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
        'cer': total_char_errors / total_chars if total_chars > 0 else 1.0,
        'wer': total_word_errors / total_words if total_words > 0 else 1.0
    }

def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance between two strings or lists"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

class TrainingVisualizer:
    def __init__(self, save_path='etc/visualization'):
        self.save_path = save_path
        self.metrics = defaultdict(list)
        os.makedirs(save_path, exist_ok=True)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.metrics[k].append(v)
    
    def save_plots(self):
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['loss'], label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{self.save_path}/training_loss.png')
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['accuracy'], label='Accuracy')
        plt.plot(self.metrics['cer'], label='CER')
        plt.plot(self.metrics['wer'], label='WER')
        plt.title('Model Performance Metrics')
        plt.xlabel('Batch')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig(f'{self.save_path}/performance_metrics.png')
        plt.close()

def save_checkpoint(model, optimizer, epoch, save_dir):
    """Save model checkpoint with optimizer state"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': model.embedding.num_embeddings,
        'd_model': model.d_model,
        'config': {
            'nhead': model.transformer.nhead,
            'num_encoder_layers': len(model.transformer.encoder.layers),
            'num_decoder_layers': len(model.transformer.decoder.layers),
            'dim_feedforward': model.transformer.encoder.layers[0].linear1.out_features,
            'dropout': model.transformer.encoder.layers[0].dropout.p
        }
    }
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_path = save_dir / f'checkpoint_epoch{epoch}_{timestamp}.pt'
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest checkpoint (overwrite)
    latest_path = save_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    
    logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Optional: Remove old checkpoints to save space
    keep_last_n = 3  # Keep only last N checkpoints
    checkpoints = sorted(save_dir.glob('checkpoint_epoch*.pt'))
    if len(checkpoints) > keep_last_n:
        for checkpoint_to_remove in checkpoints[:-keep_last_n]:
            checkpoint_to_remove.unlink()
            logging.info(f"Removed old checkpoint: {checkpoint_to_remove}")

def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved config
    model = OttomanTransformer(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        **checkpoint['config']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer (useful for resuming training)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch']