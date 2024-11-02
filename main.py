import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging
from pathlib import Path
import time
from src.models.transformer import Seq2SeqLSTM
from src.models.dataset import NoisyWordDataset
from src.utils.data_handler import DataHandler
from src.visualization.training_viz import TrainingVisualizer
from src.visualization.confusion_viz import ConfusionVisualizer
from src.visualization.stats_viz import StatisticsVisualizer
from src.visualization.error_viz import ErrorAnalysisVisualizer
from src.visualization.model_viz import ModelArchitectureVisualizer
from src.utils.progress import TrainingProgressTracker
import torch.cuda as cuda
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Ottoman OCR Training')
    parser.add_argument('--live-viz', type=str, default='training',
                       choices=['training', 'confusion', 'stats', 'error', 'model', 'none'],
                       help='Which visualization to show during training')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all visualizations at the end')
    return parser.parse_args()

def save_training_metrics(metrics, prefix='training'):
    """Save training metrics to JSON and CSV files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_dir / f'{prefix}_metrics_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save as CSV
    csv_path = output_dir / f'{prefix}_metrics_{timestamp}.csv'
    with open(csv_path, 'w') as f:
        # Write header
        headers = metrics[0].keys()
        f.write(','.join(headers) + '\n')
        # Write data
        for record in metrics:
            values = [str(record[h]) for h in headers]
            f.write(','.join(values) + '\n')
    
    logger.info(f"Saved training metrics to {json_path} and {csv_path}")

def train(args):
    # Force CUDA if available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow!")
    device = torch.device('cuda')  # Force CUDA
    torch.cuda.set_device(0)  # Use first GPU
    logger.info(f"Using device: {device} - {torch.cuda.get_device_name(0)}")
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Load and prepare data
    logger.info("Loading training data...")
    handler = DataHandler()
    pairs = handler.load_all_data(
        txt_path='outputs/noisy_clean_pairs.txt',
        json_path='goldset.json'
    )
    
    # Create dataset and move to GPU
    dataset = NoisyWordDataset(pairs)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Optimize data loading for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4
    )
    
    # Initialize model on GPU
    model = Seq2SeqLSTM(
        vocab_size=dataset.vocab_size,
        hidden_size=256,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Initialize optimizer and criterion on GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Initialize visualizer if requested
    visualizer = None
    if args.live_viz == 'training':
        try:
            visualizer = TrainingVisualizer()
        except Exception as e:
            logger.warning(f"Could not initialize visualizer: {e}")
    
    # Initialize metrics storage
    train_metrics = []
    best_loss = float('inf')
    
    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('outputs/visualization').mkdir(parents=True, exist_ok=True)
    
    try:
        with TrainingProgressTracker() as progress:
            # Create progress tasks
            epoch_task = progress.create_epoch_task(args.epochs)
            batch_task = progress.create_batch_task(len(train_loader))
            
            for epoch in range(args.epochs):
                model.train()
                total_train_loss = 0
                samples_processed = 0
                
                # Update epoch progress
                progress.update_task(epoch_task, advance=1, epoch=epoch+1)
                
                # Reset batch progress for new epoch
                progress.reset_batch_progress()
                
                for batch_idx, (noisy, clean) in enumerate(train_loader):
                    # Move data to GPU
                    noisy = noisy.to(device, non_blocking=True)
                    clean = clean.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    output = model(noisy)
                    loss = criterion(output.view(-1, dataset.vocab_size), clean.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    total_train_loss += loss.item()
                    samples_processed += noisy.size(0)
                    
                    # Calculate CUDA memory usage
                    cuda_memory_used = cuda.memory_allocated() / 1024**2  # MB
                    
                    # Update progress
                    progress.update_task(
                        batch_task,
                        advance=1,
                        batch=batch_idx+1,
                        loss=loss.item(),
                        samples_processed=samples_processed,
                        cuda_memory=cuda_memory_used,
                        learning_rate=optimizer.param_groups[0]['lr']
                    )
                
                # Validation phase
                model.eval()
                total_val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy = noisy.to(device, non_blocking=True)
                        clean = clean.to(device, non_blocking=True)
                        
                        output = model(noisy)
                        loss = criterion(output.view(-1, dataset.vocab_size), clean.view(-1))
                        total_val_loss += loss.item()
                        
                        pred = output.argmax(dim=-1)
                        correct += (pred == clean).sum().item()
                        total += clean.numel()
                
                # Update metrics after validation
                accuracy = (correct / total) * 100
                progress.update_metrics(
                    accuracy=accuracy,
                    loss=total_val_loss / len(val_loader)
                )
                
                # Save checkpoint if best model
                if total_val_loss < best_loss:
                    best_loss = total_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, 'checkpoints/best_model.pt')
                
                # Save training metrics
                train_metrics.append({
                    'epoch': epoch,
                    'train_loss': total_train_loss / len(train_loader),
                    'val_loss': total_val_loss / len(val_loader),
                    'accuracy': accuracy,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'cuda_memory': cuda.memory_allocated() / 1024**2,  # MB
                    'samples_processed': samples_processed,
                    'time_elapsed': time.time() - progress._start_time
                })
                
                # Calculate average losses for this epoch
                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(val_loader)
                
                # Update visualization if available
                if visualizer:
                    visualizer.update(
                        epoch,
                        avg_train_loss,
                        avg_val_loss,
                        optimizer.param_groups[0]['lr'],
                        accuracy
                    )
    
    except KeyboardInterrupt:
        logger.info('Training interrupted')
    
    finally:
        # Save training metrics if requested
        if args.save_all and train_metrics:  # Check if we have metrics to save
            save_training_metrics(train_metrics)
        
        logger.info('Training completed')
        
        if visualizer:
            visualizer.save('training')

if __name__ == '__main__':
    args = parse_args()
    train(args)