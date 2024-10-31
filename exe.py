import torch
from src.models.transformer import Seq2SeqLSTM
from src.preprocessor import TextPreprocessor
from src.models.dataset import NoisyWordDataset
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCorrector:
    def __init__(self, model_path='checkpoints/final_model.pt'):
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # Load vocabulary from training pairs
        logger.info("Loading vocabulary from training pairs...")
        pairs_path = 'outputs/noisy_clean_pairs.txt'
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"Training pairs not found at: {pairs_path}")
            
        # Read training pairs
        with open(pairs_path, 'r', encoding='utf-8') as f:
            pairs = [line.strip().split('\t') for line in f if line.strip()]
        
        # Initialize dataset with training pairs to get the same vocabulary
        self.dataset = NoisyWordDataset(pairs, max_length=30)
        vocab_size = self.dataset.vocab_size
        logger.info(f"Reconstructed vocabulary size: {vocab_size}")
        
        # Initialize model with correct vocabulary size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Seq2SeqLSTM(
            vocab_size=vocab_size,
            embed_size=128,
            hidden_size=256
        ).to(self.device)
        
        # Load trained weights
        logger.info(f"Loading model from {model_path}")
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Initialize preprocessor with same vocabulary
        self.preprocessor = TextPreprocessor()
        # Build vocabulary mappings
        self.preprocessor.char_to_idx = self.dataset.char_to_idx
        self.preprocessor.idx_to_char = {idx: char for char, idx in self.dataset.char_to_idx.items()}
        
        logger.info("Vocabulary loaded successfully")
        logger.info(f"Character set: {sorted(self.preprocessor.char_to_idx.keys())}")
        
        # Print some debug info
        logger.info("Loading model state:")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        logger.info(f"Model parameters: {len(state_dict)} layers")
        for name, param in state_dict.items():
            logger.info(f"{name}: {param.shape}")
    
    def correct_text(self, text: str) -> str:
        """Correct OCR errors in the given text."""
        try:
            # Split text into words
            words = text.strip().split()
            corrected_words = []
            
            # Process each word
            for word in words:
                # Debug input processing
                logger.debug(f"Processing word: {word}")
                tensor_input = self.preprocessor.text_to_tensor(word).unsqueeze(0).to(self.device)
                logger.debug(f"Input tensor: {tensor_input}")
                
                # Get model prediction
                with torch.no_grad():
                    output = self.model.predict(tensor_input)
                logger.debug(f"Output tensor: {output}")
                
                # Convert output tensor back to text
                corrected_word = self.preprocessor.tensor_to_text(output.squeeze(0))
                logger.debug(f"Corrected word: {corrected_word}")
                corrected_words.append(corrected_word)
            
            # Join words back together
            return ' '.join(corrected_words)
            
        except Exception as e:
            logger.error(f"Error during text correction: {e}")
            return text  # Return original text if correction fails

def main():
    # Set debug logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("\nOttoman Text Correction System")
    print("=" * 30)
    
    try:
        # Initialize corrector
        corrector = TextCorrector()
        
        # Print some training examples from the dataset
        print("\nTraining examples:")
        for i, (noisy, clean) in enumerate(corrector.dataset.pairs[:5]):
            print(f"Example {i+1}:")
            print(f"Input : {noisy}")
            print(f"Output: {clean}\n")
        
        print("Model loaded successfully! Ready to correct text.")
        print("Enter Ottoman text to correct (or 'q' to quit)")
        
        while True:
            print("\n> ", end='')
            text = input()
            
            if text.lower() == 'q':
                break
            
            corrected = corrector.correct_text(text)
            print("\nOriginal :", text)
            print("Corrected:", corrected)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()