from pathlib import Path
from .data.preprocessor import TextPreprocessor
from .data.noise_generator import NoiseGenerator

def test_preprocessing():
    # Sample words
    test_words = [
        "Mısır'da",  # Valid with apostrophe
        "Şa'ban",    # Valid with apostrophe
        "Şafi'i",    # Valid with apostrophe
        "ab",        # Too short
        "constantinopolisten",  # Too long
        "normal",    # Valid without apostrophe
        "test''test",  # Invalid consecutive apostrophes
        "test'",     # Invalid ending apostrophe
        "'test",     # Invalid starting apostrophe
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test word validation
    for word in test_words:
        valid = preprocessor.is_valid_word(word)
        print(f"Word: {word:20} Valid: {valid}")
    
    # Test noise generation
    noise_gen = NoiseGenerator()
    valid_words = [w for w in test_words if preprocessor.is_valid_word(w)]
    
    print("\nNoise generation examples:")
    for word in valid_words:
        noisy = noise_gen.generate_noisy_sample(word)
        print(f"Original: {word:20} Noisy: {noisy}")

if __name__ == "__main__":
    test_preprocessing() 