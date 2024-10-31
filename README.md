# Ottoman Turkish NLP Project: Data Preparation & Preprocessing

This project focuses on building models for preprocessing, noise detection, and correction of Ottoman Turkish text in Latinized forms. The pipeline covers word extraction, dictionary creation, noise detection, and word correction using machine learning models.

---

## Table of Contents
- [Data Preparation & Preprocessing](#data-preparation--preprocessing)
  - [Extracting Words from Corpus](#extracting-words-from-corpus)
  - [Creating a Lexicon](#creating-a-lexicon)
- [Model Development](#model-development)
  - [Noise Detection Model](#noise-detection-model)
  - [Word Correction Model](#word-correction-model)
- [Training the Model](#training-the-model)
  - [Preparing Noisy Word Pairs](#preparing-noisy-word-pairs)
  - [Data Augmentation](#data-augmentation)
- [Correction Workflow](#correction-workflow)

---

## Data Preparation & Preprocessing

### Extracting Words from Corpus

1. **Text Extraction**:
   - Extract words from the Ottoman Turkish corpus.
   - Clean each word by removing non-letter characters to filter only valid words.
   - Store the cleaned words in a structured format (e.g., CSV or JSON) for easy access during model training and evaluation.

2. **Handling Special Characters**:
   - Identify and remove any non-alphabetic characters, keeping only the core letters.
   - Special attention may be needed for diacritics and characters specific to Ottoman Turkish or historical contexts.

3. **Storage of Valid Words**:
   - Save cleaned and valid words into a structured text file or database format to serve as a training source for the models.

### Creating a Lexicon

- Compile a dictionary or lexicon of valid Ottoman Turkish words:
  - This may involve using existing dictionary files or clean text corpora as a foundation.
  - The lexicon can help in verifying word validity and serve as a lookup table during both training and correction.

---

## Model Development
```
╭─────── Training Status ───────╮
│ Training Progress             │
│ ├─ Epoch: 1/10                │
│ ├─ Batch: 6656/161397         │
│ ├─ Samples: 53,248            │
│ └─ Speed: 1461.43 samples/sec │
│                               │
│ Performance Metrics           │
│ ├─ Current Loss: 0.9692       │
│ ├─ Best Loss: 0.0000          │
│ ├─ Accuracy: 70.70%           │
│ └─ Best Accuracy: 70.70%      │
│                               │
│ Resource Usage                │
│ ├─ CUDA Memory: 49.36 MB      │
│ ├─ Learning Rate: 0.001000    │
│ └─ Runtime: 03:00:36          │
╰───────────────────────────────╯
⠹  Epochs      ━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                       10%    •   1/10                •  -:--:--  
⠹  Batches     ━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                        4%    •    6660/161397        •  0:13:51  

╭────── Training Status ──────╮
│ Training Progress           │
│ ├─ Epoch: 7/10              │
│ ├─ Batch: 60756/161397      │
│ ├─ Samples: 486,048         │
│ └─ Speed: 91.97 samples/sec │
│                             │
│ Performance Metrics         │
│ ├─ Current Loss: 0.2339     │
│ ├─ Best Loss: 0.0000        │
│ ├─ Accuracy: 93.11%         │
│ └─ Best Accuracy: 100.00%   │
│                             │
│ Resource Usage              │
│ ├─ CUDA Memory: 49.36 MB    │
│ ├─ Learning Rate: 0.001000  │
│ └─ Runtime: 04:28:04        │
│                             │
│                             │
╰─────────────────────────────╯
⠴  Epochs      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━                       70%    •   7/10                •  -:--:--  
⠴  Batches     ━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━                       38%    •   60760/161397        •  0:08:42  
```
### Noise Detection Model

1. **Goal**:
   - The purpose of this model is to detect "noisy" or "broken" words (e.g., "o n lar ın" instead of "onların").

2. **Model Type**:
   - Use a character-based embedding approach to represent words as sequences of characters.
   - Develop a Binary Classifier (such as a simple neural network) to label words as either "clean" or "noisy".

3. **Implementation**:
   - Embed each word at the character level to capture fine-grained differences.
   - Train the classifier to predict whether a given word is clean or noisy based on its structure.
   
4. **Benefits**:
   - By filtering out noisy words early, the model ensures that only valid data is passed to the word correction stage.

### Word Correction Model

1. **Goal**:
   - This model corrects corrupted or misspelled words, which are common in OCR and transliteration errors.

2. **Model Architecture**:
   - Consider using a sequence-to-sequence (Seq2Seq) model or Transformer model (like BERT with a custom head).
   - Alternatively, a Recurrent Neural Network (RNN) with attention mechanisms can also handle this task effectively.

3. **Training Data**:
   - Train the model on a set of noisy words (OCR errors) paired with their correct forms.
   - Create a dataset of corrupted-to-corrected word pairs for supervised training.

4. **Comparison with Spell Correction**:
   - Similar to spell correction models, this approach allows for mapping corrupted words back to their valid forms based on learned patterns in character sequences.

---

## Training the Model

### Preparing Noisy Word Pairs

1. **Collect Noisy Data**:
   - Gather pairs of noisy words and their corresponding correct forms from OCR-transliterated texts or manually annotated datasets.
   
2. **Synthetic Noise Generation**:
   - Introduce synthetic noise to expand the training dataset.
   - Generate common OCR error types observed in Ottoman Turkish texts to enhance model performance.

### Data Augmentation

1. **Augmentation Techniques**:
   - Apply transformations such as character insertions, deletions, or swaps that mimic real OCR errors.
   - Augmentation allows for a more robust model capable of handling various noise patterns.

2. **Dataset Expansion**:
   - Increase dataset size by systematically applying noise to correct words, providing more examples for the model to learn from.

---

## Correction Workflow

1. **Prediction**:
   - For each input word, the noise detection model first predicts whether it is clean or noisy.
   
2. **Correction Model**:
   - If the word is labeled as "noisy," pass it through the correction model.
   - The correction model outputs the most likely corrected form of the word based on its learned patterns.

3. **Output Validation**:
   - Check the corrected output against the lexicon to ensure that the correction is a valid Ottoman Turkish word.
   - This post-processing step improves accuracy by only accepting lexicon-validated corrections.

4. **Final Output**:
   - The model’s final output is a clean and corrected corpus, ready for downstream tasks or analysis.

---

## Conclusion

This pipeline ensures a systematic approach to extracting, cleaning, and correcting Ottoman Turkish text data. By leveraging models for noise detection and correction, we improve the quality and accuracy of text data, facilitating further NLP applications.

