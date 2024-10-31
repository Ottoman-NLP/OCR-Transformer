import re

def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-ZğüşöçİĞÜŞÖÇ]', ' ', text)
    cleaned_words = [word for word in cleaned_text.split() if len(word) > 2]
    return cleaned_words

with open('raw_corpus.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

cleaned_words = preprocess_text(raw_text)

with open('cleaned_words.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(cleaned_words))

import random

def generate_noisy_word(word):
    if len(word) < 5:
        return word
    noisy_word = list(word)
    if random.choice([True, False]):
        # Randomly add spaces
        insert_idx = random.randint(1, len(noisy_word) - 1)
        noisy_word.insert(insert_idx, ' ')
    else:
        # Randomly remove a character
        remove_idx = random.randint(0, len(noisy_word) - 1)
        del noisy_word[remove_idx]
    return ''.join(noisy_word)

# Generate a dataset of clean and noisy word pairs
clean_noisy_pairs = [(word, generate_noisy_word(word)) for word in cleaned_words]

# Save pairs to a file
with open('noisy_clean_pairs.txt', 'w', encoding='utf-8') as file:
    for clean, noisy in clean_noisy_pairs:
        file.write(f"{noisy}\t{clean}\n")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define Dataset class
class NoisyWordDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                noisy, clean = line.strip().split('\t')
                self.data.append((noisy, clean))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define an LSTM-based sequence-to-sequence model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2SeqLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        _, (hidden, cell) = self.encoder(embed)
        outputs, _ = self.decoder(embed, (hidden, cell))
        outputs = self.fc(outputs)
        return outputs

# Hyperparameters and Training Loop
VOCAB_SIZE = 100  # To be replaced with actual vocab size
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

dataset = NoisyWordDataset('noisy_clean_pairs.txt')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Seq2SeqLSTM(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    for batch in dataloader:
        noisy_words, clean_words = batch
        noisy_words_tensor = torch.LongTensor([[ord(char) for char in word] for word in noisy_words])
        clean_words_tensor = torch.LongTensor([[ord(char) for char in word] for word in clean_words])

        optimizer.zero_grad()
        output = model(noisy_words_tensor)
        loss = criterion(output.view(-1, VOCAB_SIZE), clean_words_tensor.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
import pymupdf4llm

def extract_and_correct_text(pdf_path, model):
    with pymupdf4llm.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            cleaned_words = preprocess_text(text)
            corrected_words = [correct_word(word, model) for word in cleaned_words]
            full_text += " ".join(corrected_words) + "\n"

    return full_text

def correct_word(word, model):
    word_tensor = torch.LongTensor([ord(char) for char in word]).unsqueeze(0)
    with torch.no_grad():
        output = model(word_tensor)
    corrected_chars = [chr(torch.argmax(char_probs).item()) for char_probs in output.squeeze()]
    return "".join(corrected_chars)

corrected_text = extract_and_correct_text('sample.pdf', model)

with open('corrected_output.txt', 'w', encoding='utf-8') as file:
    file.write(corrected_text)
