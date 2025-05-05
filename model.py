import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class NewsClassifier(nn.Module):
    def __init__(self, embedding_dim=100, hidden_dim=64, min_vocab_size=1000):
        super(NewsClassifier, self).__init__()
        
        # Initialize vocabulary
        self.vocab = {'<PAD>': 0, '<UNK>': 1}  # Add special tokens
        self.vocab_size = 2
        self.stop_words = set(stopwords.words('english'))
        self.min_vocab_size = min_vocab_size
        
        # Word embedding layer (will be resized after vocabulary building)
        self.embedding = nn.Embedding(2, embedding_dim)  # Start with minimal size
        
        # LSTM for processing word sequences
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def build_vocabulary(self, headlines):
        """Build vocabulary from headlines."""
        print("\nBuilding vocabulary...")
        print(f"Processing {len(headlines)} headlines")
        
        # First pass: count word frequencies
        word_counts = {}
        total_words = 0
        unique_words = 0
        
        for i, headline in enumerate(headlines):
            # Clean the text: remove special characters and punctuation, handle apostrophes
            cleaned_text = headline.lower()
            # Replace common contractions
            cleaned_text = cleaned_text.replace("'s", " is")
            cleaned_text = cleaned_text.replace("'re", " are")
            cleaned_text = cleaned_text.replace("'ve", " have")
            cleaned_text = cleaned_text.replace("'ll", " will")
            cleaned_text = cleaned_text.replace("'d", " would")
            cleaned_text = cleaned_text.replace("n't", " not")
            # Remove remaining special characters
            cleaned_text = ''.join(c for c in cleaned_text if c.isalnum() or c.isspace())
            
            words = word_tokenize(cleaned_text)
            
            for word in words:
                if word not in self.stop_words and len(word) > 1:  # Also filter out single characters
                    word_counts[word] = word_counts.get(word, 0) + 1
                    total_words += 1
            
            if (i + 1) % 10 == 0:  # Print progress every 10 headlines
                print(f"Processed {i + 1}/{len(headlines)} headlines")
        
        unique_words = len(word_counts)
        print(f"\nFound {total_words} total words (excluding stop words and single characters)")
        print(f"Found {unique_words} unique words")
        
        # Sort words by frequency and take top N
        print(f"\nSelecting top {self.min_vocab_size} most frequent words...")
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add words to vocabulary
        for word, count in sorted_words[:self.min_vocab_size]:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Print some statistics about the vocabulary
        print(f"\nVocabulary statistics:")
        print(f"Total vocabulary size: {len(self.vocab)} words")
        print(f"Most frequent words:")
        for word, count in sorted_words[:10]:  # Show top 10 words
            print(f"  '{word}': {count} occurrences")
        
        # Update embedding layer size
        self.embedding = nn.Embedding(len(self.vocab), self.embedding.embedding_dim)
        print("\nVocabulary building complete!")
        
    def preprocess_text(self, text):
        """Convert text to sequence of word indices."""
        # Clean the text: remove special characters and punctuation, handle apostrophes
        cleaned_text = text.lower()
        # Replace common contractions
        cleaned_text = cleaned_text.replace("'s", " is")
        cleaned_text = cleaned_text.replace("'re", " are")
        cleaned_text = cleaned_text.replace("'ve", " have")
        cleaned_text = cleaned_text.replace("'ll", " will")
        cleaned_text = cleaned_text.replace("'d", " would")
        cleaned_text = cleaned_text.replace("n't", " not")
        # Remove remaining special characters
        cleaned_text = ''.join(c for c in cleaned_text if c.isalnum() or c.isspace())
        
        words = word_tokenize(cleaned_text)
        
        indices = []
        for word in words:
            if word not in self.stop_words and len(word) > 1:  # Also filter out single characters
                indices.append(self.vocab.get(word, 1))  # 1 is <UNK> token
        if not indices:  # Handle empty sequences
            indices = [0]  # Use <PAD> token
        return torch.LongTensor(indices)
    
    def forward(self, x):
        # x is a list of headlines
        if not x:  # Handle empty batch
            return torch.zeros(1, 1)
            
        batch_size = len(x)
        max_len = max(len(headline) for headline in x)
        
        # Create padded tensor
        padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, headline in enumerate(x):
            padded[i, :len(headline)] = headline
        
        # Get embeddings
        embedded = self.embedding(padded)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Get the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

    def predict(self, x: torch.Tensor) -> int:
        """Make a binary prediction (0 for CNN, 1 for Fox)."""
        with torch.no_grad():
            output = self(x)
            return 1 if output.item() > 0.5 else 0

    def update(self, features: torch.Tensor, target: torch.Tensor, learning_rate: float = 0.001):
        """Update the model with new data."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for _ in range(10):  # Number of epochs
            optimizer.zero_grad()
            output = self(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        return loss.item()

    def update_and_evaluate(self, features: torch.Tensor, targets: torch.Tensor, learning_rate: float = 0.001):
        """Update the model with new data and evaluate performance."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        
        # Training loop
        for _ in range(10):  # Number of epochs
            optimizer.zero_grad()
            output = self(features)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        with torch.no_grad():
            predictions = (self(features) > 0.5).float()
            accuracy = accuracy_score(targets.numpy(), predictions.numpy())
            precision = precision_score(targets.numpy(), predictions.numpy())
            recall = recall_score(targets.numpy(), predictions.numpy())
            
            print(f"Training Metrics:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}") 