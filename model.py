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
    def __init__(self, min_vocab_size=2000):
        super(NewsClassifier, self).__init__()
        self.vocab = {}
        self.min_vocab_size = min_vocab_size
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize embedding layer with correct size
        self.embedding = nn.Embedding(min_vocab_size + 1, self.embedding_dim)  # +1 for unknown token
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 2)  # 2 classes: CNN and Fox News
        
    def build_vocabulary(self, headlines):
        """Build vocabulary from headlines."""
        print("\nBuilding vocabulary...")
        word_counts = {}
        
        # Count word frequencies
        for headline in headlines:
            words = self.preprocess_text(headline, return_words=True)
            for word in words:
                if word not in self.stop_words:  # Only count non-stop words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create vocabulary
        self.vocab = {'<UNK>': 0}  # Add unknown token
        for word, count in sorted_words:
            if len(self.vocab) < self.min_vocab_size:
                self.vocab[word] = len(self.vocab)
            else:
                break
        
        print(f"\nFound {sum(word_counts.values())} total words (excluding stop words and single characters)")
        print(f"Found {len(word_counts)} unique words")
        print(f"\nSelecting top {self.min_vocab_size} most frequent words...")
        
        # Print vocabulary statistics
        print("\nVocabulary statistics:")
        print(f"Total vocabulary size: {len(self.vocab)} words")
        print("Most frequent words:")
        for word, count in sorted_words[:10]:
            print(f"  '{word}': {count} occurrences")
        
        print("\nVocabulary building complete!")
        
    def preprocess_text(self, text, return_words=False):
        """Convert text to tensor or return preprocessed words."""
        # Tokenize and clean text
        words = text.lower().split()
        # Remove stop words and single characters
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        if return_words:
            return words
        
        # Convert words to indices
        indices = [self.vocab.get(word, 0) for word in words]  # 0 is <UNK>
        
        # Pad or truncate to fixed length
        max_len = 50  # Maximum sequence length
        if len(indices) < max_len:
            indices.extend([0] * (max_len - len(indices)))
        else:
            indices = indices[:max_len]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, x):
        """Forward pass through the network."""
        # x is a list of tensors, get the first one
        x = x[0]
        
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Get embeddings
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)
        
        # Get the last output
        last_output = lstm_out[:, -1, :]
        
        # Final classification layer
        output = self.fc(last_output)
        
        return output

    def predict(self, x: torch.Tensor) -> int:
        """Make a binary prediction (0 for CNN, 1 for Fox)."""
        with torch.no_grad():
            output = self(x)
            return torch.argmax(output).item()  # Return the class with highest probability

    def update(self, features: torch.Tensor, target: torch.Tensor, learning_rate: float = 0.001):
        """Update the model with new data."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for multi-class
        
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
        criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for multi-class
        
        # Training loop
        for _ in range(10):  # Number of epochs
            optimizer.zero_grad()
            output = self(features)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        with torch.no_grad():
            predictions = torch.argmax(self(features), dim=1)
            accuracy = (predictions == targets).float().mean().item()
            
            print(f"Training Metrics:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy
            } 