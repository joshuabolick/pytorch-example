import unittest
import torch
import numpy as np
from data_fetcher import DataFetcher
from model import NewsClassifier

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.data_fetcher = DataFetcher()
        
    def test_performance_tracking(self):
        # Test initial performance
        initial_perf = self.data_fetcher.get_performance()
        self.assertEqual(initial_perf['accuracy'], 0.0)
        
        # Test updating performance
        accuracy = self.data_fetcher.update_performance(1, 1)  # Correct prediction
        self.assertEqual(accuracy, 1.0)
        
        accuracy = self.data_fetcher.update_performance(0, 1)  # Incorrect prediction
        self.assertEqual(accuracy, 0.5)
        
        # Test final performance
        final_perf = self.data_fetcher.get_performance()
        self.assertEqual(final_perf['accuracy'], 0.5)

    def test_get_training_data(self):
        headlines, targets = self.data_fetcher.get_training_data()
        
        # Check that we get a list of headlines and corresponding targets
        self.assertIsInstance(headlines, list)
        self.assertIsInstance(targets, np.ndarray)
        
        # If we got data, verify its structure
        if headlines:
            self.assertEqual(len(headlines), len(targets))
            self.assertEqual(targets.shape[1], 1)  # Each target should be a single value
            self.assertTrue(all(t in [0, 1] for t in targets.flatten()))  # Targets should be 0 or 1

class TestNewsClassifier(unittest.TestCase):
    def setUp(self):
        self.model = NewsClassifier(min_vocab_size=100)
        self.test_headlines = [
            "Breaking news from CNN about politics",
            "Fox News reports on latest developments",
            "CNN covers international events",
            "Fox News exclusive interview"
        ]
        
    def test_vocabulary_building(self):
        # Test vocabulary building
        self.model.build_vocabulary(self.test_headlines)
        
        # Check that vocabulary was built
        self.assertGreater(len(self.model.vocab), 0)
        
        # Check that special tokens exist
        self.assertIn('<PAD>', self.model.vocab)
        self.assertIn('<UNK>', self.model.vocab)
        
    def test_text_preprocessing(self):
        # Build vocabulary first
        self.model.build_vocabulary(self.test_headlines)
        
        # Test preprocessing
        processed = self.model.preprocess_text("Breaking news from CNN")
        
        # Check that processed text is a tensor
        self.assertIsInstance(processed, torch.Tensor)
        self.assertTrue(all(isinstance(x.item(), int) for x in processed))
        
    def test_model_forward_pass(self):
        # Build vocabulary and prepare test data
        self.model.build_vocabulary(self.test_headlines)
        processed_texts = [self.model.preprocess_text(h) for h in self.test_headlines]
        
        # Test forward pass
        output = self.model(processed_texts)
        
        # Check output shape and values
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], len(processed_texts))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))  # Outputs should be between 0 and 1
        
    def test_model_training_step(self):
        # Build vocabulary
        self.model.build_vocabulary(self.test_headlines)
        
        # Prepare test data
        processed_texts = [self.model.preprocess_text(h) for h in self.test_headlines]
        targets = torch.FloatTensor([[0], [1], [0], [1]])  # Alternating CNN/Fox targets
        
        # Set up loss and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Perform training step
        optimizer.zero_grad()
        output = self.model(processed_texts)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        # Check that loss is a scalar
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests() 