import feedparser
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import random

class DataFetcher:
    def __init__(self):
        self.cnn_feed = "http://rss.cnn.com/rss/cnn_topstories.rss"
        self.fox_feed = "http://feeds.foxnews.com/foxnews/latest"
        self.last_fetch_time = 0
        self.fetch_interval = 60  # seconds between fetches
        self.performance_history = {
            'total': 0,
            'correct': 0,
            'accuracy': []
        }

    def fetch_headlines(self, skip_wait=False) -> List[Tuple[str, str, int]]:
        """Fetch headlines from both RSS feeds."""
        current_time = time.time()
        if not skip_wait and current_time - self.last_fetch_time < self.fetch_interval:
            time.sleep(self.fetch_interval - (current_time - self.last_fetch_time))
        
        headlines = []
        try:
            # Fetch CNN headlines
            print("Fetching CNN headlines...")
            cnn_news = feedparser.parse(self.cnn_feed)
            for entry in cnn_news.entries:
                headlines.append((entry.title, entry.link, 0))  # 0 for CNN
            
            # Fetch Fox News headlines
            print("Fetching Fox News headlines...")
            fox_news = feedparser.parse(self.fox_feed)
            for entry in fox_news.entries:
                headlines.append((entry.title, entry.link, 1))  # 1 for Fox
            
            self.last_fetch_time = time.time()
            print(f"Successfully fetched {len(headlines)} headlines")
        except Exception as e:
            print(f"Error fetching headlines: {e}")
        
        return headlines

    def get_training_data(self) -> Tuple[List[str], np.ndarray]:
        """Get a random headline and its source for training."""
        headlines = self.fetch_headlines()
        if not headlines:
            return [], np.array([])
        
        # Pick a random headline
        headline, url, source = random.choice(headlines)
        
        return [headline], np.array([[source]])

    def update_performance(self, prediction: int, actual: int):
        """Update performance tracking."""
        self.performance_history['total'] += 1
        if prediction == actual:
            self.performance_history['correct'] += 1
        
        accuracy = self.performance_history['correct'] / self.performance_history['total']
        self.performance_history['accuracy'].append(accuracy)
        
        return accuracy

    def get_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if self.performance_history['total'] == 0:
            return {'accuracy': 0.0}
        
        return {
            'accuracy': self.performance_history['correct'] / self.performance_history['total'],
            'total_samples': self.performance_history['total']
        }