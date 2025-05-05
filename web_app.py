from flask import Flask, render_template, jsonify
from data_fetcher import DataFetcher
from model import NewsClassifier
import torch
from difflib import SequenceMatcher
import numpy as np
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Global variables
data_fetcher = DataFetcher()
model = NewsClassifier(min_vocab_size=2000)
headlines_cache = {
    'cnn': [],
    'fox': [],
    'last_update': None
}
UPDATE_INTERVAL = 60  # 1 minute to match the existing functionality

def similarity_score(headline1, headline2):
    """Calculate similarity between two headlines using SequenceMatcher."""
    return SequenceMatcher(None, headline1.lower(), headline2.lower()).ratio()

def find_matching_headlines(cnn_headlines, fox_headlines, threshold=0.3):
    """Find matching headlines between CNN and Fox News."""
    matches = []
    used_cnn = set()
    used_fox = set()
    
    # Sort by similarity score
    all_pairs = []
    for cnn_h in cnn_headlines:
        for fox_h in fox_headlines:
            score = similarity_score(cnn_h, fox_h)
            if score >= threshold:
                all_pairs.append((score, cnn_h, fox_h))
    
    # Sort by similarity score in descending order
    all_pairs.sort(reverse=True)
    
    # Match headlines greedily
    for score, cnn_h, fox_h in all_pairs:
        if cnn_h not in used_cnn and fox_h not in used_fox:
            matches.append({
                'similarity': score,
                'cnn': cnn_h,
                'fox': fox_h
            })
            used_cnn.add(cnn_h)
            used_fox.add(fox_h)
    
    return matches

def update_headlines():
    """Background task to update headlines periodically."""
    while True:
        try:
            # Fetch new headlines using the existing data fetcher
            headlines = data_fetcher.fetch_headlines(skip_wait=True)
            
            # Separate CNN and Fox headlines
            cnn_headlines = [h[0] for h in headlines if h[1] == 0]
            fox_headlines = [h[0] for h in headlines if h[1] == 1]
            
            # Update cache
            headlines_cache['cnn'] = cnn_headlines
            headlines_cache['fox'] = fox_headlines
            headlines_cache['last_update'] = datetime.now()
            
            print(f"Updated headlines cache at {headlines_cache['last_update']}")
            print(f"Found {len(cnn_headlines)} CNN headlines and {len(fox_headlines)} Fox News headlines")
            
        except Exception as e:
            print(f"Error updating headlines: {e}")
        
        time.sleep(UPDATE_INTERVAL)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/headlines')
def get_headlines():
    """API endpoint to get matching headlines."""
    if not headlines_cache['last_update']:
        return jsonify({'error': 'No headlines available yet'}), 503
    
    matches = find_matching_headlines(headlines_cache['cnn'], headlines_cache['fox'])
    
    return jsonify({
        'matches': matches,
        'last_update': headlines_cache['last_update'].strftime('%Y-%m-%d %H:%M:%S'),
        'total_cnn': len(headlines_cache['cnn']),
        'total_fox': len(headlines_cache['fox'])
    })

if __name__ == '__main__':
    # Start background thread for updating headlines
    update_thread = threading.Thread(target=update_headlines, daemon=True)
    update_thread.start()
    
    # Run Flask app
    app.run(debug=True, use_reloader=False) 