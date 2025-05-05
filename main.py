import torch
import numpy as np
from data_fetcher import DataFetcher
from model import NewsClassifier
import time
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify
from threading import Thread
from datetime import datetime
from difflib import SequenceMatcher
import torch.nn.functional as F
import threading

# Initialize Flask app
app = Flask(__name__)

# Global variables
data_fetcher = DataFetcher()
model = NewsClassifier(min_vocab_size=2000)
headlines_cache = {
    'cnn': [],  # List of (headline, url) tuples
    'fox': [],  # List of (headline, url) tuples
    'last_update': None
}
loss_history = []  # Initialize loss history list
iteration = 0  # Initialize iteration counter
headlines = []  # Initialize headlines list
predicted_source = ""  # Initialize predicted source
source_name = ""  # Initialize source name
correct_predictions = 0
total_predictions = 0

def similarity_score(headline1, headline2):
    """Calculate similarity between two headlines using multiple metrics."""
    # Convert to lowercase and split into words
    words1 = set(headline1.lower().split())
    words2 = set(headline2.lower().split())
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
    words1 = words1 - common_words
    words2 = words2 - common_words
    
    # Calculate word overlap
    if not words1 or not words2:
        return 0.0
    
    overlap = words1.intersection(words2)
    total_words = words1.union(words2)
    
    # Calculate base similarity
    word_similarity = len(overlap) / len(total_words)
    
    # Calculate sequence similarity for context
    sequence_similarity = SequenceMatcher(None, headline1.lower(), headline2.lower()).ratio()
    
    # Combine both metrics (weighted average)
    return 0.7 * word_similarity + 0.3 * sequence_similarity

def find_matching_headlines(cnn_headlines, fox_headlines, threshold=0.3):  # Lowered threshold since we improved matching
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

# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/headlines')
def get_headlines():
    """API endpoint to get matching headlines and training statistics."""
    if not headlines_cache['last_update']:
        return jsonify({'error': 'No headlines available yet'}), 503
    
    # Extract just the headlines for matching
    cnn_headlines = [h[0] for h in headlines_cache['cnn']]
    fox_headlines = [h[0] for h in headlines_cache['fox']]
    matches = find_matching_headlines(cnn_headlines, fox_headlines)
    
    # Calculate current accuracy
    current_accuracy = 0.0
    if total_predictions > 0:
        current_accuracy = (correct_predictions / total_predictions) * 100
    
    # Get current loss
    current_loss = loss_history[-1] if loss_history else 0.0
    
    # Calculate accuracy history - use the same calculation method as current accuracy
    accuracy_history = []
    if loss_history:
        # Use a moving window of the last 100 predictions
        window_size = min(100, len(loss_history))
        if window_size > 0:
            # Calculate accuracy for each point in the window
            for i in range(window_size):
                idx = len(loss_history) - window_size + i
                if idx >= 0:
                    # Use the same accuracy calculation as current accuracy
                    window_accuracy = current_accuracy  # Use current accuracy for all points
                    accuracy_history.append(window_accuracy)
    
    return jsonify({
        'matches': matches,
        'cnn_headlines': [{'text': h[0], 'url': h[1]} for h in headlines_cache['cnn']],
        'fox_headlines': [{'text': h[0], 'url': h[1]} for h in headlines_cache['fox']],
        'last_update': headlines_cache['last_update'].strftime('%Y-%m-%d %H:%M:%S'),
        'total_cnn': len(headlines_cache['cnn']),
        'total_fox': len(headlines_cache['fox']),
        # Training statistics
        'accuracy': float(current_accuracy),
        'total_samples': int(total_predictions),
        'current_loss': float(current_loss),
        'loss_history': [float(loss) for loss in loss_history[-100:]],  # Keep last 100 losses
        'accuracy_history': [float(acc) for acc in accuracy_history],  # Convert all accuracies to float
        'iteration': int(iteration),
        'current_headline': headlines[0] if headlines else "No current headline",
        'prediction_result': f"Predicted: {predicted_source}, Actual: {source_name}" if headlines else "No prediction"
    })

def run_flask():
    """Run the Flask web server."""
    app.run(debug=False, use_reloader=False)

def train_model():
    """Main training loop."""
    global iteration, headlines, predicted_source, source_name, loss_history, correct_predictions, total_predictions
    
    print("\nStarting news source classifier training...")
    print("The model will learn to distinguish between CNN and Fox News headlines")
    print("Press Ctrl+C to stop\n")
    print("Web interface available at http://localhost:5000")
    
    # Build vocabulary first
    print("Building vocabulary...")
    initial_headlines = []
    for i in range(5):  # Fetch multiple batches to build better vocabulary
        print(f"\nFetching batch {i + 1}/5 for vocabulary building...")
        headlines = data_fetcher.fetch_headlines(skip_wait=True)  # Skip wait for initial fetches
        
        # Update headlines cache immediately
        cnn_headlines = [(h[0], h[1]) for h in headlines if h[2] == 0]
        fox_headlines = [(h[0], h[1]) for h in headlines if h[2] == 1]
        headlines_cache['cnn'] = cnn_headlines
        headlines_cache['fox'] = fox_headlines
        headlines_cache['last_update'] = datetime.now()
        
        print(f"Updated headlines cache:")
        print(f"CNN headlines: {len(cnn_headlines)}")
        print(f"Fox News headlines: {len(fox_headlines)}")
        
        initial_headlines.extend([h[0] for h in headlines])
        if i < 4:  # Don't sleep after the last fetch
            time.sleep(1)  # Small delay between fetches
    
    model.build_vocabulary(initial_headlines)
    print("\nVocabulary built. Starting training...")
    
    # Initialize tracking variables
    correct_predictions = 0
    total_predictions = 0
    
    while True:
        try:
            # Get training data
            headlines, labels = data_fetcher.get_training_data()
            if not headlines:
                print("No new headlines available. Waiting...")
                time.sleep(60)
                continue
            
            # Update headlines cache for web interface
            all_headlines = data_fetcher.fetch_headlines(skip_wait=True)  # Get all headlines
            cnn_headlines = [(h[0], h[1]) for h in all_headlines if h[2] == 0]
            fox_headlines = [(h[0], h[1]) for h in all_headlines if h[2] == 1]
            
            # Update cache with all headlines
            headlines_cache['cnn'] = cnn_headlines
            headlines_cache['fox'] = fox_headlines
            headlines_cache['last_update'] = datetime.now()
            
            print(f"\nUpdated headlines cache:")
            print(f"CNN headlines: {len(cnn_headlines)}")
            print(f"Fox News headlines: {len(fox_headlines)}")
            
            # Train on each headline
            for headline, label in zip(headlines, labels):
                # Convert headline to tensor
                headline_tensor = model.preprocess_text(headline)
                
                # Forward pass
                output = model([headline_tensor])
                
                # Convert label to tensor with correct shape and type
                label_value = int(label.item() if hasattr(label, 'item') else label)
                label_tensor = torch.tensor([label_value], dtype=torch.long)
                
                # Calculate loss
                loss = F.cross_entropy(output, label_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track loss
                loss_value = float(loss.item())
                loss_history.append(loss_value)
                
                # Get prediction and update accuracy
                predicted = torch.argmax(output).item()
                is_correct = (predicted == label_value)
                correct_predictions += int(is_correct)
                total_predictions += 1
                current_accuracy = (correct_predictions / total_predictions) * 100
                
                predicted_source = "CNN" if predicted == 0 else "Fox News"
                source_name = "CNN" if label_value == 0 else "Fox News"
                
                # Print progress
                print(f"\nIteration {iteration}")
                print(f"Training on headline: {headline}")
                print(f"Actual source: {source_name}")
                print(f"Predicted source: {predicted_source}")
                print(f"Current accuracy: {current_accuracy:.2f}%")
                print(f"Loss: {loss_value:.4f}")
                
                # Save checkpoint
                if iteration % 1 == 0:  # Save every iteration
                    torch.save(model.state_dict(), 'news_classifier.pth')
                    print(f"\nSaved checkpoint at iteration {iteration}")
                
                iteration += 1
                
                # Sleep to allow web interface to update
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"Error during training: {e}")
            continue

if __name__ == "__main__":
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()
    
    # Start Flask app
    app.run(debug=False, use_reloader=False) 