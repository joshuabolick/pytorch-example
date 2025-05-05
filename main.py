import torch
import numpy as np
from data_fetcher import DataFetcher
from model import NewsClassifier
import time
import matplotlib.pyplot as plt

def main():
    # Initialize components
    data_fetcher = DataFetcher()
    model = NewsClassifier(min_vocab_size=2000)  # Increased minimum vocabulary size
    
    # Initialize loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting news source classifier training...")
    print("The model will learn to distinguish between CNN and Fox News headlines")
    print("Press Ctrl+C to stop")
    
    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy_line, = ax.plot([], [], 'b-', label='Accuracy')
    loss_line, = ax.plot([], [], 'r--', label='Loss (normalized)')
    ax.set_title('Model Performance Over Time')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Value')
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1 (0% to 100%)
    ax.grid(True)  # Add grid for better readability
    ax.legend()  # Add legend
    
    # Plot update interval (in iterations)
    plot_update_interval = 5  # Reduced interval for more frequent updates
    
    # Initialize loss history
    loss_history = []
    
    iteration = 0
    try:
        # First, build vocabulary from initial headlines
        print("Building vocabulary...")
        initial_headlines = []
        for i in range(5):  # Fetch multiple batches to build better vocabulary
            print(f"\nFetching batch {i + 1}/5 for vocabulary building...")
            headlines = data_fetcher.fetch_headlines(skip_wait=True)  # Skip wait for initial fetches
            initial_headlines.extend([h[0] for h in headlines])
            if i < 4:  # Don't sleep after the last fetch
                time.sleep(1)  # Small delay between fetches
        
        model.build_vocabulary(initial_headlines)
        print("Vocabulary built. Starting training...")
        
        while True:
            # Get new data
            headlines, target = data_fetcher.get_training_data()
            if not headlines:
                print("No new headlines available. Waiting...")
                time.sleep(60)
                continue

            # Preprocess headlines
            processed_headlines = [model.preprocess_text(h) for h in headlines]
            
            # Convert target to tensor
            target_tensor = torch.FloatTensor(target)

            # Forward pass
            output = model(processed_headlines)
            prediction = (output > 0.5).float()
            
            # Print prediction details
            source_name = "CNN" if target[0][0] == 0 else "Fox News"
            predicted_source = "CNN" if prediction.item() == 0 else "Fox News"
            print(f"\nTraining on headline: {headlines[0]}")
            print(f"Actual source: {source_name}")
            print(f"Predicted source: {predicted_source}")

            # Calculate loss
            loss = criterion(output, target_tensor)
            loss_history.append(loss.item())

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update performance tracking
            accuracy = data_fetcher.update_performance(int(prediction.item()), int(target[0][0]))
            print(f"Current accuracy: {accuracy:.2%}")
            print(f"Loss: {loss.item():.4f}")

            # Update the plot periodically
            if iteration % plot_update_interval == 0:
                x_data = list(range(1, len(data_fetcher.performance_history['accuracy']) + 1))
                y_data = data_fetcher.performance_history['accuracy']
                
                # Update accuracy line
                accuracy_line.set_data(x_data, y_data)
                
                # Update loss line if we have loss data
                if loss_history:
                    max_loss = max(loss_history)
                    normalized_loss = [l/max_loss for l in loss_history]
                    loss_line.set_data(x_data, normalized_loss)
                
                # Adjust axes limits
                ax.relim()
                ax.autoscale_view()
                
                # Redraw the plot
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                # Small pause to allow the plot to update
                plt.pause(0.01)

            # Save checkpoint every 100 iterations
            if iteration % 100 == 0:
                torch.save(model.state_dict(), f'checkpoint_{iteration}.pth')
                print(f"Saved checkpoint at iteration {iteration}")

            iteration += 1
            time.sleep(60)  # Wait for 1 minute before next iteration

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final model
        torch.save(model.state_dict(), 'news_classifier.pth')
        print("Model saved to 'news_classifier.pth'")
        
        # Save the final accuracy plot
        plt.savefig('accuracy_plot.png')
        print("Accuracy plot saved to 'accuracy_plot.png'")
        
        # Close the plot window
        plt.close()

if __name__ == "__main__":
    main() 