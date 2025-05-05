# PyTorch Continuous Learning Example

This project demonstrates a simple PyTorch application that continuously learns from streaming data. It showcases how to:
- Create a neural network using PyTorch
- Fetch real-time data from an API
- Preprocess and feed data to the model
- Continuously update the model with new data
- Save model checkpoints

## Project Structure
- `main.py`: Main application entry point
- `model.py`: Neural network model definition
- `data_fetcher.py`: Handles data fetching and preprocessing
- `requirements.txt`: Project dependencies

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application
To start the continuous learning process:
```bash
python main.py
```

The application will:
- Fetch data from a public API every minute
- Preprocess the data
- Update the neural network model
- Save checkpoints periodically
- Print progress information

Press Ctrl+C to stop the process. The final model state will be saved to 'final_model.pth'.

## Customization
You can modify the following aspects:
- Data source: Edit `data_fetcher.py` to use a different API
- Model architecture: Modify `model.py` to change the neural network structure
- Training parameters: Adjust learning rates and other hyperparameters in `model.py`
- Fetch interval: Change the sleep duration in `main.py`

## Notes
- The current implementation uses a simple example API. You can replace it with your own data source.
- The model architecture is basic and can be enhanced based on your specific needs.
- Error handling and logging can be improved for production use.
