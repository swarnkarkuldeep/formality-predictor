# Formality Predictor

A machine learning model that classifies text as formal or informal and provides suggestions for making informal text more formal.

## Features

- Classifies text as formal or informal
- Provides suggestions for making informal text more formal
- Combines linguistic features with BERT embeddings for accurate predictions
- Supports both training and inference modes

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/swarnkarkuldeep/formality-predictor.git
   cd formality-predictor
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Usage

### Training the Model

1. Place your training data in the `data/` directory:
   - `training_labeled.csv` - For training
   - `eval_labeled.csv` - For evaluation (optional)

2. Run the training script:
   ```bash
   python train_model.py
   ```

### Making Predictions

Run the prediction script and enter text when prompted:
```bash
python predict.py
```

Example usage:
```
Enter text (or 'quit' to exit): hey, what's up?
Prediction: Informal

Enter text (or 'quit' to exit): Good afternoon. How may I assist you today?
Prediction: Formal
```

## Project Structure

- `data_loader.py` - Handles loading and preprocessing of training data
- `feature_engineer.py` - Extracts linguistic features and generates embeddings
- `train_model.py` - Trains and saves the formality prediction model
- `predict.py` - Provides a command-line interface for making predictions
- `models/` - Directory for storing trained models (created automatically)

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- spaCy
- scikit-learn
- NumPy
- pandas
- tqdm
- joblib
- TensorFlow (for some utility functions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
