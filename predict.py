import os
import numpy as np
import joblib
from pathlib import Path
import torch
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from train_model import train_and_save_model
from data_loader import load_labeled_data
from feature_engineer import extract_linguistic_features
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')  # Suppress other warnings

MODEL_NAME = "distilbert-base-uncased"  # Aligned with training model
MODEL_PATH = "models/formality_model.pkl"
EMBEDDING_CACHE_PATH = "models/embedding_cache.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FormalityPredictor:
    """
    A classifier for determining formality level of English text.
    
    Combines linguistic features with BERT embeddings to predict whether text is formal or informal.
    Provides suggestions for making informal text more formal.
    """
    def __init__(self):
        """Initialize the predictor by loading tokenizer and models."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._load_models()



    def _load_models(self):
        """
        Load or train the formality classifier and embedding model.
        Creates models directory if needed, trains new model if none exists.
        """
        os.makedirs("models", exist_ok=True)
        if not Path(MODEL_PATH).exists():
            print("Training model (first-time setup)...")
            train_data, _ = load_labeled_data()
            self.classifier, self.embedding_model = train_and_save_model(train_data)
            self._save_models()
        else:
            print("Loading pre-trained models...")
            self._load_saved_models()

    def _save_models(self):
        """
        Save the trained classifier and embedding model state to disk.
        Uses joblib for serialization with proper model components.
        """
        joblib.dump({
            'classifier': self.classifier,
            'embedding_model_state': self.embedding_model.state_dict(),
            'model_name': MODEL_NAME
        }, MODEL_PATH)

    def _load_saved_models(self):
        """
        Load pre-trained models from disk.
        Handles both the classifier and embedding model state restoration.
        """
        models = joblib.load(MODEL_PATH)
        self.classifier = models['classifier']
        if 'embedding_model_state' in models:
            self.embedding_model = AutoModel.from_pretrained(models.get('model_name', MODEL_NAME))
            self.embedding_model.load_state_dict(models['embedding_model_state'])
        else:
            self.embedding_model = AutoModel.from_pretrained(MODEL_NAME)
        self.embedding_model.to(DEVICE)

    def _get_cached_embedding(self, text):
        """
        Get BERT embedding for text, using cache to avoid recomputation.
        
        Args:
            text: Input string to embed
            
        Returns:
            numpy array: Text embedding vector
        """
        text_hash = str(hash(text))
        if Path(EMBEDDING_CACHE_PATH).exists():
            cache = joblib.load(EMBEDDING_CACHE_PATH)
        else:
            cache = {}

        if text_hash not in cache:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            cache[text_hash] = outputs.last_hidden_state.mean(1).cpu().numpy()
            joblib.dump(cache, EMBEDDING_CACHE_PATH)

        return cache[text_hash]





    def predict(self, text):
        """
        Predict formality of input text with confidence score.
        
        The prediction combines:
        - Linguistic features (word length, contractions, punctuation)
        - Semantic embeddings from BERT
        - Presence of known informal words/phrases
        
        Args:
            text: Input string or list of strings to analyze
            
        Returns:
            dict: {
                'label': 'FORMAL'/'INFORMAL'/'UNDEFINED',
                'confidence': Prediction confidence percentage (0-100)
            }
        
        Raises:
            Exception: If feature extraction or model prediction fails
        """
        try:
            if isinstance(text, str):
                text = [text]

            original_text = text[0]
            linguistic = extract_linguistic_features(text)
            embedding = self._get_cached_embedding(original_text)
            features = np.hstack([linguistic, embedding])
            prediction = self.classifier.predict(features)[0]
            confidence = round(np.max(self.classifier.predict_proba(features)[0]) * 100, 2)

            return {
                "label": "FORMAL" if prediction == 1 else "INFORMAL",
                "confidence": confidence
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "label": "UNDEFINED",
                "confidence": 0.0
            }
if __name__ == "__main__":
    predictor = FormalityPredictor()
    print("\nFormality Prediction Bot (Type 'quit' to exit)")

    while True:
        try:
            text = input("\nEnter text: ").strip()
            if text.lower() in ('quit', 'exit'):
                print("Exiting...")
                break

            result = predictor.predict(text)
            print(f"\nPrediction: {result['label']} (Confidence: {result['confidence']}%)")

        except KeyboardInterrupt:
            print("\nExiting...")
            break