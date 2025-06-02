import numpy as np
import joblib
import os
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from feature_engineer import extract_linguistic_features, get_embeddings
from transformers import AutoModel

# Configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/formality_model.joblib"
EMBEDDING_MODEL_PATH = "models/embedding_model.joblib"
MODEL_NAME = "distilbert-base-uncased"

def train_and_save_model(train_data):
    """Train and save model with separate embedding model"""
    # Prepare data
    X_informal = train_data["Original Sentence"].tolist()
    X_formal = train_data["Formal Sentence"].tolist()
    X_text = X_informal + X_formal
    y = [0]*len(X_informal) + [1]*len(X_formal)

    # Feature extraction
    print("Extracting linguistic features...")
    linguistic_features = extract_linguistic_features(X_text)
    print("Generating BERT embeddings...")
    embeddings = get_embeddings(X_text)
    X = np.hstack([linguistic_features, embeddings])

    # Train and save
    print("Training classifier...")
    model = HistGradientBoostingClassifier(max_iter=100)
    model.fit(X, y)
    
    # Save embedding model separately
    os.makedirs("models", exist_ok=True)
    embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    joblib.dump({
        'classifier': model,
        'embedding_model_state': embedding_model.state_dict()
    }, MODEL_PATH)
    
    print(f"Model saved to {MODEL_PATH}")
    return model, embedding_model

def load_model():
    """Load both classifier and embedding model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    data = joblib.load(MODEL_PATH)
    model = data['classifier']
    
    # Recreate embedding model
    embedding_model = AutoModel.from_pretrained(MODEL_NAME)
    embedding_model.load_state_dict(data['embedding_model_state'])
    embedding_model.to(DEVICE)
    
    # Attach to classifier object
    model.embedding_model = embedding_model
    return model