import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModel, BertModel
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"  # make sure this matches with what the state_dict expects

nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def extract_linguistic_features(texts):
    """
    Extract linguistic features from input texts.
    
    Args:
        texts: List of input strings to analyze
        
    Returns:
        numpy.ndarray: Array of features for each text containing:
            - Average word length
            - Count of contractions
            - Count of punctuation marks
    """
    features = []
    for doc in nlp.pipe(texts, batch_size=128):
        words = [token.text for token in doc if not token.is_punct]
        features.append([
            np.mean([len(w) for w in words]) if words else 0,
            sum(1 for t in doc if "'" in t.text),
            sum(1 for t in doc if t.is_punct)
        ])
    return np.array(features)

def get_embeddings(texts, model_weights_path=None):
    """
    Generate BERT embeddings for input texts.
    
    Args:
        texts: List of input strings to embed
        model_weights_path: Optional path to saved model weights
        
    Returns:
        numpy.ndarray: Array of embeddings for each text
    """
    if model_weights_path and os.path.exists(model_weights_path):
        print(f"Loading saved model from {model_weights_path}")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
    else:
        print("Using default transformer model")
        model = AutoModel.from_pretrained(MODEL_NAME)
    
    model.to(DEVICE)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(texts), 32), desc="Embedding"):
        batch = texts[i:i+32]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(embeddings)
