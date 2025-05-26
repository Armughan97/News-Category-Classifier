from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Load preprocessing objects and model
tfidf     = joblib.load("tfidf.pkl")
label_enc = joblib.load("label_encoder.pkl")
model     = joblib.load("model.pkl")

# Download NLTK resources if not already
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 2. Define preprocessing function to match training
def preprocess_text(text: str) -> str:
    """
    Preprocessing function that matches the training data preparation.
    The TF-IDF vectorizer will handle stopwords removal internally.
    """
    # Basic text cleaning (match what was done to create 'clean_content')
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)         # Replace multiple spaces with single space
    text = text.strip()                      # Remove leading/trailing whitespace
    
    return text

# 3. FastAPI request/response model
class NewsItem(BaseModel):
    text: str

class Prediction(BaseModel):
    predicted_category: str

app = FastAPI(
    title="News Category Classifier",
    description="Send a piece of text and get back its predicted news category.",
    version="1.0.0",
)

# 4. Prediction endpoint
@app.post("/predict", response_model=Prediction)
def predict(item: NewsItem):
    if not item.text or len(item.text) < 5:
        raise HTTPException(status_code=400, detail="`text` is too short.")
    
    try:
        # Preprocess text (basic cleaning only)
        clean_text = preprocess_text(item.text)
        
        # Transform using TF-IDF (this will handle stopwords, ngrams, etc.)
        vec = tfidf.transform([clean_text])
        
        # Debug: Verify feature count matches
        print(f"Feature count: {vec.shape[1]} (should be 8000)")

        # FIX: Reduce features from 8000 to 7999
        if vec.shape[1] == 8000 and model.n_features_in_ == 7999:
            # Remove the last feature to match model expectations
            vec = vec[:, :7999]
            print(f"Adjusted features from 8000 to {vec.shape[1]}")
        
        # Make prediction
        label_idx = model.predict(vec)[0]
        category = label_enc.inverse_transform([label_idx])[0]
        
        return Prediction(predicted_category=category)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"Hi": "there"}






