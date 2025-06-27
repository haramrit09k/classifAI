from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import re
from datetime import datetime
import sqlite3
import os

# Initialize FastAPI app
app = FastAPI(
    title="Review Classifier API",
    description="Classifies app reviews into 5 categories: Positive, Negative, Bug Report, Feature Request, Spam",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load the trained model and vectorizer
try:
    model = joblib.load('data/models/classifier_model.pkl')
    vectorizer = joblib.load('data/models/tfidf_vectorizer.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Request/Response models
class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    review: str
    predicted_label: str
    confidence: float
    timestamp: str

# Database setup
def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT,
            predicted_label TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(review: str, label: str, confidence: float):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO predictions (review, predicted_label, confidence) VALUES (?, ?, ?)',
        (review, label, confidence)
    )
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Web interface route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=ReviewResponse)
async def predict_review(request: ReviewRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        cleaned_text = preprocess_text(request.review)
        
        if len(cleaned_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Review text is empty after preprocessing")
        
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()
        
        log_prediction(request.review, prediction, confidence)
        
        return ReviewResponse(
            review=request.review,
            predicted_label=prediction,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/stats")
async def get_stats():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = cursor.fetchone()[0]
    
    cursor.execute('SELECT predicted_label, COUNT(*) FROM predictions GROUP BY predicted_label')
    label_counts = dict(cursor.fetchall())
    
    cursor.execute('SELECT AVG(confidence) FROM predictions')
    avg_confidence = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_predictions": total_predictions,
        "predictions_by_label": label_counts,
        "average_confidence": avg_confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)