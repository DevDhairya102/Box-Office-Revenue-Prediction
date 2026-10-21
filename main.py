"""
FastAPI Backend for Box Office Revenue Prediction
Compatible with Render deployment and Vercel / Postman.
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==========================================
# Supabase Integration
# ==========================================
try:
    from supabase import create_client
    SUPABASE_INSTALLED = True
except ImportError:
    SUPABASE_INSTALLED = False

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
supabase_client = None

if SUPABASE_INSTALLED and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase.")
    except Exception as e:
        print(f"Supabase connection warning: {e}")

def save_to_database(record: dict) -> bool:
    """Save record to Supabase, fallback to predictions_log.csv"""
    success = False
    if supabase_client:
        try:
            supabase_client.table("predictions").insert(record).execute()
            success = True
        except Exception as e:
            print(f"Supabase insert error: {e}")
    
    # Also log to local CSV
    try:
        record_copy = dict(record)
        record_copy["created_at"] = datetime.datetime.utcnow().isoformat()
        df = pd.DataFrame([record_copy])
        if os.path.exists("predictions_log.csv"):
            df.to_csv("predictions_log.csv", mode="a", header=False, index=False)
        else:
            df.to_csv("predictions_log.csv", index=False)
    except Exception as e:
        print(f"CSV log error: {e}")
    return success

def fetch_predictions(limit: int = 15):
    """Retrieve predictions from Supabase or fallback CSV"""
    if supabase_client:
        try:
            res = supabase_client.table("predictions").select("*").order("created_at", desc=True).limit(limit).execute()
            if res.data:
                return res.data
        except Exception as e:
            print(f"Supabase fetch error: {e}")

    if os.path.exists("predictions_log.csv"):
        try:
            df = pd.read_csv("predictions_log.csv")
            if not df.empty:
                return df.tail(limit).iloc[::-1].to_dict(orient="records")
        except Exception:
            pass
    return []

# ==========================================
# FastAPI Setup & CORS
# ==========================================
app = FastAPI(title="Box Office Revenue Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ML Model & Encoders
# ==========================================
GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 
    'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Science Fiction',
    'Thriller', 'War', 'Western', 'Documentary', 'Music', 'History'
]
COMPANIES = [
    'Walt Disney Pictures', 'Warner Bros.', 'Universal Pictures', 
    'Paramount Pictures', '20th Century Fox', 'Sony Pictures', 'MGM',
    'Lionsgate', 'New Line Cinema', 'DreamWorks', 'Independent'
]
SEASONS = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}

model = None
scaler = None

def load_ml_model():
    global model, scaler
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    if not os.path.exists(model_path):
        import train_or_bootstrap_model
        train_or_bootstrap_model.ensure_model_and_data()

    try:
        model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        print("ML Model loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")

load_ml_model()

# ==========================================
# Request / Response Schemas
# ==========================================
class MovieInput(BaseModel):
    movie_name: str
    budget: float
    runtime: float
    primary_genre: str
    lead_actor: str
    director: str
    production_company: str
    release_year: int = 2026
    release_month: int = 7
    vote_count: int = 150000
    popularity: float = 25.0
    vote_average: Optional[float] = 6.5

# ==========================================
# API Routes
# ==========================================
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Box Office Revenue Prediction API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "supabase_connected": supabase_client is not None
    }

@app.get("/metadata")
def metadata():
    return {
        "genres": GENRES,
        "production_companies": COMPANIES
    }

@app.post("/predict")
def predict(payload: MovieInput):
    if model is None:
        load_ml_model()
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model not available.")

    # Feature transformation matching research methodology
    season = SEASONS.get(payload.release_month, 'Summer')
    season_idx = ['Spring', 'Summer', 'Fall', 'Winter'].index(season) if season in ['Spring', 'Summer', 'Fall', 'Winter'] else 1
    genre_idx = GENRES.index(payload.primary_genre) if payload.primary_genre in GENRES else 0
    company_idx = COMPANIES.index(payload.production_company) if payload.production_company in COMPANIES else len(COMPANIES) - 1
    actor_hash = abs(hash(payload.lead_actor)) % 1000
    director_hash = abs(hash(payload.director)) % 1000

    budget_per_min = payload.budget / payload.runtime if payload.runtime > 0 else 0
    vote_weighted_score = payload.vote_average * np.log1p(payload.vote_count)
    budget_log = np.log1p(payload.budget)
    popularity_log = np.log1p(payload.popularity)

    features = np.array([
        payload.budget, payload.runtime, payload.vote_average, payload.vote_count, payload.popularity,
        payload.release_year, payload.release_month, genre_idx, actor_hash,
        director_hash, company_idx, season_idx,
        budget_per_min, vote_weighted_score, budget_log, popularity_log
    ]).reshape(1, -1)

    predicted_rev = float(max(0, model.predict(features)[0]))
    roi = (predicted_rev - payload.budget) / payload.budget if payload.budget > 0 else 0.0

    if roi >= 1.5:
        category = "Blockbuster Hit"
    elif roi > 0:
        category = "Profitable"
    elif roi > -0.3:
        category = "Moderate Performer"
    else:
        category = "Flop"

    # Save to Supabase
    db_record = {
        "movie_name": payload.movie_name,
        "budget": payload.budget,
        "runtime": payload.runtime,
        "primary_genre": payload.primary_genre,
        "lead_actor": payload.lead_actor,
        "director": payload.director,
        "production_company": payload.production_company,
        "release_year": payload.release_year,
        "release_month": payload.release_month,
        "vote_count": payload.vote_count,
        "popularity": payload.popularity,
        "predicted_revenue": round(predicted_rev, 2),
        "roi": round(roi, 4)
    }
    db_success = save_to_database(db_record)

    return {
        "movie_name": payload.movie_name,
        "budget": payload.budget,
        "predicted_revenue": round(predicted_rev, 2),
        "formatted_predicted_revenue": f"${predicted_rev:,.2f}",
        "roi": round(roi, 4),
        "roi_percentage": f"{roi * 100:.1f}%",
        "category": category,
        "supabase_logged": db_success
    }

@app.get("/predictions")
def get_history(limit: int = 15):
    return fetch_predictions(limit=limit)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
