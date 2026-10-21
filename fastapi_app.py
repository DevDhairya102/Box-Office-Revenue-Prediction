"""
FastAPI Backend for Box Office Revenue Prediction
Compatible with Render deployment and Vercel / Postman consumers.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import Supabase service
from supabase_helper import supabase_service

app = FastAPI(
    title="Box Office Revenue Prediction API",
    description="Machine learning API for forecasting movie box office revenue using ensemble models.",
    version="1.0.0"
)

# Enable CORS for Vercel frontend, Postman, and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Pydantic Schemas
# ==========================================
class MoviePredictionRequest(BaseModel):
    movie_name: str = Field(..., example="Interstellar 2")
    budget: float = Field(..., gt=0, example=165000000)
    runtime: float = Field(..., gt=0, example=145)
    primary_genre: str = Field(..., example="Science Fiction")
    lead_actor: str = Field(..., example="Matthew McConaughey")
    director: str = Field(..., example="Christopher Nolan")
    production_company: str = Field(..., example="Warner Bros.")
    release_year: int = Field(default=2026, example=2026)
    release_month: int = Field(default=7, ge=1, le=12, example=7)
    vote_count: int = Field(default=250000, ge=0, example=250000)
    popularity: float = Field(default=35.0, ge=0.0, example=35.0)
    vote_average: Optional[float] = Field(default=6.5, ge=0.0, le=10.0, example=7.2)

class MoviePredictionResponse(BaseModel):
    movie_name: str
    budget: float
    predicted_revenue: float
    formatted_predicted_revenue: str
    roi: float
    roi_percentage: str
    category: str
    supabase_logged: bool

# ==========================================
# ML Engine Manager
# ==========================================
class MLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_info = {}
        self.genre_list = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 
            'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Science Fiction',
            'Thriller', 'War', 'Western', 'Documentary', 'Music', 'History'
        ]
        self.company_list = [
            'Walt Disney Pictures', 'Warner Bros.', 'Universal Pictures', 
            'Paramount Pictures', '20th Century Fox', 'Sony Pictures', 'MGM',
            'Lionsgate', 'New Line Cinema', 'DreamWorks', 'Independent'
        ]
        self.season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        self.load_model()

    def load_model(self):
        model_path = os.path.join("models", "best_model.pkl")
        scaler_path = os.path.join("models", "scaler.pkl")
        info_path = os.path.join("models", "model_info.json")

        # Auto-bootstrap if files missing
        if not os.path.exists(model_path):
            try:
                import train_or_bootstrap_model
                train_or_bootstrap_model.ensure_model_and_data()
            except Exception as e:
                print(f"Bootstrap trigger error: {e}")

        try:
            self.model = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    self.model_info = json.load(f)
            print("ML Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def prepare_features(self, req: MoviePredictionRequest) -> np.ndarray:
        season = self.season_map.get(req.release_month, 'Summer')
        season_idx = ['Spring', 'Summer', 'Fall', 'Winter'].index(season) if season in ['Spring', 'Summer', 'Fall', 'Winter'] else 1
        
        genre_idx = self.genre_list.index(req.primary_genre) if req.primary_genre in self.genre_list else 0
        company_idx = self.company_list.index(req.production_company) if req.production_company in self.company_list else len(self.company_list) - 1
        
        actor_hash = abs(hash(req.lead_actor)) % 1000
        director_hash = abs(hash(req.director)) % 1000

        budget_per_min = req.budget / req.runtime if req.runtime > 0 else 0
        vote_weighted_score = req.vote_average * np.log1p(req.vote_count)
        budget_log = np.log1p(req.budget)
        popularity_log = np.log1p(req.popularity)

        features = np.array([
            req.budget,
            req.runtime,
            req.vote_average,
            req.vote_count,
            req.popularity,
            req.release_year,
            req.release_month,
            genre_idx,
            actor_hash,
            director_hash,
            company_idx,
            season_idx,
            budget_per_min,
            vote_weighted_score,
            budget_log,
            popularity_log
        ]).reshape(1, -1)
        
        return features

    def predict(self, req: MoviePredictionRequest) -> float:
        if self.model is None:
            self.load_model()
        features = self.prepare_features(req)
        raw_pred = self.model.predict(features)[0]
        return float(max(0, raw_pred))

predictor = MLPredictor()

# ==========================================
# API Endpoints
# ==========================================
@app.get("/")
def root():
    return {
        "title": "Box Office Revenue Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "supabase_connected": supabase_service.client is not None
    }

@app.get("/model-info")
def get_model_info():
    return predictor.model_info or {
        "best_model_name": "Gradient Boosting",
        "r2_score": 0.7639,
        "rmse": 93678947,
        "mae": 45100000
    }

@app.get("/metadata")
def get_metadata():
    return {
        "genres": predictor.genre_list,
        "production_companies": predictor.company_list,
        "seasons": ["Spring", "Summer", "Fall", "Winter"]
    }

@app.post("/predict", response_model=MoviePredictionResponse)
def predict_revenue(payload: MoviePredictionRequest):
    try:
        predicted_revenue = predictor.predict(payload)
        roi = (predicted_revenue - payload.budget) / payload.budget if payload.budget > 0 else 0.0

        if roi >= 1.5:
            category = "Blockbuster Hit"
        elif roi > 0:
            category = "Profitable"
        elif roi > -0.3:
            category = "Moderate Performer"
        else:
            category = "Flop"

        # Log prediction to Supabase & CSV
        logged = supabase_service.log_prediction(payload.dict(), predicted_revenue, roi)

        return MoviePredictionResponse(
            movie_name=payload.movie_name,
            budget=payload.budget,
            predicted_revenue=round(predicted_revenue, 2),
            formatted_predicted_revenue=f"${predicted_revenue:,.2f}",
            roi=round(roi, 4),
            roi_percentage=f"{roi * 100:.1f}%",
            category=category,
            supabase_logged=logged
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions")
def get_predictions(limit: int = 15):
    return supabase_service.get_recent_predictions(limit=limit)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=port, reload=True)
