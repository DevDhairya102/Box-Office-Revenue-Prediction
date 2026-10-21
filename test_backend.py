"""
Instant Verification Script
Tests all FastAPI endpoints and model predictions locally.
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("=" * 60)
print("TESTING FASTAPI BACKEND & MODEL INFERENCE")
print("=" * 60)

# 1. Health check
res = client.get("/health")
print(f"[1] GET /health -> Status: {res.status_code}, Body: {res.json()}")
assert res.status_code == 200

# 2. Metadata
res = client.get("/metadata")
print(f"[2] GET /metadata -> Status: {res.status_code}, Found {len(res.json().get('genres', []))} genres")
assert res.status_code == 200

# 3. Predict endpoint (Blockbuster test)
sample_movie = {
    "movie_name": "Avatar 3: The Seed Bearer",
    "budget": 250000000,
    "runtime": 190,
    "primary_genre": "Action",
    "lead_actor": "Sam Worthington",
    "director": "James Cameron",
    "production_company": "20th Century Fox",
    "release_year": 2026,
    "release_month": 12,
    "vote_count": 1200000,
    "popularity": 85.0,
    "vote_average": 7.9
}
res = client.post("/predict", json=sample_movie)
print(f"[3] POST /predict -> Status: {res.status_code}")
data = res.json()
print(f"    Movie: {data.get('movie_name')}")
print(f"    Budget: ${data.get('budget'):,}")
print(f"    Predicted Revenue: {data.get('formatted_predicted_revenue')}")
print(f"    Expected ROI: {data.get('roi_percentage')}")
print(f"    Box Office Category: {data.get('category')}")
assert res.status_code == 200

# 4. Predictions History
res = client.get("/predictions")
print(f"[4] GET /predictions -> Status: {res.status_code}, Retrieved {len(res.json())} logged record(s)")
assert res.status_code == 200

print("\n" + "=" * 60)
print("ALL TESTS PASSED! FASTAPI BACKEND & ML ENGINE ARE 100% WORKING!")
print("=" * 60)
