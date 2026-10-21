import os
import datetime
import pandas as pd

# Attempt to import supabase client
try:
    from supabase import create_client, Client
    SUPABASE_INSTALLED = True
except ImportError:
    SUPABASE_INSTALLED = False

class SupabaseService:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "").strip()
        self.key = os.getenv("SUPABASE_KEY", "").strip()
        self.client = None
        self.local_log_file = "predictions_log.csv"

        if SUPABASE_INSTALLED and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                print("Supabase client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Supabase client: {e}")
                self.client = None
        else:
            print("Supabase credentials not configured or library missing. Operating in local CSV fallback mode.")

    def log_prediction(self, data: dict, predicted_revenue: float, roi: float) -> bool:
        """Save prediction to Supabase and fallback to local CSV"""
        record = {
            "movie_name": data.get("movie_name", "Untitled"),
            "budget": float(data.get("budget", 0)),
            "runtime": float(data.get("runtime", 0)),
            "primary_genre": data.get("primary_genre", "Unknown"),
            "lead_actor": data.get("lead_actor", "Unknown"),
            "director": data.get("director", "Unknown"),
            "production_company": data.get("production_company", "Unknown"),
            "release_year": int(data.get("release_year", datetime.datetime.now().year)),
            "release_month": int(data.get("release_month", datetime.datetime.now().month)),
            "vote_count": int(data.get("vote_count", 0)),
            "popularity": float(data.get("popularity", 0.0)),
            "predicted_revenue": float(predicted_revenue),
            "roi": float(roi)
        }

        # 1. Supabase attempt
        supabase_success = False
        if self.client:
            try:
                self.client.table("predictions").insert(record).execute()
                supabase_success = True
            except Exception as e:
                print(f"Supabase insert failed: {e}")

        # 2. Local fallback CSV
        try:
            record_copy = dict(record)
            record_copy["created_at"] = datetime.datetime.utcnow().isoformat()
            df = pd.DataFrame([record_copy])
            if os.path.exists(self.local_log_file):
                df.to_csv(self.local_log_file, mode="a", header=False, index=False)
            else:
                df.to_csv(self.local_log_file, index=False)
        except Exception as e:
            print(f"CSV local fallback failed: {e}")

        return supabase_success

    def get_recent_predictions(self, limit: int = 15) -> list:
        """Fetch latest predictions from Supabase or fallback CSV"""
        if self.client:
            try:
                res = self.client.table("predictions").select("*").order("created_at", desc=True).limit(limit).execute()
                if res.data:
                    return res.data
            except Exception as e:
                print(f"Supabase fetch failed: {e}")

        # Fallback to local CSV
        if os.path.exists(self.local_log_file):
            try:
                df = pd.read_csv(self.local_log_file)
                if not df.empty:
                    df = df.tail(limit).iloc[::-1]
                    return df.to_dict(orient="records")
            except Exception as e:
                print(f"Reading CSV fallback failed: {e}")

        return []

# Singleton instance
supabase_service = SupabaseService()
