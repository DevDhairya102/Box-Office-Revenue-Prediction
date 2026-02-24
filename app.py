import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.preprocessing import LabelEncoder
import os
import json

st.set_page_config(
    page_title="Movie Revenue Predictor",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #06beb6 0%, #48b1bf 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class MovieRevenuePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.predictions_file = "predictions_log.csv"
        self.historical_averages = None
        self.model_info = {}
        
    def load_historical_averages(self):
        try:
            self.historical_averages = pd.read_csv("historical_averages.csv")
            print(f"Loaded {len(self.historical_averages)} historical records")
        except Exception as e:
            print(f"Could not load historical averages: {e}")
            self.historical_averages = None
    
    def load_model(self):
        try:
            self.model = joblib.load("models/best_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
            
            try:
                with open("models/model_info.json", "r") as f:
                    self.model_info = json.load(f)
            except:
                self.model_info = {
                    'best_model_name': 'Gradient Boosting',
                    'r2_score': 0.7639,
                    'rmse': 93678947
                }
            
            self.feature_names = [
                'budget', 'runtime', 'vote_average', 'vote_count', 'popularity',
                'release_year', 'release_month', 'primary_genre', 'lead_actor',
                'director', 'production_company', 'release_season',
                'budget_per_minute', 'vote_weighted_score', 'budget_log', 'popularity_log'
            ]
            
            self.create_default_encoders()
            self.load_historical_averages()
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def create_default_encoders(self):
        genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 
                 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Science Fiction',
                 'Thriller', 'War', 'Western', 'Documentary', 'Music', 'History', 'TV Movie']
        
        companies = ['Walt Disney Pictures', 'Warner Bros.', 'Universal Pictures', 
                    'Paramount Pictures', '20th Century Fox', 'Sony Pictures', 'MGM',
                    'Lionsgate', 'New Line Cinema', 'DreamWorks', 'Miramax', 'Independent']
        
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        
        self.label_encoders = {}
        
        le_genre = LabelEncoder()
        le_genre.fit(genres)
        self.label_encoders['primary_genre'] = le_genre
        
        le_company = LabelEncoder()
        le_company.fit(companies)
        self.label_encoders['production_company'] = le_company
        
        le_season = LabelEncoder()
        le_season.fit(seasons)
        self.label_encoders['release_season'] = le_season
        
        self.label_encoders['lead_actor'] = None
        self.label_encoders['director'] = None
    
    def process_multiple_actors(self, actor_string):
        if not actor_string.strip():
            return "Unknown"
        actors = [actor.strip() for actor in actor_string.split(',')]
        return actors[0] if actors else "Unknown"
    
    def estimate_engagement_metrics(self, genre, lead_actor, director, production_company):
        default_vote_count = 3000
        default_popularity = 8.0
        
        if self.historical_averages is None:
            st.warning("Historical data not loaded. Using defaults.")
            return default_vote_count, default_popularity
        
        df = self.historical_averages.copy()
        df['lead_actor_clean'] = df['lead_actor'].str.strip()
        df['director_clean'] = df['director'].str.strip()
        lead_actor_clean = lead_actor.strip()
        director_clean = director.strip()
        
        # Level 1: Exact match
        match = df[
            (df['primary_genre'] == genre) &
            (df['lead_actor_clean'] == lead_actor_clean) &
            (df['director_clean'] == director_clean) &
            (df['production_company'] == production_company)
        ]
        
        if not match.empty:
            st.success(f"Match: {genre} + {lead_actor} + {director} + {production_company}")
            return int(match['vote_count'].iloc[0]), float(match['popularity'].iloc[0])
        
        # Level 2: Genre + Actor + Director
        match = df[
            (df['primary_genre'] == genre) &
            (df['lead_actor_clean'] == lead_actor_clean) &
            (df['director_clean'] == director_clean) &
            (df['production_company'] == 'Any')
        ]
        
        if not match.empty:
            st.success(f"Match: {genre} + {lead_actor} + {director}")
            return int(match['vote_count'].iloc[0]), float(match['popularity'].iloc[0])
        
        # Level 3: Genre + Actor
        match = df[
            (df['primary_genre'] == genre) &
            (df['lead_actor_clean'] == lead_actor_clean) &
            (df['director'] == 'Any') &
            (df['production_company'] == 'Any')
        ]
        
        if not match.empty:
            st.success(f"Match: {genre} + {lead_actor}")
            return int(match['vote_count'].iloc[0]), float(match['popularity'].iloc[0])
        
        # Level 4: Actor only
        match = df[
            (df['lead_actor_clean'] == lead_actor_clean) &
            (df['primary_genre'] == 'Any') &
            (df['director'] == 'Any') &
            (df['production_company'] == 'Any')
        ]
        
        if not match.empty:
            st.success(f"Match: {lead_actor} (actor only)")
            return int(match['vote_count'].iloc[0]), float(match['popularity'].iloc[0])
        
        # Show available actors
        actor_samples = df[df['primary_genre'] == 'Any']['lead_actor'].unique()[:15]
        st.warning(f"No match for '{lead_actor}'. Try: {', '.join(actor_samples[:5])}")
        return default_vote_count, default_popularity
    
    def encode_categorical(self, value, category):
        if category in ['lead_actor', 'director']:
            return hash(str(value)) % 1000
        
        encoder = self.label_encoders.get(category)
        if encoder is None:
            return 0
        
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0
    
    def prepare_features(self, movie_data):
        budget = movie_data['budget']
        runtime = movie_data['runtime']
        vote_average = movie_data.get('vote_average', 6.5)
        vote_count = movie_data['vote_count']
        popularity = movie_data['popularity']
        
        month = movie_data['release_month']
        if month in [12, 1, 2]:
            season = 'Winter'
        elif month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        else:
            season = 'Fall'
        
        # Debug: Show what's being fed to model
        st.write("Debug - Feature Values:")
        st.write(f"Budget: ${budget:,}, Runtime: {runtime}min, Votes: {vote_count:,}, Popularity: {popularity}")
        st.write(f"Genre: {movie_data['primary_genre']}, Season: {season}, Year: {movie_data['release_year']}, Month: {month}")
        
        features = np.array([
            budget, runtime, vote_average, vote_count, popularity,
            movie_data['release_year'], month,
            self.encode_categorical(movie_data['primary_genre'], 'primary_genre'),
            self.encode_categorical(movie_data['lead_actor'], 'lead_actor'),
            self.encode_categorical(movie_data['director'], 'director'),
            self.encode_categorical(movie_data['production_company'], 'production_company'),
            self.encode_categorical(season, 'release_season'),
            budget / runtime if runtime > 0 else 0,
            vote_average * np.log1p(vote_count),
            np.log1p(budget), np.log1p(popularity)
        ]).reshape(1, -1)
        
        return features
    
    def predict_revenue(self, movie_data):
        try:
            features = self.prepare_features(movie_data)
            prediction = self.model.predict(features)[0]
            return max(0, prediction)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def log_prediction(self, movie_data, prediction):
        try:
            log_data = {
                'movie_name': movie_data.get('movie_name', ''),
                'budget': movie_data.get('budget', 0),
                'runtime': movie_data.get('runtime', 0),
                'vote_count': movie_data.get('vote_count', 0),
                'popularity': movie_data.get('popularity', 0),
                'release_year': movie_data.get('release_year', 0),
                'release_month': movie_data.get('release_month', 0),
                'primary_genre': movie_data.get('primary_genre', ''),
                'lead_actor': movie_data.get('lead_actor', ''),
                'director': movie_data.get('director', ''),
                'production_company': movie_data.get('production_company', ''),
                'predicted_revenue': prediction,
                'prediction_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            df = pd.DataFrame([log_data])
            
            if os.path.exists(self.predictions_file):
                df.to_csv(self.predictions_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.predictions_file, index=False)
                
            return True
        except Exception as e:
            st.warning(f"Logging failed: {e}")
            return False

def main():
    st.markdown('<h1 class="main-header">Movie Revenue Predictor</h1>', unsafe_allow_html=True)
    
    predictor = MovieRevenuePredictor()
    
    if not predictor.load_model():
        st.error("Failed to load model files.")
        st.stop()
    
    with st.sidebar:
        st.header("Model Information")
        
        model_name = predictor.model_info['best_model_name']
        r2_score = predictor.model_info['r2_score']
        rmse = predictor.model_info['rmse']
        
        st.markdown(f"""
        <div class="feature-importance" style="color: #000000;">
        <strong>Selected Model:</strong> {model_name}<br>
        <strong>Accuracy:</strong> R² = {r2_score:.4f}<br>
        <strong>Error:</strong> ±${rmse:,.0f} RMSE<br>
        <strong>Training Data:</strong> 4,880 movies
        </div>
        """, unsafe_allow_html=True)
        
        st.header("Model Comparison")
        
        # Model comparison data
        model_data = {
            'Model': [
                'Gradient Boosting*',
                'Random Forest (Tuned)',
                'Gradient Boosting (Tuned)',
                'Random Forest',
                'Ridge Regression',
                'Lasso Regression',
                'Linear Regression'
            ],
            'R² Score': [0.7639, 0.7415, 0.7365, 0.7363, 0.7262, 0.7262, 0.7262],
            'RMSE ($M)': [93.7, 98.0, 99.0, 99.0, 100.9, 100.9, 100.9]
        }
        
        comparison_df = pd.DataFrame(model_data)
        
        st.markdown("**All Models Tested:**")
        st.dataframe(
            comparison_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                              .highlight_min(subset=['RMSE ($M)'], color='lightgreen')
                              .format({'R² Score': '{:.4f}', 'RMSE ($M)': '{:.1f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("* Selected model based on highest R² and lowest RMSE")
        
        st.header("Top Predictive Features")
        st.markdown("""
        1. **Vote Count** (56%)
        2. **Budget per Minute** (11%)
        3. **Total Budget** (7%)
        4. **Runtime & Genre** (6%)
        5. **Release Timing** (4%)
        """)
    
    st.header("Enter Movie Details")
    
    movie_name = st.text_input("Movie Title", placeholder="e.g., Avengers: Endgame")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input("Budget (USD)", min_value=1000, value=50000000, step=1000000)
        runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=300, value=120, step=5)
        genre = st.selectbox("Primary Genre", [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama',
            'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Science Fiction',
            'Thriller', 'War', 'Western', 'Documentary', 'Music', 'History'
        ])
        
    with col2:
        release_date = st.date_input("Release Date", value=datetime.date.today())
        production_company = st.selectbox("Production Company", [
            'Walt Disney Pictures', 'Warner Bros.', 'Universal Pictures',
            'Paramount Pictures', '20th Century Fox', 'Sony Pictures',
            'MGM', 'Lionsgate', 'New Line Cinema', 'DreamWorks', 'Independent'
        ])
    
    lead_actor = st.text_input("Lead Actor(s)", placeholder="e.g., Aaron Eckhart, Aaron Paul", 
                               help="For ensemble films, enter the most prominent lead actor. Model trained on single lead actor per film.")
    
    director = st.text_input("Director", placeholder="e.g., Jon Amiel, Scott Waugh")
    
    st.markdown("**Audience Engagement Estimation:**")
    estimation_method = st.radio(
        "How would you like to set Vote Count and Popularity?",
        ["Manual Input", "Auto-Calculate from Historical Data"],
        horizontal=True
    )
    
    vote_count_value = 3000
    popularity_value = 8.0
    
    if estimation_method == "Auto-Calculate from Historical Data":
        if st.button("Calculate Based on Cast/Genre/Director/Studio"):
            if lead_actor.strip() and director.strip():
                primary_actor = predictor.process_multiple_actors(lead_actor)
                estimated_votes, estimated_popularity = predictor.estimate_engagement_metrics(
                    genre, primary_actor, director, production_company
                )
                
                st.session_state.auto_vote_count = estimated_votes
                st.session_state.auto_popularity = estimated_popularity
            else:
                st.warning("Please enter lead actor and director first")
        
        vote_count_value = st.session_state.get('auto_vote_count', 3000)
        popularity_value = st.session_state.get('auto_popularity', 8.0)
    
    popularity = st.number_input("Popularity Score", min_value=0.1, max_value=1000.0, 
                               value=float(popularity_value), step=0.1)
    vote_count = st.number_input("Expected Vote Count", min_value=1, max_value=2000000, 
                               value=int(vote_count_value), step=1000)
    
    if st.button("Predict Revenue", key="predict"):
        if not movie_name.strip():
            st.warning("Please enter a movie title.")
        elif not lead_actor.strip():
            st.warning("Please enter a lead actor.")
        elif not director.strip():
            st.warning("Please enter a director.")
        else:
            processed_lead_actor = predictor.process_multiple_actors(lead_actor)
            
            movie_data = {
                'movie_name': movie_name,
                'budget': budget,
                'runtime': runtime,
                'vote_average': 6.5,
                'vote_count': vote_count,
                'popularity': popularity,
                'release_year': release_date.year,
                'release_month': release_date.month,
                'primary_genre': genre,
                'lead_actor': processed_lead_actor,
                'director': director,
                'production_company': production_company
            }
            
            with st.spinner("Analyzing movie potential..."):
                prediction = predictor.predict_revenue(movie_data)
            
            if prediction is not None:
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Revenue</h2>
                    <h1>${prediction:,.0f}</h1>
                    <p>Expected box office performance for "{movie_name}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    roi = (prediction - budget) / budget
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>ROI</h3>
                        <h2>{roi:.2f}x</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    profit = prediction - budget
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>Profit</h3>
                        <h2>${profit:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                if predictor.log_prediction(movie_data, prediction):
                    st.success(f"Prediction logged to {predictor.predictions_file}")
                
                st.info("Model Accuracy: This prediction has a typical error margin of ±$94M (model RMSE). Movie revenue prediction is inherently uncertain due to unpredictable factors like cultural timing, marketing effectiveness, and audience reception.")
                
                st.info("Prediction Confidence Guide:\n- High confidence (±$30-50M): Mid-budget films ($20-60M budget)\n- Medium confidence (±$70-100M): Big budget films ($80-150M)\n- Lower confidence (±$100M+): Blockbusters ($200M+ budget)")
                
                if ',' in lead_actor:
                    st.info("Ensemble Film: Prediction based on primary lead actor. Consider manually increasing vote_count to 50,000+ for A-list ensemble casts.")

if __name__ == "__main__":
    main()