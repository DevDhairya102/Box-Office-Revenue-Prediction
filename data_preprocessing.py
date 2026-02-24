import pandas as pd
import numpy as np
import ast
import json
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_and_merge_data(self, movies_path="movies_metadata.csv", credits_path="credits.csv"):
        """Load and merge movies metadata with credits data"""
        print("Loading movie metadata...")
        
        # Load movies metadata
        movies_df = pd.read_csv(movies_path, low_memory=False)
        
        # Load credits
        credits_df = pd.read_csv(credits_path)
        
        print(f"Movies metadata shape: {movies_df.shape}")
        print(f"Credits shape: {credits_df.shape}")
        
        # Clean and merge
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
        
        # Merge on ID
        df = movies_df.merge(credits_df, on='id', how='inner')
        print(f"Merged dataset shape: {df.shape}")
        
        return df
    
    def extract_json_field(self, json_str, field_key, index=0):
        """Extract specific field from JSON string"""
        try:
            if pd.isna(json_str):
                return 'Unknown'
            data = ast.literal_eval(json_str)
            if data and len(data) > index:
                return data[index].get(field_key, 'Unknown')
            return 'Unknown'
        except:
            return 'Unknown'
    
    def get_director(self, crew_str):
        """Extract director from crew JSON"""
        try:
            if pd.isna(crew_str):
                return 'Unknown'
            crew = ast.literal_eval(crew_str)
            directors = [person['name'] for person in crew if person['job'] == 'Director']
            return directors[0] if directors else 'Unknown'
        except:
            return 'Unknown'
    
    def feature_engineering(self, df):
        """Extract and engineer features from raw data"""
        print("Engineering features...")
        
        df = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter valid movies (budget > 0, revenue > 0)
        df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
        df = df[df['runtime'] > 0]
        
        # Extract release year and month
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df.dropna(subset=['release_date'])
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        
        # Filter movies from 1980 onwards for consistency
        df = df[df['release_year'] >= 1980]
        
        # Extract features from JSON fields
        df['primary_genre'] = df['genres'].apply(lambda x: self.extract_json_field(x, 'name'))
        df['lead_actor'] = df['cast'].apply(lambda x: self.extract_json_field(x, 'name'))
        df['director'] = df['crew'].apply(self.get_director)
        df['production_company'] = df['production_companies'].apply(lambda x: self.extract_json_field(x, 'name'))
        
        # Create season feature
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['release_season'] = df['release_month'].apply(get_season)
        
        # Feature engineering
        df['budget_per_minute'] = df['budget'] / df['runtime']
        df['vote_weighted_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        df['budget_log'] = np.log1p(df['budget'])
        df['popularity_log'] = np.log1p(df['popularity'])
        
        # ROI calculation for analysis
        df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        
        # Select final features
        feature_cols = [
            'budget', 'runtime', 'vote_average', 'vote_count', 'popularity',
            'release_year', 'release_month', 'primary_genre', 'lead_actor',
            'director', 'production_company', 'release_season',
            'budget_per_minute', 'vote_weighted_score', 'budget_log', 'popularity_log'
        ]
        
        df = df[feature_cols + ['revenue', 'roi']].dropna()
        
        print(f"Final dataset shape after feature engineering: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, top_n=50):
        """Encode categorical variables"""
        categorical_cols = ['primary_genre', 'lead_actor', 'director', 'production_company', 'release_season']
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            # Keep only top categories for high-cardinality features
            if col in ['lead_actor', 'director', 'production_company']:
                top_categories = df[col].value_counts().head(top_n).index
                df_encoded[col] = df_encoded[col].apply(lambda x: x if x in top_categories else 'Other')
            
            # Label encode
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        return df_encoded
    
    def prepare_data(self, movies_path="movies_metadata.csv", credits_path="credits.csv"):
        """Main preprocessing pipeline"""
        # Load and merge data
        raw_df = self.load_and_merge_data(movies_path, credits_path)
        
        # Feature engineering
        processed_df = self.feature_engineering(raw_df)
        
        # Encode categorical features
        encoded_df = self.encode_categorical_features(processed_df)
        
        # Separate features and target
        X = encoded_df.drop(['revenue', 'roi'], axis=1)
        y = encoded_df['revenue']
        
        return X, y, processed_df, self.label_encoders

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X, y, df, encoders = preprocessor.prepare_data()
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
