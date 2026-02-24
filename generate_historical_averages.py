"""
Generate historical averages for vote count and popularity
Run this script after training your models to create the lookup table
"""

import pandas as pd
from data_preprocessing import DataPreprocessor

def generate_historical_averages():
    """Generate historical averages CSV for UI auto-calculation"""
    print("Generating historical averages from training data...")
    
    # Load the same processed data used for training
    preprocessor = DataPreprocessor()
    X, y, processed_df, encoders = preprocessor.prepare_data()
    
    print(f"Processing {len(processed_df)} movies...")
    
    # Calculate averages by different combinations with production company
    
    # 1. Genre + Actor + Director + Production (most specific)
    full_combo = processed_df.groupby(['primary_genre', 'lead_actor', 'director', 'production_company']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    
    # 2. Genre + Actor + Director (high specificity)
    genre_actor_director = processed_df.groupby(['primary_genre', 'lead_actor', 'director']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    genre_actor_director['production_company'] = 'Any'
    
    # 3. Genre + Actor (medium specificity)
    genre_actor = processed_df.groupby(['primary_genre', 'lead_actor']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    genre_actor['director'] = 'Any'
    genre_actor['production_company'] = 'Any'
    
    # 4. Actor only (for known actors)
    actor_only = processed_df.groupby(['lead_actor']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    actor_only['primary_genre'] = 'Any'
    actor_only['director'] = 'Any'
    actor_only['production_company'] = 'Any'
    
    # Combine all levels
    all_averages = pd.concat([
        full_combo,
        genre_actor_director,
        genre_actor,
        actor_only
    ], ignore_index=True)
    
    # Reorder columns
    all_averages = all_averages[['primary_genre', 'lead_actor', 'director', 'production_company', 'vote_count', 'popularity', 'revenue', 'budget']]
    
    # Round values for cleaner display
    all_averages['vote_count'] = all_averages['vote_count'].round().astype(int)
    all_averages['popularity'] = all_averages['popularity'].round(1)
    all_averages['revenue'] = all_averages['revenue'].round().astype(int)
    all_averages['budget'] = all_averages['budget'].round().astype(int)
    
    # Save to CSV
    all_averages.to_csv("historical_averages.csv", index=False)
    
    print(f"Historical averages saved to historical_averages.csv")
    print(f"Total combinations: {len(all_averages):,}")
    
    # Show some examples
    print("\nSample genre averages:")
    genre_samples = all_averages[all_averages['lead_actor'] == 'Any'][all_averages['director'] == 'Any']
    print(genre_samples[['primary_genre', 'vote_count', 'popularity', 'revenue']].head(10).to_string(index=False))
    
    # Show top actors by average revenue
    print("\nTop actors by average revenue:")
    actor_revenues = all_averages[
        (all_averages['primary_genre'] == 'Any') & 
        (all_averages['director'] == 'Any')
    ].sort_values('revenue', ascending=False).head(10)
    print(actor_revenues[['lead_actor', 'vote_count', 'popularity', 'revenue']].to_string(index=False))
    
    return all_averages

if __name__ == "__main__":
    averages_df = generate_historical_averages()