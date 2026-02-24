import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ExploratoryAnalyzer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def basic_statistics(self, df):
        """Generate comprehensive dataset statistics"""
        print("="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"Total movies: {len(df):,}")
        print(f"Years covered: {df['release_year'].min()} - {df['release_year'].max()}")
        print(f"Average budget: ${df['budget'].mean():,.0f}")
        print(f"Median budget: ${df['budget'].median():,.0f}")
        print(f"Average revenue: ${df['revenue'].mean():,.0f}")
        print(f"Median revenue: ${df['revenue'].median():,.0f}")
        print(f"Average ROI: {df['roi'].mean():.2f}x")
        print(f"Median ROI: {df['roi'].median():.2f}x")
        
        # Revenue statistics by percentiles
        revenue_percentiles = np.percentile(df['revenue'], [25, 50, 75, 90, 95, 99])
        print(f"\nRevenue Distribution:")
        print(f"25th percentile: ${revenue_percentiles[0]:,.0f}")
        print(f"50th percentile: ${revenue_percentiles[1]:,.0f}")
        print(f"75th percentile: ${revenue_percentiles[2]:,.0f}")
        print(f"90th percentile: ${revenue_percentiles[3]:,.0f}")
        print(f"95th percentile: ${revenue_percentiles[4]:,.0f}")
        print(f"99th percentile: ${revenue_percentiles[5]:,.0f}")
        
        # Genre analysis
        print(f"\nGenre Distribution:")
        genre_counts = df['primary_genre'].value_counts().head(10)
        for genre, count in genre_counts.items():
            print(f"{genre}: {count:,} movies")
        
        # Temporal analysis
        print(f"\nTemporal Patterns:")
        decade_counts = df['release_year'].apply(lambda x: f"{x//10*10}s").value_counts().sort_index()
        for decade, count in decade_counts.items():
            print(f"{decade}: {count:,} movies")
            
        return {
            'basic_stats': df.describe(),
            'genre_distribution': genre_counts,
            'decade_distribution': decade_counts
        }
    
    def correlation_analysis(self, df):
        """Analyze correlations between features and revenue"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate correlations with revenue
        numeric_cols = ['budget', 'runtime', 'vote_average', 'vote_count', 'popularity', 
                       'release_year', 'budget_per_minute', 'vote_weighted_score']
        
        correlations = df[numeric_cols + ['revenue']].corr()['revenue'].sort_values(ascending=False)
        
        print("Correlation with Revenue:")
        for feature, corr in correlations.items():
            if feature != 'revenue':
                print(f"{feature}: {corr:.3f}")
        
        return correlations
    
    def create_distribution_plots(self, df, save_path="results/"):
        """Create distribution and relationship plots"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Movie Revenue Prediction: Data Distribution Analysis', fontsize=16, y=0.98)
        
        # Revenue distribution (log scale)
        axes[0,0].hist(np.log1p(df['revenue']), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Revenue Distribution (Log Scale)', fontweight='bold')
        axes[0,0].set_xlabel('Log(Revenue + 1)')
        axes[0,0].set_ylabel('Frequency')
        
        # Budget vs Revenue
        axes[0,1].scatter(np.log1p(df['budget']), np.log1p(df['revenue']), alpha=0.5, color='coral')
        axes[0,1].set_title('Budget vs Revenue Relationship', fontweight='bold')
        axes[0,1].set_xlabel('Log(Budget + 1)')
        axes[0,1].set_ylabel('Log(Revenue + 1)')
        
        # Genre performance
        genre_revenue = df.groupby('primary_genre')['revenue'].mean().sort_values(ascending=False).head(12)
        bars = axes[0,2].barh(range(len(genre_revenue)), genre_revenue.values, color='lightgreen')
        axes[0,2].set_yticks(range(len(genre_revenue)))
        axes[0,2].set_yticklabels(genre_revenue.index, fontsize=9)
        axes[0,2].set_title('Average Revenue by Genre', fontweight='bold')
        axes[0,2].set_xlabel('Average Revenue ($)')
        
        # Release month patterns
        monthly_revenue = df.groupby('release_month')['revenue'].mean()
        bars = axes[1,0].bar(monthly_revenue.index, monthly_revenue.values, color='gold')
        axes[1,0].set_title('Seasonal Revenue Patterns', fontweight='bold')
        axes[1,0].set_xlabel('Release Month')
        axes[1,0].set_ylabel('Average Revenue ($)')
        axes[1,0].set_xticks(range(1, 13))
        
        # Vote average vs Revenue
        axes[1,1].scatter(df['vote_average'], np.log1p(df['revenue']), alpha=0.5, color='purple')
        axes[1,1].set_title('Rating vs Revenue', fontweight='bold')
        axes[1,1].set_xlabel('Vote Average (IMDb Rating)')
        axes[1,1].set_ylabel('Log(Revenue + 1)')
        
        # Runtime distribution
        axes[1,2].hist(df['runtime'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1,2].set_title('Runtime Distribution', fontweight='bold')
        axes[1,2].set_xlabel('Runtime (minutes)')
        axes[1,2].set_ylabel('Frequency')
        
        # Revenue trends over time
        yearly_revenue = df.groupby('release_year')['revenue'].mean()
        axes[2,0].plot(yearly_revenue.index, yearly_revenue.values, marker='o', linewidth=2, color='red')
        axes[2,0].set_title('Revenue Trends Over Time', fontweight='bold')
        axes[2,0].set_xlabel('Release Year')
        axes[2,0].set_ylabel('Average Revenue ($)')
        axes[2,0].grid(True, alpha=0.3)
        
        # Budget vs ROI
        axes[2,1].scatter(np.log1p(df['budget']), df['roi'], alpha=0.5, color='brown')
        axes[2,1].set_title('Budget vs ROI', fontweight='bold')
        axes[2,1].set_xlabel('Log(Budget + 1)')
        axes[2,1].set_ylabel('Return on Investment (ROI)')
        axes[2,1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # Popularity vs Revenue
        axes[2,2].scatter(np.log1p(df['popularity']), np.log1p(df['revenue']), alpha=0.5, color='teal')
        axes[2,2].set_title('Popularity vs Revenue', fontweight='bold')
        axes[2,2].set_xlabel('Log(Popularity + 1)')
        axes[2,2].set_ylabel('Log(Revenue + 1)')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution plots saved to {save_path}distribution_analysis.png")
    
    def create_correlation_heatmap(self, df, save_path="results/"):
        """Create detailed correlation heatmap"""
        numeric_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 
                       'popularity', 'release_year', 'release_month', 
                       'budget_per_minute', 'vote_weighted_score']
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0,
                    mask=mask,
                    square=True,
                    fmt='.2f',
                    cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_path}correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation heatmap saved to {save_path}correlation_heatmap.png")
    
    def genre_analysis(self, df, save_path="results/"):
        """Detailed genre analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Genre Analysis', fontsize=16, fontweight='bold')
        
        # Genre count
        genre_counts = df['primary_genre'].value_counts().head(15)
        bars = axes[0,0].barh(range(len(genre_counts)), genre_counts.values, color='lightblue')
        axes[0,0].set_yticks(range(len(genre_counts)))
        axes[0,0].set_yticklabels(genre_counts.index)
        axes[0,0].set_title('Movie Count by Genre')
        axes[0,0].set_xlabel('Number of Movies')
        
        # Average revenue by genre
        genre_revenue = df.groupby('primary_genre')['revenue'].mean().sort_values(ascending=False).head(15)
        bars = axes[0,1].barh(range(len(genre_revenue)), genre_revenue.values, color='lightcoral')
        axes[0,1].set_yticks(range(len(genre_revenue)))
        axes[0,1].set_yticklabels(genre_revenue.index)
        axes[0,1].set_title('Average Revenue by Genre')
        axes[0,1].set_xlabel('Average Revenue ($)')
        
        # Genre ROI
        genre_roi = df.groupby('primary_genre')['roi'].mean().sort_values(ascending=False).head(15)
        bars = axes[1,0].barh(range(len(genre_roi)), genre_roi.values, color='lightgreen')
        axes[1,0].set_yticks(range(len(genre_roi)))
        axes[1,0].set_yticklabels(genre_roi.index)
        axes[1,0].set_title('Average ROI by Genre')
        axes[1,0].set_xlabel('Average ROI')
        axes[1,0].axvline(x=1, color='red', linestyle='--', alpha=0.7)
        
        # Genre budget
        genre_budget = df.groupby('primary_genre')['budget'].mean().sort_values(ascending=False).head(15)
        bars = axes[1,1].barh(range(len(genre_budget)), genre_budget.values, color='gold')
        axes[1,1].set_yticks(range(len(genre_budget)))
        axes[1,1].set_yticklabels(genre_budget.index)
        axes[1,1].set_title('Average Budget by Genre')
        axes[1,1].set_xlabel('Average Budget ($)')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Genre analysis saved to {save_path}genre_analysis.png")
    
    def comprehensive_eda(self, df, save_path="results/"):
        """Run complete exploratory data analysis"""
        print("Running comprehensive exploratory data analysis...")
        
        # Basic statistics
        stats_results = self.basic_statistics(df)
        
        # Correlation analysis
        correlation_results = self.correlation_analysis(df)
        
        # Create visualizations
        self.create_distribution_plots(df, save_path)
        self.create_correlation_heatmap(df, save_path)
        self.genre_analysis(df, save_path)
        
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files:")
        print(f"- {save_path}distribution_analysis.png")
        print(f"- {save_path}correlation_heatmap.png")
        print(f"- {save_path}genre_analysis.png")
        
        return {
            'statistics': stats_results,
            'correlations': correlation_results
        }

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    X, y, df, encoders = preprocessor.prepare_data()
    
    # Run EDA
    analyzer = ExploratoryAnalyzer()
    results = analyzer.comprehensive_eda(df)
