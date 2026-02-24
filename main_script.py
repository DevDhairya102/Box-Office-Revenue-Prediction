"""
Movie Revenue Prediction - Research Project
Main script to run the complete analysis pipeline
"""

import os
import time
from data_preprocessing import DataPreprocessor
from exploratory_analysis import ExploratoryAnalyzer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

def create_directories():
    """Create necessary directories"""
    directories = ['results', 'models', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully.")

def main():
    """Main execution pipeline"""
    print("="*80)
    print("MOVIE REVENUE PREDICTION - RESEARCH PROJECT")
    print("="*80)
    print("Starting comprehensive analysis pipeline...\n")
    
    start_time = time.time()
    
    # Create directories
    create_directories()
    
    # Step 1: Data Preprocessing
    print("STEP 1: DATA PREPROCESSING")
    print("-" * 40)
    preprocessor = DataPreprocessor()
    X, y, processed_df, encoders = preprocessor.prepare_data()
    
    # Step 2: Exploratory Data Analysis
    print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")
    print("-" * 40)
    analyzer = ExploratoryAnalyzer()
    eda_results = analyzer.comprehensive_eda(processed_df)
    
    # Step 3: Model Training
    print("\nSTEP 3: MODEL TRAINING AND CROSS-VALIDATION")
    print("-" * 40)
    trainer = ModelTrainer()
    training_results, data_splits = trainer.train_models(X, y, perform_tuning=True)
    
    # Step 4: Statistical Significance Testing
    print("\nSTEP 4: STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 40)
    trainer.statistical_significance_test()
    
    # Step 5: Model Evaluation
    print("\nSTEP 5: COMPREHENSIVE MODEL EVALUATION")
    print("-" * 40)
    evaluator = ModelEvaluator()
    performance_summary = evaluator.comprehensive_evaluation(
        training_results, trainer.feature_importance
    )
    
    # Step 6: Save Models
    print("\nSTEP 6: SAVING MODELS")
    print("-" * 40)
    best_model_name = trainer.save_models()
    
    # Final Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - RESEARCH RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total movies analyzed: {len(processed_df):,}")
    print(f"  Features used: {len(X.columns)}")
    print(f"  Time period: {processed_df['release_year'].min()}-{processed_df['release_year'].max()}")
    
    print(f"\nBest Performing Model: {best_model_name}")
    best_performance = performance_summary.iloc[0]
    print(f"  R² Score: {best_performance['Test_R2']:.4f}")
    print(f"  RMSE: ${best_performance['Test_RMSE']:,.0f}")
    print(f"  MAE: ${best_performance['Test_MAE']:,.0f}")
    
    print(f"\nAnalysis completed in {duration:.1f} seconds")
    
    print(f"\nGenerated Files:")
    print("  Data Analysis:")
    print("    - results/distribution_analysis.png")
    print("    - results/correlation_heatmap.png")
    print("    - results/genre_analysis.png")
    print("  Model Evaluation:")
    print("    - results/model_comparison.png")
    print("    - results/prediction_plots.png")
    print("    - results/residual_analysis.png")
    print("    - results/feature_importance.png")
    print("    - results/evaluation_report.txt")
    print("  Saved Models:")
    print("    - models/best_model.pkl")
    print("    - models/scaler.pkl")

    return {
        'processed_data': processed_df,
        'model_results': training_results,
        'performance_summary': performance_summary,
        'best_model': best_model_name,
        'eda_results': eda_results
    }

if __name__ == "__main__":
    results = main()