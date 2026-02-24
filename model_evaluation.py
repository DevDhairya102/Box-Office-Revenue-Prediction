import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        
    def create_performance_comparison(self, results, save_path="results/"):
        """Create comprehensive model performance comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        
        # R² Score comparison
        r2_scores = [results[name]['test_r2'] for name in model_names]
        bars = axes[0,0].bar(range(len(model_names)), r2_scores, color='skyblue', edgecolor='black')
        axes[0,0].set_title('R² Score Comparison', fontweight='bold')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_xticks(range(len(model_names)))
        axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE comparison
        rmse_scores = [results[name]['test_rmse'] for name in model_names]
        bars = axes[0,1].bar(range(len(model_names)), rmse_scores, color='lightcoral', edgecolor='black')
        axes[0,1].set_title('RMSE Comparison', fontweight='bold')
        axes[0,1].set_ylabel('RMSE (USD)')
        axes[0,1].set_xticks(range(len(model_names)))
        axes[0,1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # Cross-validation RMSE with error bars
        cv_means = [results[name]['cv_rmse_mean'] for name in model_names]
        cv_stds = [results[name]['cv_rmse_std'] for name in model_names]
        bars = axes[0,2].bar(range(len(model_names)), cv_means, yerr=cv_stds, 
                            color='lightgreen', edgecolor='black', capsize=5)
        axes[0,2].set_title('Cross-Validation RMSE', fontweight='bold')
        axes[0,2].set_ylabel('CV RMSE (USD)')
        axes[0,2].set_xticks(range(len(model_names)))
        axes[0,2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,2].grid(True, alpha=0.3)
        
        # MAE comparison
        mae_scores = [results[name]['test_mae'] for name in model_names]
        bars = axes[1,0].bar(range(len(model_names)), mae_scores, color='gold', edgecolor='black')
        axes[1,0].set_title('Mean Absolute Error Comparison', fontweight='bold')
        axes[1,0].set_ylabel('MAE (USD)')
        axes[1,0].set_xticks(range(len(model_names)))
        axes[1,0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # Model ranking based on R²
        ranking_data = pd.DataFrame({
            'Model': model_names,
            'R²': r2_scores,
            'RMSE': rmse_scores,
            'MAE': mae_scores
        }).sort_values('R²', ascending=False)
        
        axes[1,2].axis('tight')
        axes[1,2].axis('off')
        table = axes[1,2].table(cellText=[[f"{row['Model']}", f"{row['R²']:.4f}", 
                                          f"{row['RMSE']:,.0f}", f"{row['MAE']:,.0f}"] 
                                         for _, row in ranking_data.iterrows()],
                               colLabels=['Model', 'R²', 'RMSE', 'MAE'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1,2].set_title('Model Performance Ranking', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plots saved to {save_path}model_comparison.png")
        
    def create_prediction_plots(self, results, save_path="results/"):
        """Create predicted vs actual plots for best models"""
        # Get top 3 models by R²
        sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)[:3]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Predicted vs Actual Revenue (Top 3 Models)', fontsize=16, fontweight='bold')
        
        for i, (name, result) in enumerate(sorted_models):
            y_test = result['y_test']
            y_pred = result['y_pred']
            r2 = result['test_r2']
            rmse = result['test_rmse']
            
            # Scatter plot
            axes[i].scatter(y_test, y_pred, alpha=0.6, color='blue')
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Formatting
            axes[i].set_xlabel('Actual Revenue (USD)')
            axes[i].set_ylabel('Predicted Revenue (USD)')
            axes[i].set_title(f'{name}\nR² = {r2:.4f}, RMSE = ${rmse:,.0f}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Format axes to show values in millions/billions
            axes[i].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}prediction_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction plots saved to {save_path}prediction_plots.png")
    
    def create_residual_analysis(self, results, save_path="results/"):
        """Create residual analysis plots"""
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_result = results[best_model_name]
        
        y_test = best_result['y_test']
        y_pred = best_result['y_pred']
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Residual Analysis: {best_model_name}', fontsize=16, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0,0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0,0].axhline(y=0, color='red', linestyle='--')
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Predicted Values')
        axes[0,0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[0,1].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('Residuals')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Residuals')
        axes[0,1].grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot (Normality Check)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Absolute residuals vs predicted (heteroscedasticity check)
        axes[1,1].scatter(y_pred, np.abs(residuals), alpha=0.6, color='green')
        axes[1,1].set_xlabel('Predicted Values')
        axes[1,1].set_ylabel('Absolute Residuals')
        axes[1,1].set_title('Absolute Residuals vs Predicted (Heteroscedasticity)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Residual analysis saved to {save_path}residual_analysis.png")
        
        # Statistical tests
        print("\n" + "="*50)
        print("RESIDUAL ANALYSIS STATISTICS")
        print("="*50)
        
        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
        if len(residuals) < 5000:
            normality_stat, normality_p = stats.shapiro(residuals)
            normality_test = "Shapiro-Wilk"
        else:
            normality_result = stats.anderson(residuals, dist='norm')
            normality_stat = normality_result.statistic
            normality_p = "N/A (Anderson-Darling)"
            normality_test = "Anderson-Darling"
        
        print(f"{normality_test} normality test:")
        print(f"  Statistic: {normality_stat:.4f}")
        if normality_p != "N/A (Anderson-Darling)":
            print(f"  P-value: {normality_p:.4f}")
            normal_dist = "Yes" if normality_p > 0.05 else "No"
            print(f"  Normally distributed: {normal_dist}")
        
        # Residual statistics
        print(f"\nResidual Statistics:")
        print(f"  Mean: ${residuals.mean():,.0f}")
        print(f"  Std Dev: ${residuals.std():,.0f}")
        print(f"  Min: ${residuals.min():,.0f}")
        print(f"  Max: ${residuals.max():,.0f}")
        print(f"  25th percentile: ${np.percentile(residuals, 25):,.0f}")
        print(f"  75th percentile: ${np.percentile(residuals, 75):,.0f}")
    
    def create_feature_importance_plot(self, feature_importance_dict, save_path="results/"):
        """Create feature importance plots for tree-based models"""
        tree_models = {name: importance for name, importance in feature_importance_dict.items() 
                      if 'Random Forest' in name or 'Gradient Boosting' in name}
        
        if not tree_models:
            print("No tree-based models found for feature importance analysis.")
            return
        
        n_models = len(tree_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        for i, (name, importance_df) in enumerate(tree_models.items()):
            top_features = importance_df.head(15)
            
            bars = axes[i].barh(range(len(top_features)), top_features['importance'])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(name)
            axes[i].grid(True, alpha=0.3)
            axes[i].invert_yaxis()
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                axes[i].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plots saved to {save_path}feature_importance.png")
    
    def generate_final_report(self, results, save_path="results/"):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*70)
        print("FINAL MODEL EVALUATION REPORT")
        print("="*70)
        
        # Model performance summary
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Test_R2': [results[name]['test_r2'] for name in results.keys()],
            'Test_RMSE': [results[name]['test_rmse'] for name in results.keys()],
            'Test_MAE': [results[name]['test_mae'] for name in results.keys()],
            'CV_RMSE_Mean': [results[name]['cv_rmse_mean'] for name in results.keys()],
            'CV_RMSE_Std': [results[name]['cv_rmse_std'] for name in results.keys()]
        }).sort_values('Test_R2', ascending=False)
        
        print("\nModel Performance Summary:")
        print(performance_df.to_string(index=False, float_format='%.4f'))
        
        # Best model details
        best_model = performance_df.iloc[0]
        print(f"\nBest Performing Model: {best_model['Model']}")
        print(f"  R² Score: {best_model['Test_R2']:.4f} ({best_model['Test_R2']*100:.1f}% variance explained)")
        print(f"  RMSE: ${best_model['Test_RMSE']:,.0f}")
        print(f"  MAE: ${best_model['Test_MAE']:,.0f}")
        print(f"  Cross-validation RMSE: ${best_model['CV_RMSE_Mean']:,.0f} (±${best_model['CV_RMSE_Std']:,.0f})")
        
        # Save report to file
        with open(f"{save_path}evaluation_report.txt", "w") as f:
            f.write("MOVIE REVENUE PREDICTION - MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write("Model Performance Summary:\n")
            f.write(performance_df.to_string(index=False, float_format='%.4f'))
            f.write(f"\n\nBest Performing Model: {best_model['Model']}\n")
            f.write(f"R² Score: {best_model['Test_R2']:.4f}\n")
            f.write(f"RMSE: ${best_model['Test_RMSE']:,.0f}\n")
            f.write(f"MAE: ${best_model['Test_MAE']:,.0f}\n")
        
        print(f"\nDetailed report saved to {save_path}evaluation_report.txt")
        
        return performance_df
    
    def comprehensive_evaluation(self, results, feature_importance_dict=None, save_path="results/"):
        """Run complete model evaluation"""
        print("Running comprehensive model evaluation...")
        
        # Create all visualizations
        self.create_performance_comparison(results, save_path)
        self.create_prediction_plots(results, save_path)
        self.create_residual_analysis(results, save_path)
        
        if feature_importance_dict:
            self.create_feature_importance_plot(feature_importance_dict, save_path)
        
        # Generate final report
        performance_summary = self.generate_final_report(results, save_path)
        
        print("\n" + "="*70)
        print("MODEL EVALUATION COMPLETE")
        print("="*70)
        print("Generated files:")
        print(f"- {save_path}model_comparison.png")
        print(f"- {save_path}prediction_plots.png")
        print(f"- {save_path}residual_analysis.png")
        if feature_importance_dict:
            print(f"- {save_path}feature_importance.png")
        print(f"- {save_path}evaluation_report.txt")
        
        return performance_summary

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from model_training import ModelTrainer
    
    # Load data and train models
    preprocessor = DataPreprocessor()
    X, y, df, encoders = preprocessor.prepare_data()
    
    trainer = ModelTrainer()
    results, data_split = trainer.train_models(X, y)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    performance_summary = evaluator.comprehensive_evaluation(
        results, trainer.feature_importance
    )