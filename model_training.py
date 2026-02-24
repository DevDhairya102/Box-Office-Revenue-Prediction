import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def initialize_models(self):
        """Initialize different regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        return models
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for key models"""
        print("Performing hyperparameter tuning...")
        
        # Random Forest tuning (reduced for speed)
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 5]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_params,
            cv=2,  # Reduced from 3 to 2
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        
        # Gradient Boosting tuning (reduced for speed)
        gb_params = {
            'n_estimators': [100],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params,
            cv=2,  # Reduced from 3 to 2
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        
        tuned_models = {
            'Random Forest (Tuned)': rf_grid.best_estimator_,
            'Gradient Boosting (Tuned)': gb_grid.best_estimator_
        }
        
        print("Hyperparameter tuning completed.")
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        print(f"Best Gradient Boosting params: {gb_grid.best_params_}")
        
        return tuned_models
    
    def train_models(self, X, y, test_size=0.2, perform_tuning=True):
        """Train multiple models with cross-validation"""
        print("Starting model training process...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set size: {X_train.shape[0]:,} movies")
        print(f"Test set size: {X_test.shape[0]:,} movies")
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = self.initialize_models()
        
        # Add hyperparameter tuned models
        if perform_tuning:
            tuned_models = self.hyperparameter_tuning(X_train, y_train)
            models.update(tuned_models)
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for linear models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                X_train_use, X_test_use = X_train_scaled, X_test_scaled
            else:
                X_train_use, X_test_use = X_train, X_test
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_use, y_train, 
                cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Train on full training set
            model.fit(X_train_use, y_train)
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate additional metrics (fixed MAPE calculation)
            # Only calculate MAPE for movies with revenue > $1M to avoid division by tiny values
            mask = y_test > 1000000
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            else:
                mape = np.nan
            
            results[name] = {
                'model': model,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'test_r2': r2,
                'test_rmse': rmse,
                'test_mae': mae,
                'y_test': y_test,
                'y_pred': y_pred,
                'cv_scores': cv_rmse
            }
            
            print(f"CV RMSE: {cv_rmse.mean():,.0f} (±{cv_rmse.std():,.0f})")
            print(f"Test R²: {r2:.4f}")
            print(f"Test RMSE: {rmse:,.0f}")
            print(f"Test MAE: {mae:,.0f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance_df
                
                print(f"Top 5 important features:")
                for idx, row in importance_df.head().iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        
        return results, (X_train, X_test, y_train, y_test)
    
    def statistical_significance_test(self):
        """Perform statistical tests to compare model performance"""
        from scipy import stats
        
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        model_names = list(self.results.keys())
        cv_scores_dict = {name: self.results[name]['cv_scores'] for name in model_names}
        
        # Perform paired t-tests between models
        best_model = max(model_names, key=lambda x: self.results[x]['test_r2'])
        print(f"Best performing model: {best_model} (R² = {self.results[best_model]['test_r2']:.4f})")
        
        print(f"\nPaired t-tests comparing {best_model} with other models:")
        
        for name in model_names:
            if name != best_model:
                t_stat, p_value = stats.ttest_rel(
                    cv_scores_dict[best_model], 
                    cv_scores_dict[name]
                )
                significance = "significant" if p_value < 0.05 else "not significant"
                print(f"{name}: t={t_stat:.3f}, p={p_value:.4f} ({significance})")
    
    def save_models(self, save_dir="models/"):
        """Save trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        best_model = self.results[best_model_name]['model']
        best_r2 = self.results[best_model_name]['test_r2']
        best_rmse = self.results[best_model_name]['test_rmse']
        
        joblib.dump(best_model, f"{save_dir}best_model.pkl")
        joblib.dump(self.scaler, f"{save_dir}scaler.pkl")
        
        # Save model metadata for UI
        model_info = {
            'best_model_name': best_model_name,
            'r2_score': best_r2,
            'rmse': best_rmse
        }
        
        import json
        with open(f"{save_dir}model_info.json", "w") as f:
            json.dump(model_info, f)
        
        # Save all models
        for name, model in self.models.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            joblib.dump(model, f"{save_dir}{safe_name}.pkl")
        
        print(f"\nModels saved to {save_dir}")
        print(f"Best model ({best_model_name}) saved as best_model.pkl")
        
        return best_model_name

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load and prepare data
    preprocessor = DataPreprocessor()
    X, y, df, encoders = preprocessor.prepare_data()
    
    # Train models
    trainer = ModelTrainer()
    results, data_split = trainer.train_models(X, y, perform_tuning=True)
    
    # Statistical significance testing
    trainer.statistical_significance_test()
    
    # Save models
    best_model = trainer.save_models()