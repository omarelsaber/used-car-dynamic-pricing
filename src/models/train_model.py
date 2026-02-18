"""
Model Training Pipeline for Used Car Dynamic Pricing
====================================================

This module handles model training with MLflow experiment tracking.
Uses XGBoost with log-transformed target variable for better performance.

Author: Omar Elsaber
Date: Feb 2026
Version: 2.0 (XGBoost + Log Transform)
"""

import argparse
import io
import logging
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost


# Configure logging (ensure logs dir exists; use UTF-8 for Windows console)
Path("logs").mkdir(parents=True, exist_ok=True)
_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'buffer') else sys.stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(_stdout),
        logging.FileHandler('logs/model_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str = "mlruns", experiment_name: str = "Used_Car_Price_Prediction") -> None:
    """
    Setup MLflow tracking URI and experiment.
    
    Args:
        tracking_uri (str): Path to MLflow tracking directory
        experiment_name (str): Name of the MLflow experiment
    """
    logger.info("Setting up MLflow...")
    
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    
    logger.info("✅ MLflow setup completed")


def load_featured_data(input_path: str) -> pd.DataFrame:
    """
    Load featured data from CSV file.
    
    Args:
        input_path (str): Path to the featured CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    logger.info(f"Loading featured data from: {input_path}")
    
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Featured data file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def prepare_train_test_split(df: pd.DataFrame, 
                             target_col: str = 'price',
                             test_size: float = 0.2,
                             random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets with LOG TRANSFORMATION.
    
    IMPORTANT: Applies log1p transformation to target variable for better
    model performance on skewed price distributions.
    
    Args:
        df (pd.DataFrame): Featured dataframe
        target_col (str): Name of the target variable
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train_log, y_test_log)
    """
    logger.info("Preparing train/test split...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target variable: {target_col}")
    logger.info(f"Target range (original): ${y.min():,.2f} - ${y.max():,.2f}")
    
    # Split data FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Apply LOG TRANSFORMATION to target variable
    logger.info("🔄 Applying log1p transformation to target variable...")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    logger.info(f"Target range (log-transformed): {y_train_log.min():.4f} - {y_train_log.max():.4f}")
    logger.info("✅ Log transformation applied - this improves model performance on skewed data!")
    
    logger.info(f"✅ Train/test split completed:")
    logger.info(f"   Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    logger.info(f"   Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train_log, y_test_log


def train_xgboost_model(X_train: pd.DataFrame,
                        y_train_log: pd.Series,
                        n_estimators: int = 500,
                        learning_rate: float = 0.05,
                        max_depth: int = 4,
                        reg_lambda: float = 5.0,
                        reg_alpha: float = 1.0,
                        subsample: float = 0.8,
                        colsample_bytree: float = 0.8,
                        min_child_weight: int = 3,
                        gamma: float = 0.1,
                        random_state: int = 42) -> xgb.XGBRegressor:
    """
    Train an XGBoost Regressor with anti-overfitting regularization.
    
    NEW: Added regularization parameters based on Kaggle best practices.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train_log (pd.Series): Log-transformed training target
        n_estimators (int): Number of boosting rounds
        learning_rate (float): Learning rate (eta)
        max_depth (int): Maximum depth of trees
        reg_lambda (float): L2 regularization (weight decay)
        reg_alpha (float): L1 regularization (lasso)
        subsample (float): Fraction of samples per tree
        colsample_bytree (float): Fraction of features per tree
        min_child_weight (int): Minimum sum of weights in child
        gamma (float): Minimum loss reduction for split
        random_state (int): Random seed for reproducibility
        
    Returns:
        xgb.XGBRegressor: Trained model
    """
    logger.info("Training XGBoost Regressor with Anti-Overfitting Regularization (V3)...")
    logger.info(f"Hyperparameters:")
    logger.info(f"   Tree Structure:")
    logger.info(f"     ├─ n_estimators: {n_estimators}")
    logger.info(f"     ├─ max_depth: {max_depth}")
    logger.info(f"     └─ learning_rate: {learning_rate}")
    logger.info(f"   Regularization (Anti-Overfitting):")
    logger.info(f"     ├─ reg_lambda (L2): {reg_lambda}")
    logger.info(f"     ├─ reg_alpha (L1): {reg_alpha}")
    logger.info(f"     ├─ gamma: {gamma}")
    logger.info(f"     └─ min_child_weight: {min_child_weight}")
    logger.info(f"   Sampling (Anti-Overfitting):")
    logger.info(f"     ├─ subsample: {subsample}")
    logger.info(f"     └─ colsample_bytree: {colsample_bytree}")
    
    # Initialize model with ALL regularization parameters
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        reg_lambda=reg_lambda,           # NEW
        reg_alpha=reg_alpha,             # NEW
        subsample=subsample,             # NEW
        colsample_bytree=colsample_bytree,  # NEW
        min_child_weight=min_child_weight,  # NEW
        gamma=gamma,                     # NEW
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        objective='reg:squarederror',
        tree_method='hist',  # Faster histogram-based algorithm
        verbosity=0
    )
    
    # Train model
    logger.info("🚀 Training in progress...")
    model.fit(X_train, y_train_log)
    
    logger.info("✅ Model training completed")
    
    return model


def evaluate_model(model: xgb.XGBRegressor,
                  X_train: pd.DataFrame,
                  y_train_log: pd.Series,
                  X_test: pd.DataFrame,
                  y_test_log: pd.Series) -> dict:
    """
    Evaluate model performance on train and test sets.
    
    IMPORTANT: Applies INVERSE transformation (expm1) to get predictions
    back to original scale before calculating metrics.
    
    Args:
        model (xgb.XGBRegressor): Trained model
        X_train (pd.DataFrame): Training features
        y_train_log (pd.Series): Log-transformed training target
        X_test (pd.DataFrame): Test features
        y_test_log (pd.Series): Log-transformed test target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    # Predictions on LOG scale
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)
    
    # INVERSE TRANSFORM: Convert predictions back to ORIGINAL SCALE (dollars)
    logger.info("🔄 Applying inverse transformation (expm1) to predictions...")
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)
    
    # Convert targets back to original scale for metrics
    y_train_original = np.expm1(y_train_log)
    y_test_original = np.expm1(y_test_log)
    
    # Calculate metrics on ORIGINAL SCALE (real dollars)
    train_mae = mean_absolute_error(y_train_original, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred))
    train_r2 = r2_score(y_train_original, y_train_pred)
    
    test_mae = mean_absolute_error(y_test_original, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred))
    test_r2 = r2_score(y_test_original, y_test_pred)
    
    # Log metrics
    logger.info("="*80)
    logger.info("MODEL PERFORMANCE METRICS (Original Scale - Real Dollars)")
    logger.info("="*80)
    logger.info("TRAINING SET:")
    logger.info(f"   MAE:  ${train_mae:,.2f}")
    logger.info(f"   RMSE: ${train_rmse:,.2f}")
    logger.info(f"   R²:   {train_r2:.4f}")
    logger.info("")
    logger.info("TEST SET:")
    logger.info(f"   MAE:  ${test_mae:,.2f}")
    logger.info(f"   RMSE: ${test_rmse:,.2f}")
    logger.info(f"   R²:   {test_r2:.4f}")
    logger.info("="*80)
    
    # Check for overfitting
    r2_diff = train_r2 - test_r2
    if r2_diff > 0.1:
        logger.warning(f"⚠️  Possible overfitting detected (R² diff: {r2_diff:.4f})")
    else:
        logger.info(f"✅ Model generalization looks good (R² diff: {r2_diff:.4f})")
    
    # Compare with previous baseline
    if test_r2 > 0.80:
        logger.info("🎉 EXCELLENT! R² > 0.80 - Model performance is production-ready!")
    elif test_r2 > 0.70:
        logger.info("👍 GOOD! R² > 0.70 - Model performance is acceptable")
    else:
        logger.warning("⚠️  R² < 0.70 - Consider hyperparameter tuning")
    
    metrics = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'mae': test_mae,  # Primary metric for DVC
        'rmse': test_rmse,  # Primary metric for DVC
        'r2': test_r2  # Primary metric for DVC
    }
    
    return metrics


def log_to_mlflow(model: xgb.XGBRegressor,
                 params: dict,
                 metrics: dict,
                 X_train: pd.DataFrame) -> str:
    """
    Log model, parameters, and metrics to MLflow.
    
    Args:
        model (xgb.XGBRegressor): Trained model
        params (dict): Model hyperparameters
        metrics (dict): Evaluation metrics
        X_train (pd.DataFrame): Training features (for signature)
        
    Returns:
        str: MLflow run ID
    """
    logger.info("Logging to MLflow...")
    
    with mlflow.start_run() as run:
        # Log parameters
        logger.info("Logging parameters...")
        mlflow.log_params(params)
        
        # Log metrics
        logger.info("Logging metrics...")
        mlflow.log_metrics(metrics)
        
        # Log model using MLflow's XGBoost flavor
        logger.info("Logging XGBoost model artifact...")
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="UsedCarPricePredictor_XGBoost"
        )
        
        # Log feature importance
        logger.info("Logging feature importance...")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance as artifact
        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Log top 10 important features
        top_features = feature_importance.head(10)
        logger.info("Top 10 Important Features:")
        for idx, row in top_features.iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Log model type as tag
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("target_transform", "log1p")
        
        run_id = run.info.run_id
        logger.info(f"✅ MLflow logging completed. Run ID: {run_id}")
        
        return run_id


def save_model(model: xgb.XGBRegressor, output_path: str) -> None:
    """
    Save trained model to disk using joblib.
    
    Args:
        model (xgb.XGBRegressor): Trained model
        output_path (str): Path where model will be saved
    """
    logger.info(f"Saving model to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    
    file_size = output_file.stat().st_size / 1024 / 1024  # Size in MB
    logger.info(f"✅ Model saved successfully ({file_size:.2f} MB)")


def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Save metrics to JSON file for DVC tracking.
    
    Args:
        metrics (dict): Evaluation metrics
        output_path (str): Path where metrics will be saved
    """
    logger.info(f"Saving metrics to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save only the primary metrics for DVC
    dvc_metrics = {
        'mae': metrics['test_mae'],
        'rmse': metrics['test_rmse'],
        'r2': metrics['test_r2']
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(dvc_metrics, f, indent=4)
    
    logger.info(f"✅ Metrics saved successfully")


def main(input_path: str,
         output_model_path: str,
         output_metrics_path: str,
         n_estimators: int = 500,
         learning_rate: float = 0.05,
         max_depth: int = 4,
         reg_lambda: float = 5.0,
         reg_alpha: float = 1.0,
         subsample: float = 0.8,
         colsample_bytree: float = 0.8,
         min_child_weight: int = 3,
         gamma: float = 0.1,
         test_size: float = 0.2,
         random_state: int = 42,
         target_col: str = 'price') -> None:
    """
    Main model training pipeline with XGBoost and log transformation (V3 - Anti-Overfitting).
    
    Args:
        input_path (str): Path to featured CSV file
        output_model_path (str): Path to save trained model
        output_metrics_path (str): Path to save metrics JSON
        n_estimators (int): Number of boosting rounds
        learning_rate (float): Learning rate
        max_depth (int): Maximum depth of trees
        reg_lambda (float): L2 regularization weight
        reg_alpha (float): L1 regularization weight
        subsample (float): Fraction of samples per tree
        colsample_bytree (float): Fraction of features per tree
        min_child_weight (int): Minimum sum of weights in child
        gamma (float): Minimum loss reduction for split
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        target_col (str): Name of target variable
    """
    logger.info("="*80)
    logger.info("STARTING MODEL TRAINING PIPELINE V3.0 (Anti-Overfitting)")
    logger.info("MODEL: XGBoost with Anti-Overfitting Regularization & Log Transform")
    logger.info("="*80)
    
    try:
        # 1. Setup MLflow
        setup_mlflow()
        
        # 2. Load featured data
        df = load_featured_data(input_path)
        
        # 3. Prepare train/test split WITH LOG TRANSFORMATION
        X_train, X_test, y_train_log, y_test_log = prepare_train_test_split(
            df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state
        )
        
        # 4. Train XGBoost model with V3 anti-overfitting regularization
        model = train_xgboost_model(
            X_train,
            y_train_log,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=random_state
        )
        
        # 5. Evaluate model (with inverse transform)
        metrics = evaluate_model(model, X_train, y_train_log, X_test, y_test_log)
        
        # 6. Prepare parameters for logging
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'test_size': test_size,
            'random_state': random_state,
            'model_type': 'XGBoost',
            'target_transform': 'log1p',
            'objective': 'reg:squarederror',
            'n_features': X_train.shape[1],
            'n_samples_train': X_train.shape[0],
            'n_samples_test': X_test.shape[0]
        }
        
        # 7. Log to MLflow
        run_id = log_to_mlflow(model, params, metrics, X_train)
        
        # 8. Save model locally
        save_model(model, output_model_path)
        
        # 9. Save metrics
        save_metrics(metrics, output_metrics_path)
        
        logger.info("="*80)
        logger.info("TRAINING SUMMARY (V3.0 - Anti-Overfitting)")
        logger.info("="*80)
        logger.info(f"Model: XGBoost Regressor")
        logger.info(f"Target Transform: log1p (improves performance on skewed data)")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Test MAE: ${metrics['test_mae']:,.2f}")
        logger.info(f"Test RMSE: ${metrics['test_rmse']:,.2f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info("="*80)
        logger.info("✅ MODEL TRAINING PIPELINE V3.0 COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost model with log transformation for car price prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/models/train_model.py \\
    --input data/processed/featured_cars.csv \\
    --output-model models/model.pkl \\
    --output-metrics metrics/scores.json
  
  # With XGBoost hyperparameters
  python src/models/train_model.py \\
    --input data/processed/featured_cars.csv \\
    --output-model models/model.pkl \\
    --output-metrics metrics/scores.json \\
    --n-estimators 1000 \\
    --learning-rate 0.05 \\
    --max-depth 6
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to featured CSV file (e.g., data/processed/featured_cars.csv)'
    )
    
    parser.add_argument(
        '--output-model',
        type=str,
        required=True,
        help='Path to save trained model (e.g., models/model.pkl)'
    )
    
    parser.add_argument(
        '--output-metrics',
        type=str,
        required=True,
        help='Path to save metrics JSON (e.g., metrics/scores.json)'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=1000,
        help='Number of boosting rounds (default: 1000)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.05,
        help='Learning rate (default: 0.05)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=4,
        help='Maximum depth of trees (default: 4)'
    )
    
    parser.add_argument(
        '--reg-lambda',
        type=float,
        default=5.0,
        help='L2 regularization weight for anti-overfitting (default: 5.0)'
    )
    
    parser.add_argument(
        '--reg-alpha',
        type=float,
        default=1.0,
        help='L1 regularization weight for anti-overfitting (default: 1.0)'
    )
    
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Fraction of samples per tree for anti-overfitting (default: 0.8)'
    )
    
    parser.add_argument(
        '--colsample-bytree',
        type=float,
        default=0.8,
        help='Fraction of features per tree for anti-overfitting (default: 0.8)'
    )
    
    parser.add_argument(
        '--min-child-weight',
        type=int,
        default=3,
        help='Minimum sum of weights in child for anti-overfitting (default: 3)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Minimum loss reduction for split for anti-overfitting (default: 0.1)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='price',
        help='Name of target variable (default: price)'
    )
    
    args = parser.parse_args()
    
    main(
        args.input,
        args.output_model,
        args.output_metrics,
        args.n_estimators,
        args.learning_rate,
        args.max_depth,
        args.reg_lambda,
        args.reg_alpha,
        args.subsample,
        args.colsample_bytree,
        args.min_child_weight,
        args.gamma,
        args.test_size,
        args.random_state,
        args.target
    )