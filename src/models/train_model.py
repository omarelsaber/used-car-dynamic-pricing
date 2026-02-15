"""
Model Training Pipeline for Used Car Dynamic Pricing
====================================================

This module handles model training with MLflow experiment tracking.
Trains a RandomForestRegressor and logs all artifacts, metrics, and parameters.

Author: MLOps Team
Date: 2024
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/model_training.log')
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
    
    logger.info("MLflow setup completed")


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
    logger.info(f"Data loaded successfully: {df.shape[0]} rows x {df.shape[1]} columns")
    
    return df


def prepare_train_test_split(df: pd.DataFrame, 
                             target_col: str = 'price',
                             test_size: float = 0.2,
                             random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Featured dataframe
        target_col (str): Name of the target variable
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing train/test split...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target variable: {target_col}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Train/test split completed:")
    logger.info(f"   Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    logger.info(f"   Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       n_estimators: int = 100,
                       max_depth: int = 10,
                       random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the tree
        random_state (int): Random seed for reproducibility
        
    Returns:
        RandomForestRegressor: Trained model
    """
    logger.info("Training Random Forest Regressor...")
    logger.info(f"Hyperparameters:")
    logger.info(f"   n_estimators: {n_estimators}")
    logger.info(f"   max_depth: {max_depth}")
    logger.info(f"   random_state: {random_state}")
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    
    return model


def evaluate_model(model: RandomForestRegressor,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> dict:
    """
    Evaluate model performance on train and test sets.
    
    Args:
        model (RandomForestRegressor): Trained model
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Log metrics
    logger.info("="*80)
    logger.info("MODEL PERFORMANCE METRICS")
    logger.info("="*80)
    logger.info("TRAINING SET:")
    logger.info(f"   MAE:  ${train_mae:,.2f}")
    logger.info(f"   RMSE: ${train_rmse:,.2f}")
    logger.info(f"   R2:   {train_r2:.4f}")
    logger.info("")
    logger.info("TEST SET:")
    logger.info(f"   MAE:  ${test_mae:,.2f}")
    logger.info(f"   RMSE: ${test_rmse:,.2f}")
    logger.info(f"   R2:   {test_r2:.4f}")
    logger.info("="*80)
    
    # Check for overfitting
    r2_diff = train_r2 - test_r2
    if r2_diff > 0.1:
        logger.warning(f"Possible overfitting detected (R2 diff: {r2_diff:.4f})")
    else:
        logger.info(f"Model generalization looks good (R2 diff: {r2_diff:.4f})")
    
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


def log_to_mlflow(model: RandomForestRegressor,
                 params: dict,
                 metrics: dict,
                 X_train: pd.DataFrame) -> str:
    """
    Log model, parameters, and metrics to MLflow.
    
    Args:
        model (RandomForestRegressor): Trained model
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
        
        # Log model
        logger.info("Logging model artifact...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="UsedCarPricePredictor"
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
        
        run_id = run.info.run_id
        logger.info(f"MLflow logging completed. Run ID: {run_id}")
        
        return run_id


def save_model(model: RandomForestRegressor, output_path: str) -> None:
    """
    Save trained model to disk using joblib.
    
    Args:
        model (RandomForestRegressor): Trained model
        output_path (str): Path where model will be saved
    """
    logger.info(f"Saving model to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    
    file_size = output_file.stat().st_size / 1024 / 1024  # Size in MB
    logger.info(f"Model saved successfully ({file_size:.2f} MB)")


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
    
    logger.info(f"Metrics saved successfully")


def main(input_path: str,
         output_model_path: str,
         output_metrics_path: str,
         n_estimators: int = 100,
         max_depth: int = 10,
         test_size: float = 0.2,
         random_state: int = 42,
         target_col: str = 'price') -> None:
    """
    Main model training pipeline with MLflow tracking.
    
    Args:
        input_path (str): Path to featured CSV file
        output_model_path (str): Path to save trained model
        output_metrics_path (str): Path to save metrics JSON
        n_estimators (int): Number of trees in random forest
        max_depth (int): Maximum depth of trees
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        target_col (str): Name of target variable
    """
    logger.info("="*80)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    try:
        # 1. Setup MLflow
        setup_mlflow()
        
        # 2. Load featured data
        df = load_featured_data(input_path)
        
        # 3. Prepare train/test split
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state
        )
        
        # 4. Train model
        model = train_random_forest(
            X_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        # 5. Evaluate model
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # 6. Prepare parameters for logging
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'test_size': test_size,
            'random_state': random_state,
            'model_type': 'RandomForestRegressor',
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
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Model: RandomForestRegressor")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Test MAE: ${metrics['test_mae']:,.2f}")
        logger.info(f"Test RMSE: ${metrics['test_rmse']:,.2f}")
        logger.info(f"Test R2: {metrics['test_r2']:.4f}")
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info("="*80)
        logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for car price prediction with MLflow tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/models/train_model.py \\
    --input data/processed/featured_cars.csv \\
    --output-model models/model.pkl \\
    --output-metrics metrics/scores.json
  
  # With hyperparameters
  python src/models/train_model.py \\
    --input data/processed/featured_cars.csv \\
    --output-model models/model.pkl \\
    --output-metrics metrics/scores.json \\
    --n-estimators 200 \\
    --max-depth 15
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
        default=100,
        help='Number of trees in random forest (default: 100)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=10,
        help='Maximum depth of trees (default: 10)'
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
        args.max_depth,
        args.test_size,
        args.random_state,
        args.target
    )
