"""
Feature Engineering Pipeline for Used Car Dynamic Pricing
==========================================================

This module handles feature creation and preprocessing transformations.
It creates a fitted preprocessor artifact for consistent train/inference.

Author: MLOps Team
Date: 2024
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/feature_engineering.log')
    ]
)
logger = logging.getLogger(__name__)


def load_processed_data(input_path: str) -> pd.DataFrame:
    """
    Load processed data from CSV file.
    
    Args:
        input_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    logger.info(f"Loading processed data from: {input_path}")
    
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Processed data file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features added
    """
    logger.info("Creating new features...")
    
    df_featured = df.copy()
    
    # Feature 1: Car Age (current year - year)
    current_year = datetime.now().year
    df_featured['car_age'] = current_year - df_featured['year']
    
    logger.info(f"✅ Created feature: car_age (range: {df_featured['car_age'].min()} to {df_featured['car_age'].max()} years)")
    
    # Feature 2: Mileage per year
    df_featured['mileage_per_year'] = df_featured['mileage'] / df_featured['car_age'].replace(0, 1)  # Avoid division by zero
    
    logger.info(f"✅ Created feature: mileage_per_year (mean: {df_featured['mileage_per_year'].mean():.0f} miles/year)")
    
    # Feature 3: Price per year (depreciation indicator)
    df_featured['price_per_year'] = df_featured['price'] / df_featured['car_age'].replace(0, 1)
    
    logger.info(f"✅ Created feature: price_per_year (mean: ${df_featured['price_per_year'].mean():.0f}/year)")
    
    logger.info(f"Total features after creation: {df_featured.shape[1]}")
    
    return df_featured


def define_feature_columns(df: pd.DataFrame, target_col: str = 'price'):
    """
    Define categorical and numerical feature columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target variable to exclude
        
    Returns:
        tuple: (categorical_features, numerical_features)
    """
    logger.info("Defining feature columns...")
    
    # Exclude target and identifier columns
    exclude_cols = {target_col, 'year'}  # year is used to create car_age, not as direct feature
    
    # Categorical features
    categorical_features = ['name', 'color', 'condition']
    
    # Numerical features
    numerical_features = ['mileage', 'car_age', 'mileage_per_year']
    
    # Validation: Check all defined features exist
    all_features = categorical_features + numerical_features
    missing_features = set(all_features) - set(df.columns)
    
    if missing_features:
        raise ValueError(f"Defined features not found in dataframe: {missing_features}")
    
    logger.info(f"✅ Categorical features ({len(categorical_features)}): {categorical_features}")
    logger.info(f"✅ Numerical features ({len(numerical_features)}): {numerical_features}")
    
    return categorical_features, numerical_features


def build_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """
    Build preprocessing pipeline using ColumnTransformer.
    
    Args:
        categorical_features (list): List of categorical column names
        numerical_features (list): List of numerical column names
        
    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    logger.info("Building preprocessing pipeline...")
    
    # Define transformers
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop',  # Drop columns not specified
        verbose_feature_names_out=True
    )
    
    logger.info("✅ Preprocessing pipeline built successfully")
    logger.info(f"   - Categorical: OneHotEncoder (handle_unknown='ignore')")
    logger.info(f"   - Numerical: StandardScaler")
    
    return preprocessor


def fit_and_transform(df: pd.DataFrame, 
                     preprocessor: ColumnTransformer,
                     categorical_features: list,
                     numerical_features: list,
                     target_col: str = 'price') -> tuple:
    """
    Fit preprocessor and transform features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        preprocessor (ColumnTransformer): Preprocessing pipeline
        categorical_features (list): Categorical column names
        numerical_features (list): Numerical column names
        target_col (str): Target column name
        
    Returns:
        tuple: (transformed_df, fitted_preprocessor, feature_names)
    """
    logger.info("Fitting and transforming features...")
    
    # Separate features and target
    feature_columns = categorical_features + numerical_features
    X = df[feature_columns]
    y = df[target_col]
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target variable shape: {y.shape}")
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    logger.info(f"✅ Features transformed: {X_transformed.shape}")
    
    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()
    logger.info(f"Total features after encoding: {len(feature_names)}")
    
    # Create dataframe with transformed features
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=df.index
    )
    
    # Add target variable
    X_transformed_df[target_col] = y.values
    
    logger.info(f"✅ Final feature dataframe shape: {X_transformed_df.shape}")
    
    return X_transformed_df, preprocessor, feature_names


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str) -> None:
    """
    Save fitted preprocessor to disk using joblib.
    
    Args:
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline
        output_path (str): Path where preprocessor will be saved
    """
    logger.info(f"Saving preprocessor to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessor
    joblib.dump(preprocessor, output_path)
    
    file_size = output_file.stat().st_size / 1024  # Size in KB
    logger.info(f"✅ Preprocessor saved successfully ({file_size:.2f} KB)")


def save_featured_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save transformed features to CSV file.
    
    Args:
        df (pd.DataFrame): Transformed feature dataframe
        output_path (str): Path where CSV will be saved
    """
    logger.info(f"Saving featured data to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Featured data saved successfully")
    logger.info(f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")


def validate_outputs(featured_df: pd.DataFrame, 
                    preprocessor: ColumnTransformer,
                    target_col: str = 'price') -> bool:
    """
    Validate the feature engineering outputs.
    
    Args:
        featured_df (pd.DataFrame): Transformed feature dataframe
        preprocessor (ColumnTransformer): Fitted preprocessor
        target_col (str): Target column name
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating feature engineering outputs...")
    
    validations = []
    
    # Check target column exists
    if target_col not in featured_df.columns:
        logger.error(f"Target column '{target_col}' not found in featured data")
        validations.append(False)
    else:
        logger.info(f"✅ Target column '{target_col}' present")
        validations.append(True)
    
    # Check for NaN values
    nan_count = featured_df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠️  {nan_count} NaN values found in featured data")
        validations.append(False)
    else:
        logger.info(f"✅ No NaN values in featured data")
        validations.append(True)
    
    # Check for infinite values
    inf_count = np.isinf(featured_df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logger.warning(f"⚠️  {inf_count} infinite values found in featured data")
        validations.append(False)
    else:
        logger.info(f"✅ No infinite values in featured data")
        validations.append(True)
    
    # Check preprocessor is fitted
    try:
        # Check if preprocessor has been fitted (has transformers_ attribute)
        if hasattr(preprocessor, 'transformers_'):
            logger.info(f"✅ Preprocessor is properly fitted")
            validations.append(True)
        else:
            logger.error(f"Preprocessor is not fitted")
            validations.append(False)
    except Exception as e:
        logger.error(f"Error validating preprocessor: {str(e)}")
        validations.append(False)
    
    # Check feature dtypes are numeric
    feature_cols = [col for col in featured_df.columns if col != target_col]
    non_numeric = featured_df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.error(f"Non-numeric feature columns found: {non_numeric}")
        validations.append(False)
    else:
        logger.info(f"✅ All feature columns are numeric")
        validations.append(True)
    
    all_passed = all(validations)
    if all_passed:
        logger.info("All validations passed!")
    else:
        logger.error("Some validations failed")
    
    return all_passed


def main(input_path: str, 
         output_data_path: str,
         output_preprocessor_path: str,
         target_col: str = 'price') -> None:
    """
    Main feature engineering pipeline.
    
    Args:
        input_path (str): Path to processed CSV file
        output_data_path (str): Path to save featured CSV file
        output_preprocessor_path (str): Path to save preprocessor pickle file
        target_col (str): Name of the target variable (default: 'price')
    """
    logger.info("="*80)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("="*80)
    
    try:
        # 1. Load processed data
        df = load_processed_data(input_path)
        
        # 2. Create new features
        df_featured = create_features(df)
        
        # 3. Define feature columns
        categorical_features, numerical_features = define_feature_columns(
            df_featured, 
            target_col=target_col
        )
        
        # 4. Build preprocessor
        preprocessor = build_preprocessor(categorical_features, numerical_features)
        
        # 5. Fit and transform
        featured_df, fitted_preprocessor, feature_names = fit_and_transform(
            df_featured,
            preprocessor,
            categorical_features,
            numerical_features,
            target_col=target_col
        )
        
        # 6. Validate outputs
        if not validate_outputs(featured_df, fitted_preprocessor, target_col=target_col):
            logger.error("Feature engineering validation failed. Please review the logs.")
            sys.exit(1)
        
        # 7. Save preprocessor
        save_preprocessor(fitted_preprocessor, output_preprocessor_path)
        
        # 8. Save featured data
        save_featured_data(featured_df, output_data_path)
        
        # 9. Summary statistics
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*80)
        logger.info(f"Input rows: {len(df)}")
        logger.info(f"Output rows: {len(featured_df)}")
        logger.info(f"Original features: {len(df.columns)}")
        logger.info(f"Engineered features: {len(featured_df.columns) - 1}")
        logger.info(f"Feature names sample: {list(feature_names[:5])}")
        logger.info(f"Target variable: {target_col}")
        logger.info(f"Target range: ${featured_df[target_col].min():.0f} - ${featured_df[target_col].max():.0f}")
        
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline for car pricing model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/features/build_features.py \\
    --input data/processed/processed_cars.csv \\
    --output-data data/processed/featured_cars.csv \\
    --output-preprocessor models/preprocessor.pkl
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to processed CSV file (e.g., data/processed/processed_cars.csv)'
    )
    
    parser.add_argument(
        '--output-data',
        type=str,
        required=True,
        help='Path to save featured CSV file (e.g., data/processed/featured_cars.csv)'
    )
    
    parser.add_argument(
        '--output-preprocessor',
        type=str,
        required=True,
        help='Path to save preprocessor pickle file (e.g., models/preprocessor.pkl)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='price',
        help='Name of the target variable (default: price)'
    )
    
    args = parser.parse_args()
    
    main(
        args.input,
        args.output_data,
        args.output_preprocessor,
        args.target
    )
