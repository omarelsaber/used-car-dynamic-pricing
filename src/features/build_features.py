"""
Feature Engineering Pipeline for Cardekho V3 Dataset
===================================================

Simplified feature engineering for the rich Cardekho dataset.
No complex text parsing needed - dataset has explicit features!

Author: Omar Elsaber  
Date: Feb 2026
Version: V5.0 (Cardekho Dataset)
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
    """Load processed data from CSV file."""
    logger.info(f"Loading processed data from: {input_path}")
    
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Processed data file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def extract_brand(name: str) -> str:
    """
    Extract car brand (first word) from car name.
    
    Args:
        name (str): Full car name
        
    Returns:
        str: Extracted brand
    """
    if pd.isna(name) or name.strip() == "":
        return "Unknown"
    
    brand = name.strip().split()[0]
    return brand


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from existing columns.
    
    For Cardekho dataset, most features are already explicit.
    We just add:
    - brand (from name)
    - car_age (from year)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    logger.info("="*80)
    logger.info("CREATING FEATURES - CARDEKHO V5.0")
    logger.info("="*80)
    
    df_featured = df.copy()
    
    # Extract brand
    logger.info("Extracting brand from name...")
    df_featured['brand'] = df_featured['name'].apply(extract_brand)
    
    unique_brands = df_featured['brand'].nunique()
    logger.info(f"✅ Brand extracted: {unique_brands} unique brands")
    
    top_brands = df_featured['brand'].value_counts().head(10)
    logger.info(f"Top 10 brands:")
    for brand, count in top_brands.items():
        logger.info(f"  {brand}: {count}")
    
    # Calculate car age
    current_year = datetime.now().year
    df_featured['car_age'] = current_year - df_featured['year']
    
    logger.info(f"✅ Car age calculated (range: {df_featured['car_age'].min()}-{df_featured['car_age'].max()} years)")
    
    # Log feature summary
    logger.info("="*80)
    logger.info("FEATURE SUMMARY")
    logger.info("="*80)
    logger.info(f"Categorical features available:")
    logger.info(f"  - brand: {df_featured['brand'].nunique()} unique")
    logger.info(f"  - fuel: {df_featured['fuel'].nunique()} unique")
    logger.info(f"  - seller_type: {df_featured['seller_type'].nunique()} unique")
    logger.info(f"  - transmission: {df_featured['transmission'].nunique()} unique")
    logger.info(f"  - owner: {df_featured['owner'].nunique()} unique")
    logger.info(f"Numerical features available:")
    logger.info(f"  - year, mileage_driven, mileage, engine, max_power, seats, car_age")
    logger.info("="*80)
    
    return df_featured


def define_feature_columns(df: pd.DataFrame, target_col: str = 'price'):
    """
    Define categorical and numerical feature columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target variable
        
    Returns:
        tuple: (categorical_features, numerical_features)
    """
    logger.info("Defining feature columns...")
    
    # ========================================================================
    # CATEGORICAL FEATURES
    # ========================================================================
    categorical_features = [
        'brand',         # Extracted from name
        'fuel',          # Petrol, Diesel, CNG, LPG, Electric
        'seller_type',   # Individual, Dealer, Trustmark Dealer
        'transmission',  # Manual, Automatic
        'owner'          # First Owner, Second Owner, etc.
    ]
    
    logger.info(f"Categorical features:")
    for feat in categorical_features:
        logger.info(f"   - {feat}: {df[feat].nunique()} unique values")
    
    # ========================================================================
    # NUMERICAL FEATURES
    # ========================================================================
    numerical_features = [
        'year',            # Manufacturing year
        'mileage_driven',  # Total kilometers driven
        'mileage',         # Fuel efficiency (kmpl) - PARSED
        'engine',          # Engine size (CC) - PARSED
        'max_power',       # Maximum power (bhp) - PARSED
        'seats',           # Number of seats
        'car_age'          # Calculated: current_year - year
    ]
    
    logger.info(f"Numerical features:")
    for feat in numerical_features:
        if feat in df.columns:
            logger.info(f"   - {feat}: mean={df[feat].mean():.2f}, range=[{df[feat].min():.0f}, {df[feat].max():.0f}]")
    
    # Validation
    all_features = categorical_features + numerical_features
    missing_features = set(all_features) - set(df.columns)
    
    if missing_features:
        raise ValueError(f"Defined features not found in dataframe: {missing_features}")
    
    # Estimate total features
    estimated_features = sum(df[f].nunique() for f in categorical_features) + len(numerical_features)
    
    logger.info(f"✅ Total features defined: {len(all_features)}")
    logger.info(f"Estimated features after encoding: ~{estimated_features}")
    logger.info(f"Feature-to-sample ratio: {estimated_features}/{len(df)} = {estimated_features/len(df)*100:.1f}%")
    
    return categorical_features, numerical_features


def build_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """Build preprocessing pipeline."""
    logger.info("Building preprocessing pipeline...")
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
    
    logger.info("✅ Preprocessing pipeline built successfully")
    
    return preprocessor


def fit_and_transform(df: pd.DataFrame, 
                     preprocessor: ColumnTransformer,
                     categorical_features: list,
                     numerical_features: list,
                     target_col: str = 'price') -> tuple:
    """Fit preprocessor and transform features."""
    logger.info("Fitting and transforming features...")
    
    feature_columns = categorical_features + numerical_features
    X = df[feature_columns]
    y = df[target_col]
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target variable shape: {y.shape}")
    
    X_transformed = preprocessor.fit_transform(X)
    
    logger.info(f"✅ Features transformed: {X_transformed.shape}")
    
    feature_names = preprocessor.get_feature_names_out()
    
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=df.index
    )
    
    X_transformed_df[target_col] = y.values
    
    logger.info(f"✅ Final dataframe shape: {X_transformed_df.shape}")
    
    return X_transformed_df, preprocessor, feature_names


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str) -> None:
    """Save fitted preprocessor."""
    logger.info(f"Saving preprocessor to: {output_path}")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, output_path)
    
    file_size = output_file.stat().st_size / 1024
    logger.info(f"✅ Preprocessor saved ({file_size:.2f} KB)")


def save_featured_data(df: pd.DataFrame, output_path: str) -> None:
    """Save transformed features."""
    logger.info(f"Saving featured data to: {output_path}")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Featured data saved: {df.shape[0]} rows × {df.shape[1]} columns")


def validate_outputs(featured_df: pd.DataFrame, 
                    preprocessor: ColumnTransformer,
                    target_col: str = 'price') -> bool:
    """Validate outputs."""
    logger.info("Validating outputs...")
    
    validations = []
    
    if target_col not in featured_df.columns:
        logger.error(f"Target column missing")
        validations.append(False)
    else:
        logger.info(f"✅ Target column present")
        validations.append(True)
    
    if featured_df.isna().sum().sum() > 0:
        logger.error(f"NaN values found")
        validations.append(False)
    else:
        logger.info(f"✅ No NaN values")
        validations.append(True)
    
    if hasattr(preprocessor, 'transformers_'):
        logger.info(f"✅ Preprocessor fitted")
        validations.append(True)
    else:
        logger.error(f"Preprocessor not fitted")
        validations.append(False)
    
    return all(validations)


def main(input_path: str, 
         output_data_path: str,
         output_preprocessor_path: str,
         target_col: str = 'price') -> None:
    """Main feature engineering pipeline."""
    logger.info("="*80)
    logger.info("STARTING FEATURE ENGINEERING - CARDEKHO V5.0")
    logger.info("="*80)
    
    try:
        df = load_processed_data(input_path)
        df_featured = create_features(df)
        
        categorical_features, numerical_features = define_feature_columns(
            df_featured, target_col=target_col
        )
        
        preprocessor = build_preprocessor(categorical_features, numerical_features)
        
        featured_df, fitted_preprocessor, feature_names = fit_and_transform(
            df_featured,
            preprocessor,
            categorical_features,
            numerical_features,
            target_col=target_col
        )
        
        if not validate_outputs(featured_df, fitted_preprocessor, target_col=target_col):
            logger.error("Validation failed")
            sys.exit(1)
        
        save_preprocessor(fitted_preprocessor, output_preprocessor_path)
        save_featured_data(featured_df, output_data_path)
        
        logger.info("="*80)
        logger.info("✅ FEATURE ENGINEERING COMPLETED")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering for Cardekho V3")
    
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-data', type=str, required=True)
    parser.add_argument('--output-preprocessor', type=str, required=True)
    parser.add_argument('--target', type=str, default='price')
    
    args = parser.parse_args()
    
    main(args.input, args.output_data, args.output_preprocessor, args.target)
