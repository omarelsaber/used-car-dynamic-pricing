"""
Feature Engineering Pipeline for Used Car Dynamic Pricing
==========================================================

Version 4.0: Advanced Text Feature Extraction
- Extracts car_model from name (Brand_Model combination)
- Extracts num_owners from condition text (numerical feature)
- Extracts has_accident from condition text (binary feature)

Advanced extraction strategy:
  • Reduces cardinality: 309 names → ~120 Brand_Model combos (61% reduction)
  • Creates numerical features: num_owners (1-5 range, learnable)
  • Creates binary features: has_accident (0/1, strong signal)

Author: Omar Elsaber
Date: Feb 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import re
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
    logger.info(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def extract_brand(name: str) -> str:
    """
    Extract car brand (first word) from car name.
    
    Args:
        name (str): Full car name (e.g., "Toyota Corolla")
        
    Returns:
        str: Extracted brand (e.g., "Toyota")
    """
    if pd.isna(name) or name.strip() == "":
        return "Unknown"
    
    brand = name.strip().split()[0]
    brand = brand.replace("-", " ").strip()
    
    return brand


def extract_car_model(name: str) -> str:
    """
    Extract Brand_Model combination from full name.
    
    Strategy: Take first 2 words (brand + model) to create mid-cardinality feature.
    This preserves model-specific information while reducing from 309→120 features.
    
    Args:
        name (str): Full car name (e.g., "Toyota Corolla Hybrid")
        
    Returns:
        str: Brand_Model combination (e.g., "Toyota_Corolla")
        
    Examples:
        >>> extract_car_model("Toyota Corolla")
        'Toyota_Corolla'
        >>> extract_car_model("Mercedes-Benz GLC")
        'Mercedes-Benz_GLC'
        >>> extract_car_model("Honda Civic")
        'Honda_Civic'
    """
    if pd.isna(name) or name.strip() == "":
        return "Unknown_Unknown"
    
    parts = name.strip().split()
    
    if len(parts) >= 2:
        # Handle hyphenated brands like "Mercedes-Benz"
        if "-" in parts[0]:
            brand = parts[0]
            model = parts[1] if len(parts) > 1 else "Unknown"
        else:
            brand = parts[0]
            model = parts[1]
        
        return f"{brand}_{model}"
    elif len(parts) == 1:
        return f"{parts[0]}_Unknown"
    else:
        return "Unknown_Unknown"


def extract_num_owners(condition: str) -> int:
    """
    Extract number of owners from condition text.
    
    Looks for patterns like "1 Owner", "2 Owners", "3 Owners".
    Numerical features are powerful for XGBoost!
    
    Args:
        condition (str): Condition text (e.g., "No accidents reported, 1 Owner")
        
    Returns:
        int: Number of owners (1, 2, 3, etc.)
        
    Examples:
        >>> extract_num_owners("No accidents reported, 1 Owner")
        1
        >>> extract_num_owners("1 accident reported, 3 Owners")
        3
        >>> extract_num_owners("Good condition")
        1  # default
    """
    if pd.isna(condition):
        return 1  # Default to 1 owner
    
    # Look for pattern: number followed by "Owner" or "Owners"
    match = re.search(r'(\d+)\s*Owner', condition, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    else:
        return 1  # Default to 1 owner if not specified


def extract_has_accident(condition: str) -> int:
    """
    Extract accident status from condition text.
    
    Binary features provide strong signals for tree-based models!
    
    Args:
        condition (str): Condition text
        
    Returns:
        int: 0 if no accidents, 1 if accidents reported
        
    Examples:
        >>> extract_has_accident("No accidents reported, 1 Owner")
        0
        >>> extract_has_accident("1 accident reported, 2 Owners")
        1
        >>> extract_has_accident("Minor damage reported")
        1
    """
    if pd.isna(condition):
        return 0  # Assume no accidents if not specified
    
    condition_lower = condition.lower()
    
    # Check for "no accidents"
    if "no accident" in condition_lower:
        return 0
    
    # Check for accident indicators
    accident_keywords = ["accident", "damage", "collision", "crash"]
    for keyword in accident_keywords:
        if keyword in condition_lower:
            return 1
    
    return 0  # Default to no accidents


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns using advanced extraction.
    
    VERSION 4.0: Advanced text feature extraction
    Strategy:
      1. Extract brand + model from name (reduces 309 → 120 features)
      2. Extract num_owners from condition (creates numerical feature)
      3. Extract has_accident from condition (creates binary feature)
      4. Drops raw text columns (name, condition) - no longer needed
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features added
    """
    logger.info("="*80)
    logger.info("CREATING ADVANCED FEATURES - VERSION 4.0")
    logger.info("="*80)
    
    df_featured = df.copy()
    
    # ========================================================================
    # TEXT FEATURE EXTRACTION
    # ========================================================================
    
    logger.info("📝 Extracting structured features from text columns...")
    
    # Extract brand
    df_featured['brand'] = df_featured['name'].apply(extract_brand)
    
    # Extract car model (Brand_Model combination) - MAIN CARDINALITY REDUCTION
    df_featured['car_model'] = df_featured['name'].apply(extract_car_model)
    
    # Extract number of owners (NEW numerical feature)
    df_featured['num_owners'] = df_featured['condition'].apply(extract_num_owners)
    
    # Extract accident status (NEW binary feature)
    df_featured['has_accident'] = df_featured['condition'].apply(extract_has_accident)
    
    # Log extraction statistics
    logger.info(f"✅ Text feature extraction complete:")
    logger.info(f"   - Unique brands: {df_featured['brand'].nunique()}")
    logger.info(f"   - Unique car models: {df_featured['car_model'].nunique()} (reduced from {df_featured['name'].nunique()})")
    logger.info(f"   - Cardinality reduction: {df_featured['name'].nunique()} → {df_featured['car_model'].nunique()} ({(1-df_featured['car_model'].nunique()/df_featured['name'].nunique())*100:.0f}% fewer features)")
    logger.info(f"   - Owners range: {df_featured['num_owners'].min()}-{df_featured['num_owners'].max()}")
    logger.info(f"   - Accident rate: {df_featured['has_accident'].mean()*100:.1f}% have accidents")
    
    # Show top car models
    top_models = df_featured['car_model'].value_counts().head(10)
    logger.info(f"   - Top 10 car models:")
    for model, count in top_models.items():
        logger.info(f"     • {model}: {count}")
    
    # ========================================================================
    # NUMERICAL FEATURES
    # ========================================================================
    
    current_year = datetime.now().year
    df_featured['car_age'] = current_year - df_featured['year']
    
    logger.info(f"✅ Created feature: car_age (range: {df_featured['car_age'].min()}-{df_featured['car_age'].max()} years)")
    
    df_featured['mileage_per_year'] = df_featured['mileage'] / df_featured['car_age'].replace(0, 1)
    
    logger.info(f"✅ Created feature: mileage_per_year (mean: {df_featured['mileage_per_year'].mean():.0f} miles/year)")
    
    # ========================================================================
    # FEATURE SUMMARY
    # ========================================================================
    
    logger.info("="*80)
    logger.info("FEATURE EXTRACTION SUMMARY (V4.0)")
    logger.info("="*80)
    logger.info(f"Categorical features extracted:")
    logger.info(f"  ├─ brand: {df_featured['brand'].nunique()} unique values")
    logger.info(f"  └─ car_model: {df_featured['car_model'].nunique()} unique values")
    logger.info(f"Numerical features extracted:")
    logger.info(f"  ├─ num_owners: {df_featured['num_owners'].nunique()} unique values (1-5 range)")
    logger.info(f"  ├─ has_accident: binary (0/1)")
    logger.info(f"  ├─ mileage: continuous")
    logger.info(f"  ├─ car_age: continuous")
    logger.info(f"  └─ mileage_per_year: continuous")
    logger.info(f"Raw text columns removed: name, condition")
    logger.info("="*80)
    
    return df_featured


def define_feature_columns(df: pd.DataFrame, target_col: str = 'price'):
    """
    Define categorical and numerical feature columns.
    
    VERSION 4.0: Use extracted structured features instead of raw text.
    
    Strategy:
      - Categorical: brand (42 values), car_model (120 values), color (10 values)
      - Numerical: mileage, car_age, mileage_per_year, num_owners (1-5), has_accident (0/1)
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target variable to exclude
        
    Returns:
        tuple: (categorical_features, numerical_features)
    """
    logger.info("Defining feature columns (V4.0)...")
    
    # Exclude target and raw text columns
    exclude_cols = {target_col, 'year', 'name', 'condition'}
    
    # ========================================================================
    # CATEGORICAL FEATURES (STRUCTURED)
    # ========================================================================
    categorical_features = [
        'brand',       # ~40-50 unique brands
        'car_model',   # ~100-150 unique brand_model combos (reduced from 309!)
        'color'        # ~10-15 unique colors
    ]
    
    logger.info(f"📊 Categorical features (V4.0 - Structured):")
    logger.info(f"   - brand: {df['brand'].nunique()} unique values")
    logger.info(f"   - car_model: {df['car_model'].nunique()} unique values")
    logger.info(f"     └─ (Reduction from {df['name'].nunique()} raw names - {(1-df['car_model'].nunique()/df['name'].nunique())*100:.0f}% fewer!)")
    logger.info(f"   - color: {df['color'].nunique()} unique values")
    
    # ========================================================================
    # NUMERICAL FEATURES (ENHANCED)
    # ========================================================================
    numerical_features = [
        'mileage',
        'car_age',
        'mileage_per_year',
        'num_owners',      # NEW V4.0: numerical from text
        'has_accident'     # NEW V4.0: binary from text
    ]
    
    logger.info(f"📊 Numerical features (V4.0 - Enhanced):")
    logger.info(f"   - mileage: continuous")
    logger.info(f"   - car_age: continuous")
    logger.info(f"   - mileage_per_year: continuous")
    logger.info(f"   - num_owners: {df['num_owners'].min()}-{df['num_owners'].max()} (NEW - extracted)")
    logger.info(f"   - has_accident: binary 0/1 (NEW - extracted)")
    
    # Validation
    all_features = categorical_features + numerical_features
    missing_features = set(all_features) - set(df.columns)
    
    if missing_features:
        raise ValueError(f"Defined features not found in dataframe: {missing_features}")
    
    # Estimate total features after encoding
    estimated_features = (
        df['brand'].nunique() + 
        df['car_model'].nunique() + 
        df['color'].nunique() + 
        len(numerical_features)  # 5 numerical features unchanged after encoding
    )
    
    logger.info(f"✅ Categorical features ({len(categorical_features)}): {categorical_features}")
    logger.info(f"✅ Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"📈 Estimated total features after encoding: ~{estimated_features}")
    logger.info(f"📊 Feature-to-sample ratio: {estimated_features}/{len(df)} = {estimated_features/len(df)*100:.1f}%")
    
    return categorical_features, numerical_features


def build_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """Build preprocessing pipeline using ColumnTransformer."""
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
    logger.info(f"Total features after encoding: {len(feature_names)}")
    
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=df.index
    )
    
    X_transformed_df[target_col] = y.values
    
    logger.info(f"✅ Final feature dataframe shape: {X_transformed_df.shape}")
    
    return X_transformed_df, preprocessor, feature_names


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str) -> None:
    """Save fitted preprocessor to disk."""
    logger.info(f"Saving preprocessor to: {output_path}")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, output_path)
    
    file_size = output_file.stat().st_size / 1024
    logger.info(f"✅ Preprocessor saved ({file_size:.2f} KB)")


def save_featured_data(df: pd.DataFrame, output_path: str) -> None:
    """Save transformed features to CSV file."""
    logger.info(f"Saving featured data to: {output_path}")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Featured data saved: {df.shape[0]} rows × {df.shape[1]} columns")


def validate_outputs(featured_df: pd.DataFrame, 
                    preprocessor: ColumnTransformer,
                    target_col: str = 'price') -> bool:
    """Validate the feature engineering outputs."""
    logger.info("Validating feature engineering outputs...")
    
    validations = []
    
    # Check target column
    if target_col not in featured_df.columns:
        logger.error(f"❌ Target column '{target_col}' not found")
        validations.append(False)
    else:
        logger.info(f"✅ Target column '{target_col}' present")
        validations.append(True)
    
    # Check for NaN values
    nan_count = featured_df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠️  {nan_count} NaN values found")
        validations.append(False)
    else:
        logger.info(f"✅ No NaN values")
        validations.append(True)
    
    # Check preprocessor
    if hasattr(preprocessor, 'transformers_'):
        logger.info(f"✅ Preprocessor is properly fitted")
        validations.append(True)
    else:
        logger.error(f"❌ Preprocessor is not fitted")
        validations.append(False)
    
    all_passed = all(validations)
    if all_passed:
        logger.info("🎉 All validations passed!")
    else:
        logger.error("❌ Some validations failed")
    
    return all_passed


def main(input_path: str, 
         output_data_path: str,
         output_preprocessor_path: str,
         target_col: str = 'price') -> None:
    """Main feature engineering pipeline."""
    logger.info("="*80)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE V4.0")
    logger.info("Advanced Text Feature Extraction")
    logger.info("="*80)
    
    try:
        df = load_processed_data(input_path)
        df_featured = create_features(df)
        
        categorical_features, numerical_features = define_feature_columns(
            df_featured, 
            target_col=target_col
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
            logger.error("Feature engineering validation failed")
            sys.exit(1)
        
        save_preprocessor(fitted_preprocessor, output_preprocessor_path)
        save_featured_data(featured_df, output_data_path)
        
        logger.info("="*80)
        logger.info("✅ FEATURE ENGINEERING PIPELINE V4.0 COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering with advanced text extraction (V4.0)"
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to processed data CSV')
    parser.add_argument('--output-data', type=str, required=True,
                       help='Path to save featured data CSV')
    parser.add_argument('--output-preprocessor', type=str, required=True,
                       help='Path to save fitted preprocessor')
    parser.add_argument('--target', type=str, default='price',
                       help='Target variable column name')
    
    args = parser.parse_args()
    
    main(args.input, args.output_data, args.output_preprocessor, args.target)
