"""
Data Processing Pipeline for Cardekho V3 Dataset
===============================================

This module handles cleaning and preprocessing of the Cardekho car dataset.
Parses unit strings (CC, bhp, kmpl) and handles missing values.

Author: Omar Elsaber
Date: Feb 2026
Version: V3.0 (Cardekho Dataset)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re


# Ensure logs directory exists for file handler
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/data_processing.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        input_path (str): Path to the raw CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
    """
    logger.info(f"Loading data from: {input_path}")
    
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise


def parse_mileage(mileage_str: str) -> float:
    """
    Parse mileage string to extract float value.
    
    Examples:
        "23.4 kmpl" → 23.4
        "18.9 km/kg" → 18.9
        "null" → NaN
        
    Args:
        mileage_str (str): Mileage string from dataset
        
    Returns:
        float: Parsed mileage value or NaN
    """
    if pd.isna(mileage_str):
        return np.nan
    
    # Convert to string and handle "null" strings
    mileage_str = str(mileage_str).strip().lower()
    
    if mileage_str == "null" or mileage_str == "":
        return np.nan
    
    # Extract number using regex (handles "23.4 kmpl", "18.9 km/kg", etc.)
    match = re.search(r'(\d+\.?\d*)', mileage_str)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    
    return np.nan


def parse_engine(engine_str: str) -> int:
    """
    Parse engine string to extract integer CC value.
    
    Examples:
        "1248 CC" → 1248
        "1197 CC" → 1197
        "null" → NaN
        
    Args:
        engine_str (str): Engine string from dataset
        
    Returns:
        int: Parsed engine CC value or NaN
    """
    if pd.isna(engine_str):
        return np.nan
    
    # Convert to string and handle "null"
    engine_str = str(engine_str).strip().lower()
    
    if engine_str == "null" or engine_str == "":
        return np.nan
    
    # Extract number before "cc"
    match = re.search(r'(\d+)', engine_str)
    
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return np.nan
    
    return np.nan


def parse_max_power(power_str: str) -> float:
    """
    Parse max power string to extract float bhp value.
    
    Examples:
        "74 bhp" → 74.0
        "81.86 bhp" → 81.86
        "null" → NaN
        "" → NaN
        
    Args:
        power_str (str): Power string from dataset
        
    Returns:
        float: Parsed power value or NaN
    """
    if pd.isna(power_str):
        return np.nan
    
    # Convert to string and handle "null"
    power_str = str(power_str).strip().lower()
    
    if power_str == "null" or power_str == "":
        return np.nan
    
    # Extract number (handles "74 bhp", "81.86 bhp", etc.)
    match = re.search(r'(\d+\.?\d*)', power_str)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    
    return np.nan


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records from the dataframe."""
    initial_rows = len(df)
    logger.info(f"Row count before duplicate removal: {initial_rows}")
    
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    duplicates_removed = initial_rows - len(df_cleaned)
    logger.info(f"Row count after duplicate removal: {len(df_cleaned)}")
    logger.info(f"Duplicates removed: {duplicates_removed} ({duplicates_removed/initial_rows*100:.2f}%)")
    
    return df_cleaned


def clean_cardekho_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Cardekho V3 dataset.
    
    Steps:
    1. Rename columns to standardized names
    2. Parse unit strings (CC, bhp, kmpl)
    3. Handle missing values
    4. Drop rows with critical missing data
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    logger.info("="*80)
    logger.info("CLEANING CARDEKHO V3 DATASET")
    logger.info("="*80)
    
    df_clean = df.copy()
    
    # ========================================================================
    # STEP 1: RENAME COLUMNS
    # ========================================================================
    
    logger.info("Renaming columns to standard names...")
    
    column_mapping = {
        'selling_price': 'price',      # Target variable
        'km_driven': 'mileage_driven'  # Total kilometers driven
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    logger.info(f"Columns renamed: {column_mapping}")
    
    # ========================================================================
    # STEP 2: PARSE UNIT STRINGS
    # ========================================================================
    
    logger.info("Parsing unit strings (kmpl, CC, bhp)...")
    
    # Parse mileage (e.g., "23.4 kmpl" → 23.4)
    if 'mileage' in df_clean.columns:
        logger.info("  - Parsing mileage (kmpl)...")
        logger.info(f"    Sample before: {df_clean['mileage'].head(3).tolist()}")
        df_clean['mileage'] = df_clean['mileage'].apply(parse_mileage)
        logger.info(f"    Sample after: {df_clean['mileage'].head(3).tolist()}")
        logger.info(f"    Mileage parsed (mean: {df_clean['mileage'].mean():.2f} kmpl)")
    
    # Parse engine (e.g., "1248 CC" → 1248)
    if 'engine' in df_clean.columns:
        logger.info("  - Parsing engine (CC)...")
        logger.info(f"    Sample before: {df_clean['engine'].head(3).tolist()}")
        df_clean['engine'] = df_clean['engine'].apply(parse_engine)
        logger.info(f"    Sample after: {df_clean['engine'].head(3).tolist()}")
        logger.info(f"    Engine parsed (mean: {df_clean['engine'].mean():.0f} CC)")
    
    # Parse max_power (e.g., "74 bhp" → 74.0)
    if 'max_power' in df_clean.columns:
        logger.info("  - Parsing max_power (bhp)...")
        logger.info(f"    Sample before: {df_clean['max_power'].head(3).tolist()}")
        df_clean['max_power'] = df_clean['max_power'].apply(parse_max_power)
        logger.info(f"    Sample after: {df_clean['max_power'].head(3).tolist()}")
        logger.info(f"    Max power parsed (mean: {df_clean['max_power'].mean():.2f} bhp)")
    
    # ========================================================================
    # STEP 3: DROP TORQUE COLUMN (TOO MESSY)
    # ========================================================================
    
    if 'torque' in df_clean.columns:
        logger.info("Dropping 'torque' column (complex to parse, skipping)")
        df_clean = df_clean.drop(columns=['torque'])
    
    # ========================================================================
    # STEP 4: HANDLE MISSING VALUES
    # ========================================================================
    
    logger.info("Analyzing missing values...")
    
    missing_before = df_clean.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    
    if len(missing_cols) > 0:
        logger.info(f"Missing values found:")
        for col, count in missing_cols.items():
            pct = count / len(df_clean) * 100
            logger.info(f"  - {col}: {count} ({pct:.2f}%)")
    
    # Drop rows where critical features are missing
    critical_features = ['engine', 'max_power', 'mileage', 'price']
    
    logger.info(f"Dropping rows with missing critical features: {critical_features}")
    rows_before = len(df_clean)
    
    df_clean = df_clean.dropna(subset=critical_features)
    
    rows_after = len(df_clean)
    rows_dropped = rows_before - rows_after
    
    logger.info(f"Rows dropped due to missing values: {rows_dropped} ({rows_dropped/rows_before*100:.2f}%)")
    logger.info(f"Remaining rows: {rows_after}")
    
    # ========================================================================
    # STEP 5: DATA TYPE VALIDATION
    # ========================================================================
    
    logger.info("Validating data types...")
    
    # Ensure numerical columns are correct type
    numerical_cols = {
        'year': 'int',
        'price': 'float',
        'mileage_driven': 'int',
        'mileage': 'float',
        'engine': 'int',
        'max_power': 'float',
        'seats': 'int'
    }
    
    for col, dtype in numerical_cols.items():
        if col in df_clean.columns:
            if dtype == 'int':
                df_clean[col] = df_clean[col].astype(int)
            else:
                df_clean[col] = df_clean[col].astype(float)
    
    logger.info("Data types validated")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("="*80)
    logger.info("CLEANING SUMMARY")
    logger.info("="*80)
    logger.info(f"Final dataset shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    logger.info(f"Columns: {df_clean.columns.tolist()}")
    logger.info("="*80)
    
    return df_clean


def validate_processed_data(df: pd.DataFrame) -> bool:
    """Validate the processed dataframe meets quality requirements."""
    logger.info("Validating processed data...")
    
    validations = []
    
    # Check required columns exist
    required_columns = ['name', 'year', 'price', 'mileage_driven', 'fuel', 
                       'seller_type', 'transmission', 'owner', 'mileage', 
                       'engine', 'max_power', 'seats']
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        validations.append(False)
    else:
        logger.info(f"All required columns present")
        validations.append(True)
    
    # Check for null values in critical columns
    critical_cols = ['price', 'engine', 'max_power', 'mileage']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Column '{col}' has {null_count} null values")
                validations.append(False)
            else:
                logger.info(f"Column '{col}' has no null values")
                validations.append(True)
    
    # Check price is positive
    if 'price' in df.columns:
        if (df['price'] <= 0).any():
            logger.error(f"Negative or zero values found in 'price'")
            validations.append(False)
        else:
            logger.info(f"All price values are positive")
            validations.append(True)
    
    all_passed = all(validations)
    if all_passed:
        logger.info("All validations passed!")
    else:
        logger.error("Some validations failed")
    
    return all_passed


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed dataframe to CSV file."""
    logger.info(f"Saving processed data to: {output_path}")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    logger.info(f"Processed data saved successfully")
    logger.info(f"Final dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")


def main(input_path: str, output_path: str) -> None:
    """Main data processing pipeline for Cardekho V3 dataset."""
    logger.info("="*80)
    logger.info("STARTING DATA PROCESSING PIPELINE - CARDEKHO V3")
    logger.info("="*80)
    
    try:
        # 1. Load data
        df = load_data(input_path)
        
        # 2. Remove duplicates
        df = remove_duplicates(df)
        
        # 3. Clean data (parse units, handle missing values)
        df = clean_cardekho_data(df)
        
        # 4. Validate processed data
        if not validate_processed_data(df):
            logger.error("Data validation failed. Please review the logs.")
            sys.exit(1)
        
        # 5. Save processed data
        save_processed_data(df, output_path)
        
        logger.info("="*80)
        logger.info("✅ DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Cardekho V3 car dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw CSV file (e.g., data/raw/cardekho_v3.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save processed CSV file (e.g., data/processed/processed_cars.csv)'
    )
    
    args = parser.parse_args()
    
    main(args.input, args.output)
