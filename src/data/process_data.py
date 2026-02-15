"""
Data Processing Pipeline for Used Car Dynamic Pricing
======================================================

This module handles the cleaning and preprocessing of raw car dataset.
Based on EDA findings, it removes duplicates and cleans price/mileage columns.

Author: MLOps Team
Date: 2024
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np


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


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
    """
    initial_rows = len(df)
    logger.info(f"Row count before duplicate removal: {initial_rows}")
    
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    duplicates_removed = initial_rows - len(df_cleaned)
    logger.info(f"Row count after duplicate removal: {len(df_cleaned)}")
    logger.info(f"🗑️  Duplicates removed: {duplicates_removed} ({duplicates_removed/initial_rows*100:.2f}%)")
    
    return df_cleaned


def clean_currency(value: str) -> int:
    """
    Clean currency string and convert to integer.
    
    Removes dollar signs, commas, and converts to integer.
    
    Args:
        value (str): Currency string (e.g., "$15,988")
        
    Returns:
        int: Cleaned integer value
        
    Example:
        >>> clean_currency("$15,988")
        15988
    """
    if pd.isna(value):
        return np.nan
    
    # Remove '$' and ',' characters, then convert to int
    cleaned = str(value).replace('$', '').replace(',', '').strip()
    
    try:
        return int(float(cleaned))
    except ValueError:
        logger.warning(f"Could not convert price value: {value}")
        return np.nan


def clean_mileage(value: str) -> int:
    """
    Clean mileage string and convert to integer.
    
    Removes 'miles' text, commas, and converts to integer.
    
    Args:
        value (str): Mileage string (e.g., "45,000 miles")
        
    Returns:
        int: Cleaned integer value
        
    Example:
        >>> clean_mileage("45,000 miles")
        45000
    """
    if pd.isna(value):
        return np.nan
    
    # Remove 'miles' and ',' characters, then convert to int
    cleaned = str(value).replace('miles', '').replace(',', '').strip()
    
    try:
        return int(float(cleaned))
    except ValueError:
        logger.warning(f"Could not convert mileage value: {value}")
        return np.nan


def clean_price_column(df: pd.DataFrame, column: str = 'price') -> pd.DataFrame:
    """
    Clean the price column by removing currency symbols and converting to integer.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of the price column (default: 'price')
        
    Returns:
        pd.DataFrame: Dataframe with cleaned price column
    """
    logger.info(f"Cleaning column: {column}")
    logger.info(f"Sample values before cleaning: {df[column].head(3).tolist()}")
    
    df[column] = df[column].apply(clean_currency)
    
    # Check for any NaN values introduced during cleaning
    nan_count = df[column].isna().sum()
    if nan_count > 0:
        logger.warning(f"⚠️  {nan_count} NaN values found after cleaning {column}")
    
    logger.info(f"Sample values after cleaning: {df[column].head(3).tolist()}")
    logger.info(f"✅ {column} cleaned successfully")
    
    return df


def clean_mileage_column(df: pd.DataFrame, column: str = 'miles') -> pd.DataFrame:
    """
    Clean the mileage column by removing unit strings and converting to integer.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of the mileage column (default: 'miles')
        
    Returns:
        pd.DataFrame: Dataframe with cleaned mileage column
    """
    logger.info(f"Cleaning column: {column}")
    logger.info(f"Sample values before cleaning: {df[column].head(3).tolist()}")
    
    df[column] = df[column].apply(clean_mileage)
    
    # Check for any NaN values introduced during cleaning
    nan_count = df[column].isna().sum()
    if nan_count > 0:
        logger.warning(f"⚠️  {nan_count} NaN values found after cleaning {column}")
    
    logger.info(f"Sample values after cleaning: {df[column].head(3).tolist()}")
    logger.info(f"✅ {column} cleaned successfully")
    
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to follow standard naming conventions.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with renamed columns
    """
    logger.info("Renaming columns for standardization")
    
    column_mapping = {
        'miles': 'mileage'
    }
    
    df = df.rename(columns=column_mapping)
    
    logger.info(f"✅ Columns renamed: {column_mapping}")
    logger.info(f"Final columns: {df.columns.tolist()}")
    
    return df


def validate_processed_data(df: pd.DataFrame) -> bool:
    """
    Validate the processed dataframe meets quality requirements.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating processed data...")
    
    validations = []
    
    # Check required columns exist
    required_columns = ['name', 'year', 'mileage', 'color', 'condition', 'price']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        validations.append(False)
    else:
        logger.info(f"✅ All required columns present")
        validations.append(True)
    
    # Check for null values in critical columns
    critical_cols = ['price', 'mileage', 'year']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning(f"⚠️  Column '{col}' has {null_count} null values")
                validations.append(False)
            else:
                logger.info(f"✅ Column '{col}' has no null values")
                validations.append(True)
    
    # Check data types
    if 'price' in df.columns and not pd.api.types.is_numeric_dtype(df['price']):
        logger.error(f"❌ Column 'price' is not numeric: {df['price'].dtype}")
        validations.append(False)
    else:
        logger.info(f"✅ Column 'price' is numeric")
        validations.append(True)
    
    if 'mileage' in df.columns and not pd.api.types.is_numeric_dtype(df['mileage']):
        logger.error(f"❌ Column 'mileage' is not numeric: {df['mileage'].dtype}")
        validations.append(False)
    else:
        logger.info(f"✅ Column 'mileage' is numeric")
        validations.append(True)
    
    # Check for reasonable value ranges
    if 'price' in df.columns:
        if (df['price'] < 0).any():
            logger.error(f"❌ Negative values found in 'price'")
            validations.append(False)
        else:
            logger.info(f"✅ All price values are non-negative")
            validations.append(True)
    
    all_passed = all(validations)
    if all_passed:
        logger.info("🎉 All validations passed!")
    else:
        logger.error("❌ Some validations failed")
    
    return all_passed


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path where processed CSV will be saved
    """
    logger.info(f"Saving processed data to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Processed data saved successfully")
    logger.info(f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")


def main(input_path: str, output_path: str) -> None:
    """
    Main data processing pipeline.
    
    Args:
        input_path (str): Path to raw CSV file
        output_path (str): Path to save processed CSV file
    """
    logger.info("="*80)
    logger.info("STARTING DATA PROCESSING PIPELINE")
    logger.info("="*80)
    
    try:
        # 1. Load data
        df = load_data(input_path)
        
        # 2. Remove duplicates
        df = remove_duplicates(df)
        
        # 3. Clean price column
        df = clean_price_column(df, column='price')
        
        # 4. Clean mileage column
        df = clean_mileage_column(df, column='miles')
        
        # 5. Rename columns
        df = rename_columns(df)
        
        # 6. Validate processed data
        if not validate_processed_data(df):
            logger.error("Data validation failed. Please review the logs.")
            sys.exit(1)
        
        # 7. Save processed data
        save_processed_data(df, output_path)
        
        logger.info("="*80)
        logger.info("✅ DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw car dataset for dynamic pricing model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data/process_data.py \\
    --input data/raw/car_web_scraped_dataset.csv \\
    --output data/processed/processed_cars.csv
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw CSV file (e.g., data/raw/car_web_scraped_dataset.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save processed CSV file (e.g., data/processed/processed_cars.csv)'
    )
    
    args = parser.parse_args()
    
    main(args.input, args.output)
