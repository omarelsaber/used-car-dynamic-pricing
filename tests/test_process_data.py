"""
Unit Tests for Data Processing Module
======================================

Tests the data cleaning and processing functions.

Author: Omar Elsaber
Date: Feb 2026
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.process_data import (
    parse_mileage,
    parse_engine,
    parse_max_power,
    remove_duplicates
)


class TestParseMileage:
    """Test suite for parse_mileage function."""
    
    def test_parse_mileage_basic(self):
        """Test basic mileage parsing with kmpl."""
        result = parse_mileage("23.4 kmpl")
        assert result == 23.4
    
    def test_parse_mileage_integer(self):
        """Test mileage parsing with integer value."""
        result = parse_mileage("18 kmpl")
        assert result == 18.0
    
    def test_parse_mileage_km_kg(self):
        """Test mileage parsing with km/kg format."""
        result = parse_mileage("18.9 km/kg")
        assert result == 18.9
    
    def test_parse_mileage_null_string(self):
        """Test handling of 'null' string."""
        result = parse_mileage("null")
        assert pd.isna(result)
    
    def test_parse_mileage_nan(self):
        """Test handling of NaN values."""
        result = parse_mileage(np.nan)
        assert pd.isna(result)
    
    def test_parse_mileage_empty_string(self):
        """Test handling of empty string."""
        result = parse_mileage("")
        assert pd.isna(result)


class TestParseEngine:
    """Test suite for parse_engine function."""
    
    def test_parse_engine_basic(self):
        """Test basic engine parsing."""
        result = parse_engine("1248 CC")
        assert result == 1248
    
    def test_parse_engine_lowercase(self):
        """Test engine parsing with lowercase."""
        result = parse_engine("1197 cc")
        assert result == 1197
    
    def test_parse_engine_null_string(self):
        """Test handling of 'null' string."""
        result = parse_engine("null")
        assert pd.isna(result)
    
    def test_parse_engine_nan(self):
        """Test handling of NaN values."""
        result = parse_engine(np.nan)
        assert pd.isna(result)


class TestParseMaxPower:
    """Test suite for parse_max_power function."""
    
    def test_parse_max_power_basic(self):
        """Test basic max power parsing."""
        result = parse_max_power("74 bhp")
        assert result == 74.0
    
    def test_parse_max_power_decimal(self):
        """Test max power parsing with decimal."""
        result = parse_max_power("81.86 bhp")
        assert result == 81.86
    
    def test_parse_max_power_null_string(self):
        """Test handling of 'null' string."""
        result = parse_max_power("null")
        assert pd.isna(result)
    
    def test_parse_max_power_nan(self):
        """Test handling of NaN values."""
        result = parse_max_power(np.nan)
        assert pd.isna(result)


class TestRemoveDuplicates:
    """Test suite for remove_duplicates function."""
    
    def test_remove_duplicates_basic(self):
        """Test basic duplicate removal."""
        df = pd.DataFrame({
            'name': ['Toyota', 'Honda', 'Toyota'],
            'price': [15000, 20000, 15000]
        })
        result = remove_duplicates(df)
        assert len(result) == 2
        assert list(result['name']) == ['Toyota', 'Honda']
    
    def test_remove_duplicates_no_duplicates(self):
        """Test with no duplicates."""
        df = pd.DataFrame({
            'name': ['Toyota', 'Honda', 'Ford'],
            'price': [15000, 20000, 18000]
        })
        result = remove_duplicates(df)
        assert len(result) == 3
    
    def test_remove_duplicates_no_duplicates(self):
        """Test with no duplicates."""
        df = pd.DataFrame({
            'name': ['Toyota', 'Honda', 'Ford'],
            'price': [15000, 20000, 18000]
        })
        result = remove_duplicates(df)
        assert len(result) == 3
    
    def test_remove_duplicates_preserves_columns(self):
        """Test that columns are preserved."""
        df = pd.DataFrame({
            'name': ['Toyota', 'Honda', 'Toyota'],
            'price': [15000, 20000, 15000],
            'year': [2020, 2019, 2020]
        })
        result = remove_duplicates(df)
        assert list(result.columns) == ['name', 'price', 'year']