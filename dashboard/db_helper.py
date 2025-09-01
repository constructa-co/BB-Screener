#!/usr/bin/env python3
"""
Database Helper with Automatic Timezone Fix
Handles inconsistent timestamp types across all scanner tables
"""

import pandas as pd
import psycopg2
import os

def query_with_timezone_fix(query, conn, params=None):
    """
    Execute query and automatically handle timezone inconsistencies
    
    Args:
        query: SQL query string
        conn: Database connection
        params: Query parameters (optional)
    
    Returns:
        DataFrame with properly handled timestamps
    """
    try:
        # Execute query
        if params:
            df = pd.read_sql(query, conn, params=params)
        else:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return df
        
        # Find and fix all datetime columns
        for col in df.columns:
            try:
                # Check if column contains datetime data
                if (df[col].dtype == 'datetime64[ns]' or 
                    'timestamp' in str(df[col].dtype) or
                    pd.api.types.is_datetime64_any_dtype(df[col])):
                    
                    # Convert to datetime
                    df[col] = pd.to_datetime(df[col])
                    
                    # Check if timezone aware
                    if df[col].dt.tz is None:
                        # Timezone-naive - add UTC
                        df[col] = df[col].dt.tz_localize('UTC')
                        print(f"✅ Applied tz_localize to {col} (was timezone-naive)")
                    else:
                        # Already timezone-aware - ensure it's UTC
                        df[col] = df[col].dt.tz_convert('UTC')
                        print(f"✅ Applied tz_convert to {col} (was already timezone-aware)")
                        
            except Exception as e:
                # Skip if conversion fails
                print(f"⚠️ Skipping timezone fix for {col}: {e}")
                continue
        
        return df
        
    except Exception as e:
        print(f"❌ Query execution failed: {e}")
        return pd.DataFrame()

def convert_to_uae_time(df, column='detected_at'):
    """
    Convert a specific column to UAE timezone
    
    Args:
        df: DataFrame
        column: Column name to convert
    
    Returns:
        DataFrame with UAE timezone column
    """
    try:
        if column in df.columns and not df[column].isna().all():
            # Ensure column is timezone-aware UTC first
            if df[column].dt.tz is None:
                df[column] = df[column].dt.tz_localize('UTC')
            
            # Convert to UAE timezone
            df[column] = df[column].dt.tz_convert('Asia/Dubai')
            print(f"✅ Converted {column} to UAE timezone")
            
    except Exception as e:
        print(f"⚠️ UAE timezone conversion failed for {column}: {e}")
    
    return df

def get_db_connection():
    """Get database connection with timezone handling"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    
    try:
        conn = psycopg2.connect(database_url, options="-c timezone=UTC")
        return conn
    except Exception as e:
        raise ConnectionError(f"Database connection failed: {e}")
