#!/usr/bin/env python3
"""
Universal Timezone Handler for BB-Screener Dashboard
Handles both timezone-aware and timezone-naive timestamps consistently
"""

import pandas as pd
from datetime import datetime
import pytz

def smart_timezone_handler(df, column='detected_at', target_tz='UTC'):
    """
    Universal timezone handler that works for both aware and naive timestamps
    
    Args:
        df: pandas DataFrame
        column: column name containing timestamps (default: 'detected_at')
        target_tz: target timezone (default: 'UTC')
    
    Returns:
        DataFrame with properly handled timestamps
    """
    if column not in df.columns or df[column].isna().all():
        return df
    
    # Convert to datetime first
    df[column] = pd.to_datetime(df[column])
    
    # Check if timezone-aware or naive
    try:
        if df[column].dt.tz is None:
            # Timezone-naive - needs localization first
            df[column] = df[column].dt.tz_localize('UTC')
            print(f"✅ Applied tz_localize to {column} (was timezone-naive)")
        else:
            print(f"✅ Applied tz_convert to {column} (was already timezone-aware)")
        
        # Now convert to target timezone if needed
        if target_tz != 'UTC':
            df[column] = df[column].dt.tz_convert(target_tz)
            
    except Exception as e:
        print(f"⚠️ Timezone handling error for {column}: {e}")
        # Fallback - return as is
        pass
    
    return df

def safe_tz_convert(timestamp_series, target_tz):
    """
    Safe timezone conversion for individual timestamp series
    (Backward compatibility with existing code)
    """
    if timestamp_series.dt.tz is None:
        # Timezone-naive: localize to UTC first, then convert
        return timestamp_series.dt.tz_localize('UTC').dt.tz_convert(target_tz)
    else:
        # Already timezone-aware: convert directly
        return timestamp_series.dt.tz_convert(target_tz)
