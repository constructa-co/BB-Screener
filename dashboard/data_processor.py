import pandas as pd
import numpy as np
from datetime import datetime, date
import json

def sanitize_dataframe(df):
    """Convert problematic data types for display"""
    for col in df.columns:
        # Convert datetime objects
        if df[col].dtype == 'datetime64[ns]' or df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        # Convert any remaining objects to strings
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
        
        # Handle NaN values
        df[col] = df[col].fillna('')
    
    return df
