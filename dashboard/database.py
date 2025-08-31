import os
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/opt/bb-screener/.env')

DATABASE_URL = os.getenv('DATABASE_URL')

def get_scanner_data(scanner_name, timeframe='1h', limit=100):
    """Fetch scanner data with proper error handling"""
    engine = create_engine(DATABASE_URL)
    
    # Map scanner names to table names
    table_map = {
        'fvg': 'other_scanners.fvg_signals',
        'flagpole': 'other_scanners.flagpole_signals',
        'elliott': 'other_scanners.elliott_wave_signals',
        'fibonacci': 'other_scanners.fibonacci_signals'
    }
    
    table_name = table_map.get(scanner_name.lower())
    if not table_name:
        return pd.DataFrame()
    
    query = text(f"""
        SELECT 
            symbol,
            timeframe,
            detected_at,
            pattern_type,
            entry_price,
            stop_loss,
            take_profit_1,
            take_profit_2,
            take_profit_3,
            additional_notes
        FROM {table_name}
        WHERE detected_at > NOW() - INTERVAL '24 hours'
        ORDER BY detected_at DESC
        LIMIT :limit
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
            # Sanitize the dataframe for display
            from data_processor import sanitize_dataframe
            df = sanitize_dataframe(df)
            return df
    except Exception as e:
        print(f"Error fetching {scanner_name} data: {e}")
        return pd.DataFrame()
