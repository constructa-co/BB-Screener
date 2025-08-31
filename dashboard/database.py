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
    
    # Map scanner names to table names and their specific columns
    scanner_config = {
        'fvg': {
            'table': 'other_scanners.fvg_signals',
            'columns': 'symbol, timeframe, detected_at, gap_type, current_price, entry_price, stop_loss, target_1, target_2, target_3, setup_score, gap_status'
        },
        'flagpole': {
            'table': 'other_scanners.flagpole_signals',
            'columns': 'symbol, timeframe, detected_at, pattern_type, direction, current_price, breakout_level, target_price, stop_loss, potential_pct, risk_reward, setup_score'
        },
        'elliott': {
            'table': 'other_scanners.elliott_wave_signals',
            'columns': 'symbol, timeframe, detected_at, wave_type, current_price, entry_price, stop_loss, target_price, setup_score'
        },
        'fibonacci': {
            'table': 'other_scanners.fibonacci_signals',
            'columns': 'symbol, timeframe, detected_at, fib_level, current_price, entry_price, stop_loss, target_price, setup_score'
        }
    }
    
    config = scanner_config.get(scanner_name.lower())
    if not config:
        return pd.DataFrame()
    
    query = text(f"""
        SELECT 
            {config['columns']}
        FROM {config['table']}
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
