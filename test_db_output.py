# test_db_output.py
import sys
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.insert(0, '/Users/robertsmith/Documents/BB Screener')

from modules.output_generator import OutputGenerator
from trade_logger import TradeLogger

# Create mock data similar to what scanner produces
mock_data = {
    'symbol': ['BTC', 'ETH', 'SOL'],
    'exchange': ['binance', 'binance', 'kucoin'],
    'setup_type': ['SHORT', 'SHORT', 'LONG'],
    'probability': [85, 78, 72],
    'entry': [50000, 3200, 120],
    'stop': [51500, 3280, 115],
    'target1': [47500, 3050, 130],
    'bb_score': [24, 18, 15],
    'volume_surge_detected': [np.bool_(True), np.bool_(False), np.bool_(True)],
    'mfi_value': [75.5, 68.2, 32.1],
    'risk_pct': [3.0, 2.5, 4.2],
    'tier': ['PREMIUM', 'PREMIUM', 'HIGH']
}

# Create DataFrame
df = pd.DataFrame(mock_data)

# Mock market regime data
mock_regime = {
    'btc_dominance': 57.5,
    'fear_greed_index': 73,
    'alt_season_indicator': False,
    'market_health_score': 65,
    'regime_type': 'NEUTRAL'
}

print("Testing database logging with mock data...")
output_gen = OutputGenerator()

# This should create Excel AND log to database
filepath = output_gen.generate_excel_output(df, mock_regime, 'test_db_logging.xlsx')

print(f"✅ Test complete! Check:")
print(f"1. Excel created: {filepath}")
print(f"2. Database for new entries")

# Verify database entries
logger = TradeLogger()
if logger.connection:
    logger.cursor.execute("""
        SELECT COUNT(*) as count FROM trade_opportunities 
        WHERE created_at > NOW() - INTERVAL '1 minute'
    """)
    count = logger.cursor.fetchone()['count']
    print(f"3. New trades in database: {count}")
    logger.close()
else:
    print("❌ No database connection available")
