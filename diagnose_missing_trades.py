# diagnose_missing_trades.py - Find out WHY trades are missing
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Try to import dotenv for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not available - using system environment variables")
    def load_dotenv():
        pass

def diagnose_missing_trades():
    """Find which trades are missing and why"""
    
    # Load both files
    scanner_file = 'outputs/excel_reports/bb_analysis_20250816_192704.xlsx'
    all_analysis = pd.read_excel(scanner_file, sheet_name='All_Analysis')
    
    # Use same connection method as trade_logger.py
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ No DATABASE_URL found in environment variables")
        return set()
    
    try:
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return set()
    
    # Get all symbols from database - look for yesterday's trades (2025-08-16)
    cursor.execute("""
        SELECT DISTINCT symbol 
        FROM trade_opportunities 
        WHERE DATE(timestamp) = '2025-08-16'
    """)
    db_symbols = {row['symbol'] for row in cursor.fetchall()}
    
    # Find missing trades
    excel_symbols = set(all_analysis['symbol'].tolist())
    missing_symbols = excel_symbols - db_symbols
    
    print(f"Database has {len(db_symbols)} unique symbols")
    print(f"Excel has {len(excel_symbols)} unique symbols")
    print(f"Missing {len(missing_symbols)} symbols:\n")
    
    # Analyze missing trades
    missing_trades = all_analysis[all_analysis['symbol'].isin(missing_symbols)]
    
    # Check for patterns
    print("Missing trades analysis:")
    print(f"  Probability distribution: {missing_trades['probability'].value_counts().head()}")
    print(f"  Tier distribution: {missing_trades['tier'].value_counts().head()}")
    print(f"  Exchange distribution: {missing_trades['exchange'].value_counts().head()}")
    
    # Check for duplicate symbols in Excel
    duplicates = all_analysis[all_analysis.duplicated(subset=['symbol'], keep=False)]
    if len(duplicates) > 0:
        print(f"\n⚠️ Found {len(duplicates)} duplicate symbols in Excel!")
        print(f"Duplicate symbols: {duplicates['symbol'].unique()[:10]}")
    
    # Check the actual missing symbols
    print(f"\nFirst 20 missing symbols:")
    for symbol in list(missing_symbols)[:20]:
        trade = all_analysis[all_analysis['symbol'] == symbol].iloc[0]
        print(f"  {symbol}: prob={trade['probability']}, action={trade.get('action', 'N/A')}")
    
    cursor.close()
    conn.close()
    
    return missing_symbols

if __name__ == "__main__":
    missing = diagnose_missing_trades()
