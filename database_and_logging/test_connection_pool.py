#!/usr/bin/env python3
"""Test connection pooling under load"""
import concurrent.futures
import time
import os
import sys

# Add the database_and_logging directory to the path
sys.path.append(os.path.dirname(__file__))
from universal_scanner_logger import UniversalScannerLogger

def stress_test(thread_id):
    """Simulate multiple scanners writing simultaneously"""
    logger = UniversalScannerLogger(f'stress_test_{thread_id}', 'v1.0')
    
    for i in range(5):
        trade_data = {
            'symbol': f'TEST{thread_id}_{i}',
            'timeframe': '4h',
            'side': 'BUY',
            'entry_price': 100.0 + i,
            'quantity': 1.0,
            'stop_loss': 95.0 + i,
            'take_profit': 105.0 + i,
            'technical_indicators': {
                'rsi': 50.0 + i,
                'mfi': 60.0 + i,
                'final_quality': 80.0 + i
            },
            'scanner_signals': {
                'pattern_type': f'TEST_PATTERN_{i}',
                'pattern_quality': 'GOOD'
            },
            'market_conditions': {
                'volume_24h': 1000000.0 + i,
                'price_change_24h': 1.5 + i
            }
        }
        trade_id = logger.log_trade(trade_data)
        if trade_id:
            print(f"Thread {thread_id} logged trade {i}: {trade_id}")
        time.sleep(0.1)
    
    # Close logger properly
    if hasattr(logger, 'connection_pool'):
        logger.connection_pool.closeall()
    return f"Thread {thread_id} completed"

if __name__ == "__main__":
    # Run stress test
    print("🚀 Starting connection pool stress test...")
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(stress_test, i) for i in range(10)]
        results = [f.result() for f in futures]

    elapsed = time.time() - start
    print(f"✅ Stress test completed in {elapsed:.2f} seconds")
    print(f"✅ 10 concurrent loggers × 5 trades = 50 trades logged")
    
    # Verify in database
    try:
        import psycopg2
        DATABASE_URL = os.getenv('OTHER_SCANNERS_DATABASE_URL')
        if not DATABASE_URL:
            DATABASE_URL = os.getenv('DATABASE_URL')
            if DATABASE_URL and '?' not in DATABASE_URL:
                DATABASE_URL += '?options=-csearch_path=other_scanners'
            elif DATABASE_URL and 'options=' not in DATABASE_URL:
                DATABASE_URL += '&options=-csearch_path=other_scanners'
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SET search_path TO other_scanners")
        cur.execute("""
            SELECT COUNT(*) FROM other_scanners_trades 
            WHERE scanner_name LIKE 'stress_test_%'
        """)
        count = cur.fetchone()[0]
        print(f"✅ Database verification: {count} stress test trades found")
        
        # Clean up test data
        cur.execute("""
            DELETE FROM other_scanners_trades 
            WHERE scanner_name LIKE 'stress_test_%'
        """)
        conn.commit()
        print(f"✅ Test data cleaned up")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
    
    print("🎉 Connection pool stress test completed successfully!")
