# test_db_functions.py
from trade_logger import TradeLogger, make_json_safe
import json
import numpy as np

print("Testing database functions...")
logger = TradeLogger()

if logger.connection:
    # Test 1: Create scan
    scan_id = logger.log_scan_start('test_scanner', version='test')
    print(f"✅ Created scan_id: {scan_id}")
    
    # Test 2: Log a trade with NumPy types
    test_trade = {
        'symbol': 'TEST/USDT',
        'exchange': 'binance',
        'probability': 85,
        'entry_price': 100,
        'stop_loss': 95,
        'target_1': 110,
        'scanner_specific_data': json.dumps(make_json_safe({
            'bb_score': 24,
            'volume_surge': np.bool_(True),
            'mfi_value': np.float64(75.5)
        }))
    }
    
    logger.log_trade_opportunity(scan_id, test_trade)
    print("✅ Logged test trade")
    
    # Test 3: Log market regime
    regime_data = {
        'btc_dominance': 57.5,
        'fear_greed_index': 73,
        'regime_type': 'NEUTRAL'
    }
    
    regime_id = logger.log_market_regime(scan_id, regime_data)
    print(f"✅ Logged market regime: {regime_id}")
    
    # Test 4: Log market overview
    overview_data = {
        'total_bounces': 1000,
        'coins_analyzed': 150,
        'overall_success_rate': 76.4
    }
    
    overview_id = logger.log_market_overview(scan_id, overview_data)
    print(f"✅ Logged market overview: {overview_id}")
    
    # Complete scan
    logger.complete_scan(scan_id, 3, 2, 5.0)
    print("✅ Scan completed")
    
    # Verify everything saved
    logger.cursor.execute("""
        SELECT 
            (SELECT COUNT(*) FROM trade_opportunities WHERE scan_id = %s) as trades,
            (SELECT COUNT(*) FROM market_regime WHERE scan_id = %s) as regimes,
            (SELECT COUNT(*) FROM market_overview WHERE scan_id = %s) as overviews
    """, (scan_id, scan_id, scan_id))
    
    result = logger.cursor.fetchone()
    print(f"\n📊 Database verification:")
    print(f"   Trades: {result['trades']}")
    print(f"   Regimes: {result['regimes']}")
    print(f"   Overviews: {result['overviews']}")
    
    logger.close()
else:
    print("❌ No database connection")

