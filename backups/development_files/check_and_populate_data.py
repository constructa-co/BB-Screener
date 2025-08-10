#!/usr/bin/env python3
"""
Check database data and populate sample data for testing
"""

from trade_logger import TradeLogger
import pandas as pd
from datetime import datetime, timedelta
import random

def check_database_data():
    """Check what data is available in the database"""
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    print("✅ Database connection successful")
    
    # Check trade opportunities
    logger.cursor.execute("SELECT COUNT(*) FROM trade_opportunities")
    result = logger.cursor.fetchone()
    total_opportunities = result['count'] if isinstance(result, dict) else result[0]
    print(f"📊 Total trade opportunities: {total_opportunities}")
    
    # Check completed trades
    logger.cursor.execute("""
        SELECT COUNT(*) FROM trade_opportunities 
        WHERE trade_taken = TRUE AND trade_result IS NOT NULL
    """)
    result = logger.cursor.fetchone()
    completed_trades = result['count'] if isinstance(result, dict) else result[0]
    print(f"✅ Completed trades with results: {completed_trades}")
    
    # Check scan results
    logger.cursor.execute("SELECT COUNT(*) FROM scan_results")
    result = logger.cursor.fetchone()
    total_scans = result['count'] if isinstance(result, dict) else result[0]
    print(f"🔍 Total scan results: {total_scans}")
    
    logger.close()
    
    return completed_trades

def populate_sample_data():
    """Add sample completed trades for testing"""
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    print("📝 Adding sample scan results and completed trades...")
    
    # First, add some scan results
    scan_results = [
        {
            'scan_type': 'bb_scanner_4h',
            'symbols_scanned': 50,
            'opportunities_found': 5,
            'scan_timestamp': datetime.now() - timedelta(days=1),
            'execution_time': 2.5
        },
        {
            'scan_type': 'ict_scanner_15m',
            'symbols_scanned': 30,
            'opportunities_found': 3,
            'scan_timestamp': datetime.now() - timedelta(days=2),
            'execution_time': 1.8
        },
        {
            'scan_type': 'bb_scanner_4h',
            'symbols_scanned': 50,
            'opportunities_found': 4,
            'scan_timestamp': datetime.now() - timedelta(days=3),
            'execution_time': 2.1
        }
    ]
    
    scan_ids = []
    for scan in scan_results:
        logger.cursor.execute("""
            INSERT INTO scan_results 
            (scan_type, symbols_scanned, opportunities_found, scan_timestamp, execution_time)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            scan['scan_type'],
            scan['symbols_scanned'],
            scan['opportunities_found'],
            scan['scan_timestamp'],
            scan['execution_time']
        ))
        scan_ids.append(logger.cursor.fetchone()['id'])
    
    logger.connection.commit()
    print(f"✅ Added {len(scan_results)} scan results")
    
    # Sample trade data
    sample_trades = [
        {
            'symbol': 'BTC/USDT',
            'probability': 85,
            'entry_price': 43000,
            'stop_loss': 42500,
            'target_1': 44000,
            'risk_reward_ratio': 2.5,
            'profit_loss_percent': 3.2,
            'trade_result': 'win',
            'trade_taken': True,
            'timestamp': datetime.now() - timedelta(days=1)
        },
        {
            'symbol': 'ETH/USDT',
            'probability': 78,
            'entry_price': 2300,
            'stop_loss': 2280,
            'target_1': 2350,
            'risk_reward_ratio': 2.0,
            'profit_loss_percent': -1.5,
            'trade_result': 'loss',
            'trade_taken': True,
            'timestamp': datetime.now() - timedelta(days=2)
        },
        {
            'symbol': 'SOL/USDT',
            'probability': 92,
            'entry_price': 100,
            'stop_loss': 98,
            'target_1': 105,
            'risk_reward_ratio': 3.0,
            'profit_loss_percent': 4.8,
            'trade_result': 'win',
            'trade_taken': True,
            'timestamp': datetime.now() - timedelta(days=3)
        },
        {
            'symbol': 'XRP/USDT',
            'probability': 70,
            'entry_price': 0.55,
            'stop_loss': 0.54,
            'target_1': 0.58,
            'risk_reward_ratio': 1.8,
            'profit_loss_percent': 2.1,
            'trade_result': 'win',
            'trade_taken': True,
            'timestamp': datetime.now() - timedelta(days=4)
        },
        {
            'symbol': 'ADA/USDT',
            'probability': 65,
            'entry_price': 0.45,
            'stop_loss': 0.44,
            'target_1': 0.47,
            'risk_reward_ratio': 1.5,
            'profit_loss_percent': -2.0,
            'trade_result': 'loss',
            'trade_taken': True,
            'timestamp': datetime.now() - timedelta(days=5)
        }
    ]
    
    # Add more random trades
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
    
    for i in range(20):  # Add 20 more random trades
        symbol = random.choice(symbols)
        probability = random.randint(60, 95)
        entry_price = random.uniform(0.1, 50000)
        stop_loss = entry_price * random.uniform(0.95, 0.99)
        target_1 = entry_price * random.uniform(1.01, 1.10)
        risk_reward = random.uniform(1.2, 4.0)
        profit_loss = random.uniform(-5.0, 8.0)
        
        sample_trades.append({
            'symbol': symbol,
            'probability': probability,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'risk_reward_ratio': risk_reward,
            'profit_loss_percent': profit_loss,
            'trade_result': 'win' if profit_loss > 0 else 'loss',
            'trade_taken': True,
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 30))
        })
    
    # Insert sample trades
    for i, trade in enumerate(sample_trades):
        scan_id = scan_ids[i % len(scan_ids)]  # Cycle through scan IDs
        logger.cursor.execute("""
            INSERT INTO trade_opportunities 
            (symbol, probability, entry_price, stop_loss, target_1, 
             risk_reward_ratio, profit_loss_percent, trade_result, 
             trade_taken, timestamp, scan_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade['symbol'],
            trade['probability'],
            trade['entry_price'],
            trade['stop_loss'],
            trade['target_1'],
            trade['risk_reward_ratio'],
            trade['profit_loss_percent'],
            trade['trade_result'],
            trade['trade_taken'],
            trade['timestamp'],
            scan_id
        ))
    
    logger.connection.commit()
    print(f"✅ Added {len(sample_trades)} sample completed trades")
    
    logger.close()

def main():
    """Main function to check and populate data"""
    print("🔍 Checking database data...")
    
    completed_trades = check_database_data()
    
    if completed_trades < 10:
        print(f"\n⚠️ Only {completed_trades} completed trades found. Need at least 10 for meaningful analysis.")
        
        response = input("Add sample data for testing? (y/n): ")
        if response.lower() == 'y':
            populate_sample_data()
            print("\n✅ Sample data added! The interactive controls should now work.")
        else:
            print("Sample data not added. Interactive controls will show 'no data' message.")
    else:
        print(f"\n✅ Sufficient data available ({completed_trades} completed trades). Interactive controls should work.")

if __name__ == "__main__":
    main() 