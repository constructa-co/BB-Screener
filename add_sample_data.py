#!/usr/bin/env python3
"""
Add sample data for testing interactive controls
"""

from trade_logger import TradeLogger
from datetime import datetime, timedelta
import random

def add_sample_data():
    """Add sample data for testing"""
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    print("📝 Adding sample data...")
    
    # Add a scan result first
    logger.cursor.execute("""
        INSERT INTO scan_results (scan_type, scan_timestamp)
        VALUES ('bb_scanner_4h', %s)
        RETURNING id
    """, (datetime.now(),))
    
    scan_id = logger.cursor.fetchone()['id']
    print(f"✅ Added scan result with ID: {scan_id}")
    
    # Sample completed trades
    sample_trades = [
        ('BTC/USDT', 85, 43000, 42500, 44000, 2.5, 3.2, 'win'),
        ('ETH/USDT', 78, 2300, 2280, 2350, 2.0, -1.5, 'loss'),
        ('SOL/USDT', 92, 100, 98, 105, 3.0, 4.8, 'win'),
        ('XRP/USDT', 70, 0.55, 0.54, 0.58, 1.8, 2.1, 'win'),
        ('ADA/USDT', 65, 0.45, 0.44, 0.47, 1.5, -2.0, 'loss'),
        ('DOT/USDT', 88, 6.5, 6.4, 6.8, 2.2, 3.5, 'win'),
        ('LINK/USDT', 75, 15.2, 15.0, 15.8, 2.8, -1.2, 'loss'),
        ('MATIC/USDT', 82, 0.85, 0.84, 0.88, 2.1, 4.2, 'win'),
        ('AVAX/USDT', 79, 25.5, 25.2, 26.2, 2.4, 2.8, 'win'),
        ('ATOM/USDT', 71, 8.2, 8.1, 8.5, 1.9, -0.8, 'loss'),
    ]
    
    # Add more random trades
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
    
    for i in range(15):  # Add 15 more random trades
        symbol = random.choice(symbols)
        probability = random.randint(60, 95)
        entry_price = random.uniform(0.1, 50000)
        stop_loss = entry_price * random.uniform(0.95, 0.99)
        target_1 = entry_price * random.uniform(1.01, 1.10)
        risk_reward = random.uniform(1.2, 4.0)
        profit_loss = random.uniform(-5.0, 8.0)
        
        sample_trades.append((
            symbol, probability, entry_price, stop_loss, target_1,
            risk_reward, profit_loss, 'win' if profit_loss > 0 else 'loss'
        ))
    
    # Insert all trades
    for trade in sample_trades:
        symbol, prob, entry, stop, target, rr, pnl, result = trade
        
        logger.cursor.execute("""
            INSERT INTO trade_opportunities 
            (symbol, probability, entry_price, stop_loss, target_1, 
             risk_reward_ratio, profit_loss_percent, trade_result, 
             trade_taken, timestamp, scan_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            symbol, prob, entry, stop, target, rr, pnl, result,
            True, datetime.now() - timedelta(days=random.randint(1, 30)), scan_id
        ))
    
    logger.connection.commit()
    print(f"✅ Added {len(sample_trades)} sample completed trades")
    
    logger.close()

if __name__ == "__main__":
    add_sample_data() 