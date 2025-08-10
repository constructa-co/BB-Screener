#!/usr/bin/env python3
"""
Add Sample Backtest Data to Database
This script adds realistic backtest data to test the interactive controls
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_logger import TradeLogger
import pandas as pd
from datetime import datetime, timedelta
import random

def add_sample_backtest_data():
    """Add realistic sample backtest data to the database"""
    
    print("📊 Adding Sample Backtest Data to Database...")
    
    # Initialize trade logger
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        # Create a scan result for the backtest
        scan_id = logger.log_scan_start('BB_Backtest_R10', 'R10')
        
        print(f"✅ Created scan_id: {scan_id}")
        
        # Generate realistic sample trades
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
        base_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000,
            'SOLUSDT': 100,
            'XRPUSDT': 0.5,
            'ADAUSDT': 0.4
        }
        
        # Generate 25 sample trades
        for i in range(25):
            symbol = random.choice(symbols)
            base_price = base_prices[symbol]
            
            # Generate realistic trade data
            probability = random.uniform(65, 95)
            risk_reward = random.uniform(1.5, 4.0)
            entry_price = base_price * random.uniform(0.9, 1.1)
            stop_loss = entry_price * random.uniform(0.92, 0.98)
            target_price = entry_price * random.uniform(1.02, 1.15)
            
            # Generate realistic P&L (some wins, some losses)
            if random.random() < 0.6:  # 60% win rate
                profit_loss = random.uniform(1.0, 8.0)
            else:
                profit_loss = random.uniform(-6.0, -1.0)
            
            # Create trade data with all required fields
            trade_data = {
                'symbol': symbol,
                'exchange': 'Binance',
                'timeframe': '4h',
                'bb_score': probability,
                'probability': probability,
                'risk_reward_ratio': risk_reward,
                'current_price': entry_price,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_1': target_price,
                'target_2': target_price * 1.1,
                'target_3': target_price * 1.2,
                'rsi': random.uniform(30, 70),
                'mfi': random.uniform(20, 80),
                'stochastic_k': random.uniform(20, 80),
                'volume_surge': random.uniform(1.2, 3.0),
                'macd_signal': random.choice(['bullish', 'bearish', 'neutral']),
                'pattern_type': 'Volume Profile',
                'pattern_quality': random.choice(['high', 'medium', 'low']),
                'confluence_score': random.uniform(60, 95),
                'historical_win_rate': random.uniform(50, 85),
                'category_win_rate': random.uniform(45, 80),
                'similar_setups_count': random.randint(5, 25),
                'market_cap': base_price * 1000000,
                'volume_24h': base_price * 100000,
                'price_change_24h': random.uniform(-5, 10),
                'scanner_type': 'BB_Backtest_R10'
            }
            
            # Log to database
            success = logger.log_trade_opportunity(scan_id, trade_data)
            if success:
                print(f"✅ Added trade {i+1}: {symbol} -> {profit_loss:.2f}%")
            else:
                print(f"❌ Failed to add trade {i+1}")
        
        print(f"✅ Added {25} sample backtest trades to database")
        
    except Exception as e:
        print(f"❌ Error adding sample data: {e}")
    
    finally:
        logger.close()

def check_backtest_data():
    """Check what backtest data is in the database"""
    
    logger = TradeLogger()
    
    if logger.connection:
        try:
            # Check trade opportunities from backtest
            logger.cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    AVG(CAST(probability AS FLOAT)) as avg_probability,
                    AVG(CAST(risk_reward_ratio AS FLOAT)) as avg_rr,
                    AVG(CAST(profit_loss_percent AS FLOAT)) as avg_pnl,
                    COUNT(CASE WHEN CAST(profit_loss_percent AS FLOAT) > 0 THEN 1 END) as wins,
                    COUNT(CASE WHEN CAST(profit_loss_percent AS FLOAT) <= 0 THEN 1 END) as losses
                FROM trade_opportunities
                WHERE scanner_specific_data::text LIKE '%backtest%'
            """)
            
            results = logger.cursor.fetchall()
            
            if results:
                print("\n📊 Backtest Data in Database:")
                for row in results:
                    total, avg_prob, avg_rr, avg_pnl, wins, losses = row
                    win_rate = (wins / total * 100) if total > 0 else 0
                    print(f"  Total Trades: {total}")
                    print(f"  Avg Probability: {avg_prob:.1f}%")
                    print(f"  Avg R/R: {avg_rr:.2f}")
                    print(f"  Avg P&L: {avg_pnl:.2f}%")
                    print(f"  Win Rate: {win_rate:.1f}% ({wins}/{total})")
            else:
                print("⚠️ No backtest data found in database")
                
        except Exception as e:
            print(f"❌ Error checking data: {e}")
    
    logger.close()

if __name__ == "__main__":
    print("🔧 Sample Backtest Data Generator")
    print("=" * 50)
    
    # Check current data
    check_backtest_data()
    
    # Add sample data
    add_sample_backtest_data()
    
    # Check data again
    print("\n" + "=" * 50)
    check_backtest_data() 