#!/usr/bin/env python3
"""
Add Database Logging to Main Scanner
This script adds database logging functionality to the main scanner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_logger import TradeLogger
from datetime import datetime

def add_db_logging_to_main_scanner():
    """Add database logging to the main scanner results"""
    
    # Read the main scanner file
    with open('main_scanner.py', 'r') as f:
        content = f.read()
    
    # Find where results are processed and add database logging
    # Look for the section where results are formatted and displayed
    
    # Add import for TradeLogger at the top
    if 'from trade_logger import TradeLogger' not in content:
        # Find the imports section
        import_section = content.find('import logging')
        if import_section != -1:
            # Add the import after the existing imports
            content = content[:import_section] + 'from trade_logger import TradeLogger\n' + content[import_section:]
    
    # Find the section where results are processed (after the analysis is complete)
    # Look for the section where quality_results are processed
    
    # Add database logging after the analysis is complete
    db_logging_code = '''
            # DATABASE LOGGING - Log all quality results to database
            print("💾 Logging results to database...")
            try:
                logger = TradeLogger()
                if logger.connection:
                    # Create scan record
                    scan_id = logger.log_scan_start('bb_scanner_4h', 'BB Scanner 4H')
                    print(f"✅ Created scan_id: {scan_id}")
                    
                    # Log each quality result
                    trades_logged = 0
                    for symbol, result in quality_results.items():
                        try:
                            # Prepare trade data for database
                            trade_data = {
                                'symbol': result.get('symbol', symbol),
                                'exchange': 'Binance',  # Default exchange
                                'timeframe': '4H',
                                'bb_score': result.get('bb_score', 0),
                                'probability': result.get('probability', 0),
                                'risk_reward_ratio': result.get('risk_reward', 0),
                                'current_price': result.get('current_price', 0),
                                'entry_price': result.get('entry_price', 0),
                                'stop_loss': result.get('stop_loss', 0),
                                'target_1': result.get('target_1', 0),
                                'target_2': result.get('target_2', 0),
                                'target_3': result.get('target_3', 0),
                                'rsi': result.get('rsi', 0),
                                'mfi': result.get('mfi', 0),
                                'stochastic_k': result.get('stochastic_k', 0),
                                'volume_surge': result.get('volume_surge', 0),
                                'macd_signal': result.get('macd_signal', 'neutral'),
                                'pattern_type': result.get('setup_type', 'BB Bounce'),
                                'pattern_quality': result.get('tier', 'GOOD'),
                                'confluence_score': result.get('confluence_score', 0),
                                'historical_win_rate': result.get('historical_win_rate', 0),
                                'category_win_rate': result.get('category_win_rate', 0),
                                'similar_setups_count': result.get('similar_setups_count', 0),
                                'market_cap': result.get('market_cap', 0),
                                'volume_24h': result.get('volume_24h', 0),
                                'price_change_24h': result.get('price_change_24h', 0),
                                'scanner_type': 'bb_scanner_4h'
                            }
                            
                            # Log to database
                            success = logger.log_trade_opportunity(scan_id, trade_data)
                            if success:
                                trades_logged += 1
                                print(f"✅ Logged trade: {symbol} -> {result.get('probability', 0)}%")
                            else:
                                print(f"❌ Failed to log trade: {symbol}")
                                
                        except Exception as e:
                            print(f"❌ Error logging trade {symbol}: {e}")
                            continue
                    
                    # Complete the scan
                    logger.complete_scan(scan_id, len(quality_results), trades_logged, 120)
                    print(f"✅ Database logging complete: {trades_logged} trades logged")
                    
                else:
                    print("❌ Database connection failed")
                    
            except Exception as e:
                print(f"❌ Database logging error: {e}")
            
            # Continue with existing code...
'''
    
    # Find where to insert the database logging code
    # Look for the section after quality_results are processed
    insert_point = content.find('# Send summary report')
    if insert_point != -1:
        # Insert the database logging code before the summary report
        content = content[:insert_point] + db_logging_code + '\n            ' + content[insert_point:]
    
    # Write the updated content back to the file
    with open('main_scanner.py', 'w') as f:
        f.write(content)
    
    print("✅ Database logging added to main scanner")
    print("📝 The main scanner will now log all quality results to the database")

def test_db_logging():
    """Test the database logging functionality"""
    print("🧪 Testing database logging...")
    
    logger = TradeLogger()
    if logger.connection:
        # Create a test scan
        scan_id = logger.log_scan_start('bb_scanner_4h', 'BB Scanner 4H Test')
        print(f"✅ Test scan_id: {scan_id}")
        
        # Test trade data
        test_trade = {
            'symbol': 'BTCUSDT',
            'exchange': 'Binance',
            'timeframe': '4H',
            'bb_score': 85,
            'probability': 85,
            'risk_reward_ratio': 2.5,
            'current_price': 115000,
            'entry_price': 115000,
            'stop_loss': 112000,
            'target_1': 118000,
            'target_2': 120000,
            'target_3': 122000,
            'rsi': 65,
            'mfi': 70,
            'stochastic_k': 75,
            'volume_surge': 1.5,
            'macd_signal': 'bullish',
            'pattern_type': 'BB Bounce',
            'pattern_quality': 'PREMIUM',
            'confluence_score': 85,
            'historical_win_rate': 80,
            'category_win_rate': 75,
            'similar_setups_count': 15,
            'market_cap': 1000000000,
            'volume_24h': 50000000,
            'price_change_24h': 2.5,
            'scanner_type': 'bb_scanner_4h'
        }
        
        success = logger.log_trade_opportunity(scan_id, test_trade)
        if success:
            print("✅ Test trade logged successfully")
        else:
            print("❌ Test trade logging failed")
        
        logger.close()
    else:
        print("❌ Database connection failed")

if __name__ == "__main__":
    print("🔧 Adding Database Logging to Main Scanner")
    print("=" * 50)
    
    # Test database logging first
    test_db_logging()
    
    # Add database logging to main scanner
    add_db_logging_to_main_scanner()
    
    print("\n✅ Database logging setup complete!")
    print("🎯 Next time you run the main scanner, it will log results to the database") 