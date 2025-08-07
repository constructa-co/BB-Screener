#!/usr/bin/env python3
"""
Fix Main Scanner Database Logging
Properly add database logging to the main scanner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_main_scanner():
    """Fix the main scanner by properly adding database logging"""
    
    # Read the backup file
    with open('main_scanner_backup.py', 'r') as f:
        content = f.read()
    
    # Add the import at the top
    if 'from trade_logger import TradeLogger' not in content:
        # Find the imports section
        import_section = content.find('import logging')
        if import_section != -1:
            # Add the import after the existing imports
            content = content[:import_section] + 'from trade_logger import TradeLogger\n' + content[import_section:]
    
    # Find the correct location to add database logging
    # Look for the section where quality_results are processed and before the summary report
    insert_marker = '# Send summary report'
    insert_point = content.find(insert_marker)
    
    if insert_point != -1:
        # Add database logging before the summary report
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
            
'''
        
        # Insert the database logging code before the summary report
        content = content[:insert_point] + db_logging_code + '\n            ' + content[insert_point:]
    
    # Write the fixed content back to the main scanner
    with open('main_scanner.py', 'w') as f:
        f.write(content)
    
    print("✅ Main scanner fixed with proper database logging")

if __name__ == "__main__":
    print("🔧 Fixing Main Scanner Database Logging")
    print("=" * 50)
    
    fix_main_scanner()
    
    print("\n✅ Main scanner fixed!")
    print("🎯 The main scanner should now run without syntax errors") 