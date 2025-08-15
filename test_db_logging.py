#!/usr/bin/env python3
"""
Quick test script to verify database logging works without running the full scanner
Tests the JSON serialization fix with realistic trade data
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_database_logging():
    """Test database logging with realistic trade data"""
    
    print("🧪 Testing Database Logging (Quick Test)")
    print("="*50)
    
    try:
        from trade_logger import TradeLogger
        
        # Create a realistic trade record with NumPy types (like the scanner produces)
        test_trade = {
            'symbol': 'BTCUSDT',
            'exchange': 'Binance',
            'timeframe': '4H',
            'bb_score': np.int64(85),
            'probability': np.float64(92.5),
            'risk_reward_ratio': np.float64(2.8),
            'current_price': np.float64(45000.0),
            'entry_price': np.float64(44800.0),
            'stop_loss': np.float64(44000.0),
            'target_1': np.float64(46000.0),
            'target_2': np.float64(47000.0),
            'target_3': np.float64(48000.0),
            'rsi': np.float64(65.2),
            'mfi': np.float64(78.9),
            'stochastic_k': np.float64(82.1),
            'volume_surge': np.int64(3),
            'macd_signal': 'bullish',
            'pattern_type': 'BB Bounce',
            'pattern_quality': 'EXCELLENT',
            'confluence_score': np.float64(94.2),
            'historical_win_rate': np.float64(78.5),
            'category_win_rate': np.float64(81.3),
            'similar_setups_count': np.int64(12),
            'market_cap': np.int64(25),
            'volume_24h': np.float64(9876543.21),
            'price_change_24h': np.float64(3.2),
            'scanner_type': 'bb_scanner',
            'scanner_specific_data': {
                'volume_surge_detected': np.bool_(True),
                'mfi_priority_signal': np.bool_(False),
                'bb_squeeze_active': np.bool_(True),
                'bb_expansion_active': np.bool_(False),
                'bb_expansion_ratio': np.float64(1.85),
                'bb_position': np.float64(0.92),
                'bb_trend_direction': 'Sideways',
                'cmf_value': np.float64(0.156),
                'cci_value': np.float64(125.4),
                'volume_multiplier': np.float64(2.3),
                'setup_quality_score': np.int64(24),
                'risk_level': 'MEDIUM',
                'entry_confidence': np.float64(88.5),
                'stop_loss_distance': np.float64(2.1),
                'target_distance': np.float64(4.8),
                'market_cap_rank': np.int64(1),
                'liquidity_score': np.float64(95.2),
                'volatility_24h': np.float64(3.8),
                'correlation_btc': np.float64(0.72),
                'support_level': np.float64(44500.0),
                'resistance_level': np.float64(45500.0),
                'fibonacci_levels': {
                    'fib_236': np.float64(44650.0),
                    'fib_382': np.float64(44800.0),
                    'fib_500': np.float64(45000.0),
                    'fib_618': np.float64(45200.0),
                    'fib_786': np.float64(45400.0)
                },
                'technical_indicators': {
                    'ema_20': np.float64(44850.0),
                    'ema_50': np.float64(44700.0),
                    'sma_200': np.float64(44500.0),
                    'bollinger_upper': np.float64(45200.0),
                    'bollinger_lower': np.float64(44400.0),
                    'bollinger_middle': np.float64(44800.0)
                },
                'volume_analysis': {
                    'volume_sma_ratio': np.float64(2.3),
                    'volume_trend': 'Increasing',
                    'volume_surge_threshold': np.float64(1.5),
                    'volume_consistency': np.float64(85.2)
                },
                'momentum_indicators': {
                    'rsi_trend': 'Bullish',
                    'macd_histogram': np.float64(125.4),
                    'macd_signal': np.float64(120.1),
                    'macd_line': np.float64(130.7),
                    'stochastic_rsi': np.float64(75.3),
                    'williams_r': np.float64(-25.6)
                },
                'pattern_recognition': {
                    'candlestick_pattern': 'Hammer',
                    'chart_pattern': 'Bull Flag',
                    'support_resistance': 'Strong Support',
                    'breakout_potential': np.float64(78.5),
                    'pattern_reliability': np.float64(82.3)
                }
            }
        }
        
        print(f"📊 Test trade contains {len(test_trade)} fields")
        print(f"   • NumPy types: {sum(1 for v in test_trade.values() if isinstance(v, (np.bool_, np.bool8, np.integer, np.floating, np.ndarray)))}")
        print(f"   • Nested NumPy types: {sum(1 for v in test_trade['scanner_specific_data'].values() if isinstance(v, (np.bool_, np.bool8, np.integer, np.floating, np.ndarray)))}")
        
        # Test database connection
        logger = TradeLogger()
        if not logger.connection:
            print("❌ Database connection failed")
            return False
            
        print("✅ Database connection successful")
        
        # Create a test scan
        scan_id = logger.log_scan_start('test_scanner', version='1.0')
        print(f"✅ Created test scan_id: {scan_id}")
        
        # Test the actual database logging
        print("\n🔍 Testing database logging...")
        try:
            success = logger.log_trade_opportunity(scan_id, test_trade)
            if success:
                print("✅ Database logging successful!")
                print("✅ JSON serialization fix is working!")
            else:
                print("❌ Database logging failed")
                return False
                
        except Exception as e:
            print(f"❌ Database logging error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Complete the scan
        logger.complete_scan(scan_id, 1, 1, 30)
        print("✅ Scan completed successfully")
        
        # Test querying the logged data
        print("\n🔍 Testing data retrieval...")
        try:
            logger.cursor.execute("""
                SELECT symbol, probability, bb_score, scanner_specific_data 
                FROM trade_opportunities 
                WHERE scan_id = %s
            """, (scan_id,))
            
            result = logger.cursor.fetchone()
            if result:
                print(f"✅ Data retrieved successfully:")
                print(f"   • Symbol: {result['symbol']}")
                print(f"   • Probability: {result['probability']}%")
                print(f"   • BB Score: {result['bb_score']}")
                print(f"   • Scanner Data: {len(str(result['scanner_specific_data']))} characters")
                
                # Test that the data can be accessed (it's already a dict from psycopg2)
                parsed_data = result['scanner_specific_data']
                print(f"   • Data access successful: {len(parsed_data)} fields")
                print(f"   • Sample field: volume_surge_detected = {parsed_data.get('volume_surge_detected')}")
                
            else:
                print("❌ No data found in database")
                return False
                
        except Exception as e:
            print(f"❌ Data retrieval error: {e}")
            return False
        
        logger.close()
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Database logging is working correctly")
        print("✅ JSON serialization fix is successful")
        print("✅ Ready for full scanner run")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_trades():
    """Test logging multiple trades to ensure consistency"""
    
    print("\n🧪 Testing Multiple Trade Logging")
    print("="*50)
    
    try:
        from trade_logger import TradeLogger
        
        logger = TradeLogger()
        if not logger.connection:
            print("❌ Database connection failed")
            return False
            
        scan_id = logger.log_scan_start('multi_test_scanner', version='1.0')
        
        # Create multiple test trades with different NumPy types
        test_trades = [
            {
                'symbol': 'ETHUSDT',
                'exchange': 'Binance',
                'timeframe': '4H',
                'bb_score': np.int64(78),
                'probability': np.float64(85.2),
                'risk_reward_ratio': np.float64(2.1),
                'current_price': np.float64(3200.0),
                'entry_price': np.float64(3180.0),
                'stop_loss': np.float64(3120.0),
                'target_1': np.float64(3300.0),
                'scanner_type': 'bb_scanner',
                'scanner_specific_data': {
                    'volume_surge_detected': np.bool_(False),
                    'bb_squeeze_active': np.bool_(True),
                    'setup_quality': np.int64(18)
                }
            },
            {
                'symbol': 'ADAUSDT',
                'exchange': 'Binance',
                'timeframe': '4H',
                'bb_score': np.int64(92),
                'probability': np.float64(88.7),
                'risk_reward_ratio': np.float64(3.2),
                'current_price': np.float64(0.45),
                'entry_price': np.float64(0.44),
                'stop_loss': np.float64(0.42),
                'target_1': np.float64(0.48),
                'scanner_type': 'bb_scanner',
                'scanner_specific_data': {
                    'volume_surge_detected': np.bool_(True),
                    'bb_expansion_active': np.bool_(True),
                    'setup_quality': np.int64(25)
                }
            }
        ]
        
        success_count = 0
        for i, trade in enumerate(test_trades, 1):
            try:
                success = logger.log_trade_opportunity(scan_id, trade)
                if success:
                    success_count += 1
                    print(f"✅ Trade {i} logged: {trade['symbol']}")
                else:
                    print(f"❌ Trade {i} failed: {trade['symbol']}")
            except Exception as e:
                print(f"❌ Trade {i} error: {e}")
        
        logger.complete_scan(scan_id, len(test_trades), success_count, 60)
        logger.close()
        
        print(f"\n📊 Multiple trade test results:")
        print(f"   • Trades attempted: {len(test_trades)}")
        print(f"   • Trades successful: {success_count}")
        print(f"   • Success rate: {(success_count/len(test_trades)*100):.1f}%")
        
        if success_count == len(test_trades):
            print("✅ All multiple trades logged successfully!")
            return True
        else:
            print("❌ Some trades failed to log")
            return False
            
    except Exception as e:
        print(f"❌ Multiple trade test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("QUICK DATABASE LOGGING TEST")
    print("="*60)
    
    # Run tests
    test1_passed = test_database_logging()
    test2_passed = test_multiple_trades()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Database logging is working correctly")
        print("✅ JSON serialization fix is successful")
        print("✅ Ready for full scanner run")
        print("\nExpected results in full scanner:")
        print("   • 'Logged trade BTC: 85% probability' (not 'Error logging trade BTC')")
        print("   • 'Scan 247 completed: 20 opportunities' (not '0 opportunities')")
        print("   • Database will have all trades with complete data")
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ Do not run full scanner until tests pass")
        sys.exit(1)
