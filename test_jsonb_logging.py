#!/usr/bin/env python3
"""
Test script to verify enhanced trade logger captures all fields in JSONB
"""
from trade_logger import TradeLogger
import json
from datetime import datetime

def test_jsonb_logging():
    """Test logging a single trade with 100+ fields"""
    
    # Create a test trade with 100+ fields (simulating the 139 fields from scanner)
    test_trade = {
        'symbol': 'TEST/USDT',
        'exchange': 'binance',
        'probability': 85.5,
        'entry_price': 100.0,
        'stop_loss': 95.0,
        'target_1': 105.0,
        'target_2': 110.0,
        'target_3': 115.0,
        'bb_score': 87.2,
        'risk_reward_ratio': 2.5,
        'current_price': 101.5,
        'rsi': 65.4,
        'mfi': 72.1,
        'stochastic_k': 78.9,
        'volume_surge': 1.8,
        'macd_signal': 'bullish',
        'pattern_type': 'BB_Bounce',
        'pattern_quality': 'HIGH',
        'confluence_score': 82.3,
        'historical_win_rate': 0.75,
        'category_win_rate': 0.68,
        'similar_setups_count': 15,
        'market_cap': 50000000.0,
        'volume_24h': 2500000.0,
        'price_change_24h': 3.2,
        
        # Add 80+ additional fields to simulate full scanner output
        'setup_type': 'BB_Bounce',
        'timeframe': '15m',
        'direction': 'LONG',
        'tier': 'PREMIUM',
        'bb_upper': 102.5,
        'bb_middle': 100.0,
        'bb_lower': 97.5,
        'bb_width': 0.05,
        'bb_position': 0.6,
        'volume_ratio': 1.8,
        'atr': 2.5,
        'cci': 45.2,
        'adx': 28.7,
        'williams_r': -25.3,
        'obv': 1250000.0,
        'vwap': 100.2,
        'vwap_deviation': 0.013,
        'support_level': 98.0,
        'resistance_level': 103.0,
        'pivot_point': 100.5,
        'fib_236': 99.2,
        'fib_382': 98.8,
        'fib_500': 98.5,
        'fib_618': 98.2,
        'fib_786': 97.8,
        'gap_high': 102.8,
        'gap_low': 99.2,
        'gap_size_pct': 3.6,
        'gap_type': 'bullish',
        'swing_high': 103.5,
        'swing_low': 97.0,
        'order_block_high': 101.8,
        'order_block_low': 99.5,
        'breaker_block_high': 102.2,
        'breaker_block_low': 98.8,
        'fvg_high': 102.0,
        'fvg_low': 99.0,
        'liquidity_sweep_level': 96.5,
        'equilibrium_level': 100.2,
        'volume_surge_multiplier': 1.8,
        'relative_volume': 2.1,
        'volume_profile_poc': 100.1,
        'pattern_target': 108.0,
        'pattern_reliability': 0.82,
        'major_resistance_1': 103.5,
        'major_support_1': 97.0,
        'momentum_score': 78.5,
        'volatility_score': 65.2,
        'trend_strength': 72.8,
        'volume_quality': 85.3,
        'price_action_score': 79.1,
        'technical_score': 81.7,
        'fundamental_score': 68.4,
        'sentiment_score': 73.9,
        'overall_score': 80.2,
        'risk_level': 'MEDIUM',
        'confidence_interval': 0.85,
        'expected_return': 0.045,
        'max_drawdown': 0.025,
        'sharpe_ratio': 1.8,
        'sortino_ratio': 2.1,
        'calmar_ratio': 1.6,
        'profit_factor': 2.3,
        'expectancy': 0.032,
        'kelly_criterion': 0.28,
        'optimal_position_size': 0.15,
        'market_regime': 'BULLISH',
        'sector_performance': 0.042,
        'correlation_btc': 0.65,
        'correlation_eth': 0.58,
        'beta': 1.2,
        'alpha': 0.018,
        'information_ratio': 0.85,
        'tracking_error': 0.023,
        'var_95': 0.035,
        'cvar_95': 0.042,
        'ulcer_index': 0.015,
        'pain_ratio': 2.8,
        'calmar_ratio': 1.6,
        'sterling_ratio': 1.9,
        'burke_ratio': 1.7,
        'kestner_ratio': 1.5,
        'martin_ratio': 1.8,
        'gain_to_pain_ratio': 2.1,
        'profit_factor': 2.3,
        'recovery_factor': 1.9,
        'risk_adjusted_return': 0.045,
        'volatility_adjusted_return': 0.038,
        'downside_deviation': 0.018,
        'upside_potential_ratio': 1.6,
        'omega_ratio': 1.8,
        'kappa_ratio': 1.7,
        'gamma_ratio': 1.9,
        'berk_ratio': 1.6,
        'modigliani_ratio': 1.8,
        'treynor_ratio': 2.1,
        'jensen_alpha': 0.012,
        'appraisal_ratio': 0.85,
        'information_ratio': 0.85,
        'tracking_error': 0.023,
        'var_95': 0.035,
        'cvar_95': 0.042,
        'ulcer_index': 0.015,
        'pain_ratio': 2.8,
        'calmar_ratio': 1.6,
        'sterling_ratio': 1.9,
        'burke_ratio': 1.7,
        'kestner_ratio': 1.5,
        'martin_ratio': 1.8,
        'gain_to_pain_ratio': 2.1,
        'profit_factor': 2.3,
        'recovery_factor': 1.9,
        'risk_adjusted_return': 0.045,
        'volatility_adjusted_return': 0.038,
        'downside_deviation': 0.018,
        'upside_potential_ratio': 1.6,
        'omega_ratio': 1.8,
        'kappa_ratio': 1.7,
        'gamma_ratio': 1.9,
        'berk_ratio': 1.6,
        'modigliani_ratio': 1.8,
        'treynor_ratio': 2.1,
        'jensen_alpha': 0.012,
        'appraisal_ratio': 0.85
    }
    
    print(f"🧪 Test trade has {len(test_trade)} fields")
    print(f"📊 Sample fields: {list(test_trade.keys())[:15]}...")
    
    # Initialize logger
    logger = TradeLogger()
    if not logger.connection:
        print("❌ Database connection failed")
        return False
    
    try:
        # Create a test scan
        scan_id = logger.log_scan_start('test_scan', version='1.0')
        if not scan_id:
            print("❌ Failed to create test scan")
            return False
        
        print(f"📊 Created test scan with ID: {scan_id}")
        
        # Log the test trade
        success = logger.log_trade_opportunity(scan_id, test_trade)
        
        if success:
            print("✅ Test trade logged successfully")
            
            # Verify the JSONB contains all fields
            logger.cursor.execute("""
                SELECT scanner_specific_data 
                FROM trade_opportunities 
                WHERE symbol = 'TEST/USDT'
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            result = logger.cursor.fetchone()
            if result and result['scanner_specific_data']:
                jsonb_data = result['scanner_specific_data']
                field_count = len(jsonb_data)
                print(f"✅ JSONB contains {field_count} fields")
                
                if field_count >= 100:
                    print("🎉 SUCCESS: Enhanced logger is working correctly!")
                    print(f"📊 Sample JSONB fields: {list(jsonb_data.keys())[:10]}...")
                    return True
                else:
                    print(f"❌ JSONB only has {field_count} fields (expected 100+)")
                    return False
            else:
                print("❌ JSONB is NULL or empty")
                return False
        else:
            print("❌ Failed to log test trade")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        logger.close()

if __name__ == "__main__":
    print("🧪 Testing Enhanced Trade Logger JSONB Functionality")
    print("=" * 60)
    
    success = test_jsonb_logging()
    
    if success:
        print("\n🎉 TEST PASSED: Enhanced logger ready for production!")
    else:
        print("\n❌ TEST FAILED: Need to debug the logger")
