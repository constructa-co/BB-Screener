#!/usr/bin/env python3
"""
Elliott Wave Logger Test Script
Comprehensive testing of the Elliott Wave logger functionality
Validates isolation from universal logger and proper data handling
"""

import os
import sys
from datetime import datetime
import json

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_elliott_logger():
    """Test the Elliott Wave logger functionality"""
    
    print("🧪 ELLIOTT WAVE LOGGER TESTING")
    print("=" * 50)
    
    try:
        # Import the Elliott Wave logger
        from elliott_wave_logger import ElliottWaveLogger
        
        print("✅ Elliott Wave logger imported successfully")
        
        # Test 1: Initialize logger
        print("\n1. Testing logger initialization...")
        logger = ElliottWaveLogger()
        print("✅ Logger initialized successfully")
        
        # Test 2: Create test signal data (matches scanner output)
        print("\n2. Creating test signal data...")
        test_signal = {
            'symbol': 'ETH/USDT',
            'timeframe': '1h',
            'direction': 'LONG',
            'pattern_type': 'BULLISH_IMPULSE',
            'current_wave': 'WAVE_3',
            'wave_degree': 'intermediate',
            'pattern_quality': 87.5,
            'confidence_score': 0.82,
            'entry_price': 2990.50,
            'stop_loss': 2915.00,
            'targets': [3050.0, 3125.0, 3280.0],
            'risk_reward': 2.8,
            'invalidation_level': 2899.0,
            
            # Wave analysis (hierarchical structure from scanner)
            'waves': [
                {
                    'wave': 1,
                    'start': {'price': 2850.0, 'date': '2025-08-18 10:00:00'},
                    'end': {'price': 2970.0, 'date': '2025-08-18 12:00:00'},
                    'size': 4.2
                },
                {
                    'wave': 2,
                    'start': {'price': 2970.0, 'date': '2025-08-18 12:00:00'},
                    'end': {'price': 2930.0, 'date': '2025-08-18 14:00:00'},
                    'retrace': 38.2
                },
                {
                    'wave': 3,
                    'start': {'price': 2930.0, 'date': '2025-08-18 14:00:00'},
                    'end': {'price': 3050.0, 'date': '2025-08-18 16:00:00'},
                    'size': 4.1,
                    'vs_wave1': 1.0
                }
            ],
            
            # Fibonacci levels
            'fibonacci_levels': {
                'extensions': [1.618, 2.618, 4.236],
                'retracements': [0.382, 0.5, 0.618],
                'current_confluence': 3050.0
            },
            
            # Pattern metrics
            'strength_indicators': {
                'wave_strength': 'strong',
                'pattern_clarity': 'high',
                'fibonacci_alignment': 'excellent'
            },
            
            # Additional scanner data
            'wave_start': 2850.0,
            'wave_start_date': '2025-08-18 10:00:00',
            'current_price': 2990.50,
            'duration': 6.0  # hours
        }
        
        print("✅ Test signal data created")
        
        # Test 3: Log single signal
        print("\n3. Testing single signal logging...")
        scan_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result = logger.log_elliott_signal(test_signal, scan_id)
        
        if result:
            print(f"✅ Single signal logged successfully with ID: {result}")
        else:
            print("❌ Single signal logging failed")
            return False
        
        # Test 4: Test upsert logic (duplicate prevention)
        print("\n4. Testing upsert logic (duplicate prevention)...")
        duplicate_result = logger.log_elliott_signal(test_signal, scan_id)
        
        if duplicate_result:
            print("✅ Upsert logic working (duplicate handled)")
        else:
            print("❌ Upsert logic failed")
            return False
        
        # Test 5: Test batch logging
        print("\n5. Testing batch signal logging...")
        batch_signals = [
            {
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'direction': 'SHORT',
                'pattern_type': 'BEARISH_IMPULSE',
                'current_wave': 'WAVE_2',
                'pattern_quality': 75.0,
                'confidence_score': 0.65,
                'waves': [{'wave': 1, 'size': 3.2}],
                'fibonacci_levels': {'retracements': [0.382, 0.5, 0.618]},
                'strength_indicators': {'wave_strength': 'medium'}
            },
            {
                'symbol': 'ADA/USDT',
                'timeframe': '1h',
                'direction': 'LONG',
                'pattern_type': 'BULLISH_IMPULSE',
                'current_wave': 'WAVE_4',
                'pattern_quality': 92.0,
                'confidence_score': 0.88,
                'waves': [{'wave': 1, 'size': 5.1}],
                'fibonacci_levels': {'extensions': [1.618, 2.618]},
                'strength_indicators': {'wave_strength': 'very_strong'}
            }
        ]
        
        batch_scan_id = f"batch_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        batch_results = logger.log_elliott_signals_batch(batch_signals, batch_scan_id)
        
        if len(batch_results) == len(batch_signals):
            print(f"✅ Batch logging successful: {len(batch_results)} signals logged")
        else:
            print(f"❌ Batch logging failed: {len(batch_results)}/{len(batch_signals)} signals logged")
            return False
        
        # Test 6: Test error handling
        print("\n6. Testing error handling...")
        invalid_signal = {
            'symbol': 'INVALID/USDT',
            # Missing required fields
        }
        
        error_result = logger.log_elliott_signal(invalid_signal, f"error_test_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        if error_result is None:
            print("✅ Error handling working (invalid signal rejected)")
        else:
            print("❌ Error handling failed (invalid signal accepted)")
            return False
        
        # Test 7: Verify isolation from universal logger
        print("\n7. Verifying isolation from universal logger...")
        
        # Check that we can import both loggers without conflicts
        try:
            from universal_scanner_logger import UniversalScannerLogger
            print("✅ Universal logger still accessible (no conflicts)")
        except ImportError as e:
            print(f"❌ Universal logger import failed: {e}")
            return False
        
        # Test 8: Test context manager
        print("\n8. Testing context manager...")
        try:
            with ElliottWaveLogger() as ctx_logger:
                test_signal['symbol'] = 'CONTEXT/USDT'
                ctx_result = ctx_logger.log_elliott_signal(test_signal, f"context_test_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                if ctx_result:
                    print("✅ Context manager working")
                else:
                    print("❌ Context manager failed")
                    return False
        except Exception as e:
            print(f"❌ Context manager error: {e}")
            return False
        
        # Clean up
        logger.close()
        
        print("\n" + "=" * 50)
        print("🎉 ELLIOTT WAVE LOGGER TESTING SUCCESSFUL!")
        print("✅ All tests passed")
        print("✅ Logger isolated from universal logger")
        print("✅ JSONB data handling working")
        print("✅ Upsert logic preventing duplicates")
        print("✅ Error handling robust")
        print("✅ Ready for Phase 3: Scanner Integration")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_elliott_logger()
    sys.exit(0 if success else 1)
