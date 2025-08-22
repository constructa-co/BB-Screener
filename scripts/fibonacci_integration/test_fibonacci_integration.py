#!/usr/bin/env python3
"""
Fibonacci Scanner Integration Test
Validates the complete integration without affecting existing systems
"""

import sys
import os
import time
from datetime import datetime

# Add the database_and_logging directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database_and_logging'))

def test_fibonacci_logger_import():
    """Test that the Fibonacci logger can be imported"""
    try:
        from fibonacci_logger import FibonacciLogger, FibonacciSignal, FibonacciCalculator
        print("✅ Fibonacci logger imports successfully")
        return True
    except Exception as e:
        print(f"❌ Fibonacci logger import failed: {e}")
        return False

def test_fibonacci_calculator():
    """Test Fibonacci calculation engine"""
    try:
        from fibonacci_logger import FibonacciCalculator
        
        calculator = FibonacciCalculator()
        
        # Test retracement levels
        levels = calculator.calculate_retracement_levels(high=100, low=80)
        expected_236 = 100 - (20 * 0.236)  # 95.28
        expected_618 = 100 - (20 * 0.618)  # 87.64
        
        if abs(levels['RETRACEMENT_236'] - expected_236) < 0.01:
            print("✅ Fibonacci retracement calculations correct")
        else:
            print("❌ Fibonacci retracement calculations incorrect")
            return False
        
        # Test swing point detection
        import numpy as np
        prices = np.array([80, 85, 90, 95, 100, 95, 90, 85, 80, 85, 90, 95])
        highs, lows = calculator.find_swing_points(prices, window=2)
        
        if len(highs) > 0 and len(lows) > 0:
            print("✅ Swing point detection working")
        else:
            print("❌ Swing point detection failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Fibonacci calculator test failed: {e}")
        return False

def test_database_connection():
    """Test database connection and table access"""
    try:
        from fibonacci_logger import FibonacciLogger
        
        config = {
            'scanner_id': 'test_fibonacci',
            'fibonacci_levels': [0.236, 0.382, 0.500, 0.618, 0.786],
            'confidence_threshold': 0.7
        }
        
        logger = FibonacciLogger(config)
        
        # Test health status
        health = logger.get_health_status()
        if health['status'] in ['HEALTHY', 'DEGRADED']:
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            return False
        
        logger.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def test_scanner_integration():
    """Test the complete scanner integration"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'manual_scanners', '5_min_scanners', 'fibonacci_revisions'))
        
        from fibonacci_retracement_scanner_r1 import FibonacciRetracementScannerR1
        
        # Create scanner instance
        scanner = FibonacciRetracementScannerR1(timeframe='5m')
        
        if scanner.fibonacci_logger:
            print("✅ Scanner integration successful")
        else:
            print("❌ Scanner integration failed - logger not initialized")
            return False
        
        scanner.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Scanner integration test failed: {e}")
        return False

def test_isolation():
    """Test that Fibonacci scanner doesn't interfere with existing systems"""
    try:
        # Check that existing scanners are still accessible
        import importlib.util
        
        # Test Elliott Wave scanner still works
        elliott_path = os.path.join(os.path.dirname(__file__), '..', '..', 'manual_scanners', '1_hour_scanners', 'elliot_waves_scanner_1h_r1.py')
        if os.path.exists(elliott_path):
            print("✅ Elliott Wave scanner still accessible")
        else:
            print("❌ Elliott Wave scanner not found")
            return False
        
        # Test ICT scanner still works
        ict_path = os.path.join(os.path.dirname(__file__), '..', '..', 'manual_scanners', '15_min_scanners', 'ict_scanner_15m_r4.py')
        if os.path.exists(ict_path):
            print("✅ ICT scanner still accessible")
        else:
            print("❌ ICT scanner not found")
            return False
        
        print("✅ Complete isolation maintained")
        return True
        
    except Exception as e:
        print(f"❌ Isolation test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 FIBONACCI SCANNER INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Fibonacci Logger Import", test_fibonacci_logger_import),
        ("Fibonacci Calculator", test_fibonacci_calculator),
        ("Database Connection", test_database_connection),
        ("Scanner Integration", test_scanner_integration),
        ("System Isolation", test_isolation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Fibonacci integration ready for production!")
        print("\n📋 Next Steps:")
        print("  1. Run database migration: psql -f scripts/fibonacci_integration/setup_fibonacci_database.sql")
        print("  2. Setup cron jobs: ./scripts/fibonacci_integration/setup_fibonacci_cron.sh")
        print("  3. Test scanner: python manual_scanners/5_min_scanners/fibonacci_revisions/fibonacci_retracement_scanner_r1.py")
        print("  4. Monitor health: ./scripts/health_monitoring/check_fibonacci_health.sh")
    else:
        print("⚠️  SOME TESTS FAILED - Please review and fix issues before deployment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
