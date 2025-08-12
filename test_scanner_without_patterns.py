#!/usr/bin/env python3

"""
Test script to run BB Scanner without pattern analysis
This will help isolate if pattern analysis is causing the hang
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_scanner_without_patterns():
    """Test the scanner without pattern analysis"""
    
    print("🧪 TESTING BB SCANNER WITHOUT PATTERN ANALYSIS")
    print("="*60)
    
    try:
        # Import the scanner
        from main_scanner import ModularBBScanner
        
        # Create scanner instance
        scanner = ModularBBScanner()
        
        print("✅ Scanner initialized successfully")
        
        # Test with just one symbol to see if it completes
        print("\n🎯 Testing with BTC only...")
        start_time = time.time()
        
        # Run analysis for BTC only
        results = scanner.analyze_coin_comprehensive('BTC', 'binance')
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Analysis completed in {duration:.2f} seconds")
        print(f"📊 Result: {results.get('setup_type', 'NONE')} setup")
        
        if results.get('setup_type') != 'NONE':
            print(f"   Entry: {results.get('entry', 0)}")
            print(f"   Stop: {results.get('stop', 0)}")
            print(f"   Target: {results.get('target1', 0)}")
            print(f"   Probability: {results.get('probability', 0)}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scanner_without_patterns()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
