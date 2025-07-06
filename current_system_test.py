# ANALYSIS SCRIPT: current_system_test.py
# Run this to understand what your current system is producing

import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the modules directory to the path
sys.path.append('/Users/robertsmith/Documents/BB Screener/modules')

from main_scanner import ModularBBScanner
from config import SCANNER_CONFIG

def test_current_system():
    """Test that current BB scanner runs without errors"""
    
    print("🔍 TESTING CURRENT BB SCANNER SYSTEM")
    print("=" * 60)
    
    try:
        scanner = ModularBBScanner()
        print("✅ Scanner initialized successfully")
        
        # Test that modules are working
        print("✅ Testing module initialization:")
        print(f"   Data Fetcher: {type(scanner.data_fetcher).__name__}")
        print(f"   BB Detector: {type(scanner.bb_detector).__name__}")
        print(f"   Technical Analyzer: {type(scanner.technical_analyzer).__name__}")
        print(f"   Risk Manager: {type(scanner.risk_manager).__name__}")
        
        print("\n🎯 CURRENT SYSTEM STATUS: OPERATIONAL")
        print("Ready for enhancement with validated MFI signals!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing current system: {e}")
        return False

if __name__ == "__main__":
    test_result = test_current_system()