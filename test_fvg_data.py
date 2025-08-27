#!/usr/bin/env python3
"""
Test script to debug FVG data structure
"""
import sys
import os
import importlib.util

# Dynamic import to handle directory with numbers
scanner_path = os.path.join("manual_scanners", "1_min_scanners", "fair_value_gap_scanner_1m_r1.py")
spec = importlib.util.spec_from_file_location("fvg_r1", scanner_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FairValueGapScanner1MR1 = module.FairValueGapScanner1MR1

def main():
    scanner = FairValueGapScanner1MR1()
    results = scanner.base_scanner().scan_for_fvg_fibonacci_setups()
    
    if not results:
        print("No results found")
        return
    
    # Get first result
    result = results[0]
    print("Sample result keys:", list(result.keys()))
    print("\nGap data:")
    print(result.get('gap', {}))
    print("\nTrade data:")
    print(result.get('trade', {}))
    print("\nFibonacci data:")
    print(result.get('fibonacci', {}))
    
    # Test data extraction
    fvg_data = scanner._extract_fvg_data(result)
    print("\nExtracted FVG data:")
    for key, value in fvg_data.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
