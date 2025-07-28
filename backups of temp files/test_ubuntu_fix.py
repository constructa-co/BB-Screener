#!/usr/bin/env python3
"""
Test script for NumPy 2.x compatibility fix on Ubuntu server
Run this to verify the fix works before running the main scanner
"""

print("🧪 Testing NumPy 2.x Compatibility Fix")
print("=" * 50)

try:
    # Test 1: Import the patch
    print("1. Importing numpy_patch...")
    import numpy_patch
    print("   ✅ numpy_patch imported successfully")
    
    # Test 2: Check if NaN is available
    print("2. Checking NaN availability...")
    import numpy as np
    if hasattr(np, 'NaN'):
        print(f"   ✅ np.NaN is available: {np.NaN}")
    else:
        print("   ❌ np.NaN is NOT available")
        exit(1)
    
    # Test 3: Try direct import
    print("3. Testing direct NaN import...")
    from numpy import NaN
    print(f"   ✅ Direct import works: {NaN}")
    
    # Test 4: Test pandas_ta (common dependency that might use NaN)
    print("4. Testing pandas_ta import...")
    import pandas_ta as ta
    print("   ✅ pandas_ta imported successfully")
    
    # Test 5: Test main scanner import
    print("5. Testing main scanner import...")
    import main_scanner
    print("   ✅ Main scanner imported successfully")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("The NumPy 2.x compatibility fix is working correctly.")
    print("You can now run: python3 main_scanner.py")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    print("The fix is not working correctly.")
    exit(1) 