#!/usr/bin/env python3
"""
Wrapper script to run BB Scanner with NumPy 2.x compatibility fix
This ensures the patch is applied before any imports
"""

import sys
import numpy as np

print("🔧 Applying NumPy 2.x compatibility fix...")

# ULTRA-AGGRESSIVE PATCH - Apply before ANY other imports
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if 'NaN' not in np.__dict__:
    np.__dict__['NaN'] = np.nan
if hasattr(np, '__all__') and 'NaN' not in np.__all__:
    np.__all__.append('NaN')

print("✅ NumPy patch applied successfully")

# Now import and run the main scanner
try:
    import main_scanner
    print("✅ Main scanner imported successfully")
    
    # Run the scanner
    if __name__ == "__main__":
        main_scanner.main()
        
except Exception as e:
    print(f"❌ Error running scanner: {e}")
    sys.exit(1) 