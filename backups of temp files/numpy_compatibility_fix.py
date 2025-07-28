"""
NumPy Compatibility Fix for BB Scanner
This file ensures compatibility with NumPy 2.x where NaN was removed
"""

import numpy as np

# Ensure np.nan is available for all modules
if not hasattr(np, 'nan'):
    # Fallback for older NumPy versions
    np.nan = float('nan')

# Create a compatibility layer for any code that might import NaN
try:
    from numpy import NaN
except ImportError:
    # NumPy 2.x compatibility - NaN was removed
    NaN = np.nan

# Ensure pandas can handle NaN properly
import pandas as pd
if not hasattr(pd, 'isna'):
    pd.isna = lambda x: x is None or (hasattr(x, '__float__') and np.isnan(float(x)))

print("✅ NumPy compatibility layer loaded successfully") 