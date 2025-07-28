"""
Global NumPy 2.x Compatibility Patch
This MUST be imported before any other modules to fix NaN import issues
"""

import sys
import numpy as np

# AGGRESSIVE PATCH: Add NaN to numpy module immediately
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Also patch the module's __dict__ to ensure it's available for imports
if 'NaN' not in np.__dict__:
    np.__dict__['NaN'] = np.nan

# Patch the module's __all__ if it exists
if hasattr(np, '__all__') and 'NaN' not in np.__all__:
    np.__all__.append('NaN')

print("✅ NumPy 2.x compatibility patch applied globally")
print(f"✅ np.NaN available: {hasattr(np, 'NaN')}")
print(f"✅ np.NaN value: {np.NaN}") 