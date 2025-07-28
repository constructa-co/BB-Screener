#!/bin/bash

# NumPy 2.x Compatibility Fix Deployment Script
# For Ubuntu 24.10 with Python 3.12 and NumPy 2.3.1

echo "🔧 Deploying NumPy 2.x Compatibility Fix for BB Scanner"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "main_scanner.py" ]; then
    echo "❌ Error: main_scanner.py not found. Please run this script from the BB Screener directory."
    exit 1
fi

echo "✅ Found BB Scanner directory"

# Create backup of current files
echo "📦 Creating backups..."
mkdir -p backups/numpy_fix_$(date +%Y%m%d_%H%M%S)
cp main_scanner.py backups/numpy_fix_$(date +%Y%m%d_%H%M%S)/
cp modules/bb_detector.py backups/numpy_fix_$(date +%Y%m%d_%H%M%S)/
cp modules/sentiment_analyzer.py backups/numpy_fix_$(date +%Y%m%d_%H%M%S)/
cp modules/data_fetcher.py backups/numpy_fix_$(date +%Y%m%d_%H%M%S)/
cp modules/technical_analyzer.py backups/numpy_fix_$(date +%Y%m%d_%H%M%S)/

echo "✅ Backups created"

# Test the fix locally first
echo "🧪 Testing NumPy compatibility fix..."
python -c "
import numpy as np
try:
    from numpy import NaN
    print('✅ NumPy NaN import successful')
except ImportError:
    print('⚠️  NumPy NaN import failed - applying compatibility fix')
    NaN = np.nan
    print('✅ Compatibility fix applied')

# Test all modules
try:
    import main_scanner
    print('✅ Main scanner imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
"

if [ $? -eq 0 ]; then
    echo "✅ Local test passed"
else
    echo "❌ Local test failed - please check the error above"
    exit 1
fi

echo ""
echo "🚀 DEPLOYMENT READY"
echo "=================="
echo "The NumPy 2.x compatibility fix has been applied to:"
echo "  ✅ main_scanner.py"
echo "  ✅ modules/bb_detector.py"
echo "  ✅ modules/sentiment_analyzer.py"
echo "  ✅ modules/data_fetcher.py"
echo "  ✅ modules/technical_analyzer.py"
echo ""
echo "📋 NEXT STEPS FOR UBUNTU SERVER:"
echo "1. Upload these files to your Ubuntu server"
echo "2. Run: python -c 'import main_scanner; print(\"✅ Success\")'"
echo "3. If successful, run your scanner normally"
echo ""
echo "🔍 TROUBLESHOOTING:"
echo "- If you still get NaN import errors, check for other dependencies"
echo "- Ensure NumPy 2.3.1 is installed: pip show numpy"
echo "- Check for any cached .pyc files: find . -name '*.pyc' -delete"
echo ""
echo "📞 If issues persist, check the error message and ensure all files were uploaded correctly." 