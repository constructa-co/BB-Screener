# 🚀 DigitalOcean Deployment Guide

## ✅ Step 1: Code Pushed to GitHub (COMPLETED)
All NumPy 2.x compatibility fixes have been successfully pushed to GitHub:
- `numpy_patch.py` - Global NumPy compatibility fix
- `run_scanner.py` - Wrapper script for guaranteed compatibility
- `test_ubuntu_fix.py` - Test script to verify the fix
- `main_scanner.py` - Updated with ultra-aggressive NumPy patch

## 🔗 Step 2: Connect to Your DigitalOcean Server

```bash
ssh root@165.232.160.52
```

## 📥 Step 3: Navigate to Project Directory

```bash
cd /opt/bb-screener
```

## 📥 Step 4: Pull Latest Code from GitHub

```bash
git pull origin master
```

**Expected Output:**
```
remote: Enumerating objects: 71, done.
remote: Counting objects: 100% (71/71), done.
remote: Compressing objects: 100% (59/59), done.
Unpacking objects: 100% (59/59), done.
From https://github.com/constructa-co/BB-Screener
   92b24bd..fbec0d7  master     -> origin/master
Updating 92b24bd..fbec0d7
Fast-forward
 ... (files updated)
```

## 🧪 Step 5: Verify Files Are Present

```bash
ls -la *.py | grep -E "(numpy_patch|run_scanner|test_ubuntu)"
```

**Expected Output:**
```
-rw-r--r-- 1 root root  664 Jul 20 00:35 numpy_patch.py
-rwxr-xr-x 1 root root  833 Jul 20 00:37 run_scanner.py
-rw-r--r-- 1 root root 1458 Jul 20 00:35 test_ubuntu_fix.py
```

## 🧪 Step 6: Test the NumPy Fix

```bash
python3 test_ubuntu_fix.py
```

**Expected Output:**
```
🧪 Testing NumPy 2.x Compatibility Fix
==================================================
1. Importing numpy_patch...
✅ NumPy 2.x compatibility patch applied globally
✅ np.NaN available: True
✅ np.NaN value: nan
   ✅ numpy_patch imported successfully
2. Checking NaN availability...
   ✅ np.NaN is available: nan
3. Testing direct NaN import...
   ✅ Direct import works: nan
4. Testing pandas_ta import...
   ✅ pandas_ta imported successfully
5. Testing main scanner import...
   ✅ Main scanner imported successfully

🎉 ALL TESTS PASSED!
The NumPy 2.x compatibility fix is working correctly.
You can now run: python3 main_scanner.py
```

## 🚀 Step 7: Run the BB Scanner

### Option A: Using Wrapper Script (Recommended)
```bash
python3 run_scanner.py
```

### Option B: Direct Execution
```bash
python3 main_scanner.py
```

## 🔧 Troubleshooting

### If git pull fails:
```bash
git status
git stash
git pull origin master
```

### If test fails:
```bash
# Check NumPy version
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"

# Test patch directly
python3 -c "import numpy_patch; print('Patch applied')"

# Check if NaN is available
python3 -c "import numpy; print('NaN available:', hasattr(numpy, 'NaN'))"
```

### If scanner fails:
```bash
# Test main scanner import
python3 -c "import main_scanner; print('Main scanner loads')"

# Check for specific import errors
python3 -c "import numpy_patch; import main_scanner; print('✅ Success')"
```

## 📊 Expected Results

After successful deployment, you should see:

1. **Market Analysis Output:**
   - Exchange connections established
   - Coin data fetching
   - BB detection results

2. **Trading Opportunities:**
   - Premium trades identified
   - Risk/reward calculations
   - Sentiment analysis

3. **Excel Output:**
   - Files saved to `outputs/excel_reports/`
   - Market metadata sheets
   - Comprehensive analysis data

## 🎯 Success Indicators

- ✅ No "Import error: cannot import name 'NaN'" messages
- ✅ Scanner starts without errors
- ✅ Market data is fetched successfully
- ✅ BB analysis runs and produces results
- ✅ Excel files are generated

## 🌐 Cloud vs Local

**Local (Your Mac):**
- Runs when you execute the script
- Stops when you close terminal

**DigitalOcean (Singapore Server):**
- Runs 24/7 continuously
- Processes market data in real-time
- Generates reports automatically
- Accessible from anywhere

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting commands above
2. Verify all files are present after git pull
3. Ensure NumPy 2.3.1 is installed on the server
4. Test the fix step by step using the test script

**Ready to deploy! 🚀** 