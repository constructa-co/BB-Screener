#!/bin/bash
# File: /opt/bb-screener/test_all_scanners.sh

cd /opt/bb-screener
export $(cat .env | xargs)
export PYTHONPATH=.

# Create results file
RESULTS_FILE="scanner_test_results_$(date +%Y%m%d_%H%M%S).txt"

echo "=== COMPREHENSIVE SCANNER TEST ===" > $RESULTS_FILE
echo "Testing Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Function to test scanner
test_scanner() {
    local scanner_path=$1
    local scanner_name=$(basename $scanner_path)
    echo "Testing: $scanner_name" | tee -a $RESULTS_FILE
    
    # Run scanner with timeout and capture output
    timeout 10 python3 "$scanner_path" > /tmp/scanner_output.txt 2>&1
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS" | tee -a $RESULTS_FILE
        tail -3 /tmp/scanner_output.txt | tee -a $RESULTS_FILE
    elif [ $exit_code -eq 124 ]; then
        echo "⏱️ TIMEOUT (likely working but slow)" | tee -a $RESULTS_FILE
    else
        echo "❌ FAILED" | tee -a $RESULTS_FILE
        # Capture the error
        grep -E "Error|Exception|ModuleNotFound|TypeError|ImportError" /tmp/scanner_output.txt | head -3 | tee -a $RESULTS_FILE
    fi
    echo "---" | tee -a $RESULTS_FILE
}

# Test all scanner directories
echo "=== 1-MINUTE SCANNERS ===" | tee -a $RESULTS_FILE
for scanner in manual_scanners/1_min_scanners/*/*.py; do
    [ -f "$scanner" ] && test_scanner "$scanner"
done

echo "" | tee -a $RESULTS_FILE
echo "=== 5-MINUTE SCANNERS ===" | tee -a $RESULTS_FILE
for scanner in manual_scanners/5_min_scanners/*/*.py; do
    [ -f "$scanner" ] && test_scanner "$scanner"
done

echo "" | tee -a $RESULTS_FILE
echo "=== 15-MINUTE SCANNERS ===" | tee -a $RESULTS_FILE
for scanner in manual_scanners/15_min_scanners/*/*.py; do
    [ -f "$scanner" ] && test_scanner "$scanner"
done

echo "" | tee -a $RESULTS_FILE
echo "=== 1-HOUR SCANNERS ===" | tee -a $RESULTS_FILE
for scanner in manual_scanners/1_hour_scanners/*/*.py; do
    [ -f "$scanner" ] && test_scanner "$scanner"
done

echo "" | tee -a $RESULTS_FILE
echo "=== 4-HOUR SCANNERS ===" | tee -a $RESULTS_FILE
for scanner in manual_scanners/4_hour_scanners/*/*.py; do
    [ -f "$scanner" ] && test_scanner "$scanner"
done

echo "" | tee -a $RESULTS_FILE
echo "=== DAILY SCANNERS ===" | tee -a $RESULTS_FILE
for scanner in manual_scanners/daily_scanners/*/*.py; do
    [ -f "$scanner" ] && test_scanner "$scanner"
done

echo "" | tee -a $RESULTS_FILE
echo "=== WEEKLY SCANNERS ===" | tee -a $RESULTS_FILE
for scanner in manual_scanners/weekly_scanners/*/*.py; do
    [ -f "$scanner" ] && test_scanner "$scanner"
done

# Summary
echo "" | tee -a $RESULTS_FILE
echo "=== SUMMARY ===" | tee -a $RESULTS_FILE
echo "Working scanners:" | tee -a $RESULTS_FILE
grep "✅ SUCCESS" $RESULTS_FILE | wc -l | tee -a $RESULTS_FILE
echo "Failed scanners:" | tee -a $RESULTS_FILE
grep "❌ FAILED" $RESULTS_FILE | wc -l | tee -a $RESULTS_FILE
echo "Timeout scanners:" | tee -a $RESULTS_FILE
grep "⏱️ TIMEOUT" $RESULTS_FILE | wc -l | tee -a $RESULTS_FILE

echo "" | tee -a $RESULTS_FILE
echo "Results saved to: $RESULTS_FILE"
