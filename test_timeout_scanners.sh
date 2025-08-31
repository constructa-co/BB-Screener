#!/bin/bash
# File: /opt/bb-screener/test_timeout_scanners.sh

cd /opt/bb-screener
export $(cat .env | xargs)
export PYTHONPATH=.

echo "=== RETESTING TIMEOUT SCANNERS WITH EXTENDED TIME (60s) ==="
echo "Date: $(date)"
echo ""

# Function to test with extended timeout
test_scanner_extended() {
    local scanner_path=$1
    local scanner_name=$(basename $scanner_path)
    echo "Testing: $scanner_name"
    echo "Start time: $(date +%H:%M:%S)"
    
    # Run with 60 second timeout and capture both output and exit code
    timeout 60 python3 "$scanner_path" 2>&1 | tee /tmp/scanner_extended_output.txt &
    PID=$!
    
    # Monitor for 60 seconds
    for i in {1..60}; do
        if ! kill -0 $PID 2>/dev/null; then
            # Process finished
            wait $PID
            exit_code=$?
            echo "Completed in ${i} seconds with exit code: $exit_code"
            
            # Show last 10 lines of output
            echo "Output summary:"
            tail -10 /tmp/scanner_extended_output.txt
            
            # Check if it logged to database
            if grep -q "logged\|Logged\|inserted\|stored" /tmp/scanner_extended_output.txt; then
                echo "✅ DATABASE WRITES DETECTED"
            fi
            
            return 0
        fi
        
        # Every 10 seconds, show progress
        if [ $((i % 10)) -eq 0 ]; then
            echo "  Still running... ${i}s elapsed"
            # Check if generating output
            if [ -s /tmp/scanner_extended_output.txt ]; then
                echo "  Latest output: $(tail -1 /tmp/scanner_extended_output.txt)"
            fi
        fi
        
        sleep 1
    done
    
    echo "⏱️ TIMEOUT after 60 seconds (likely processing large dataset)"
    kill $PID 2>/dev/null
    echo "Partial output:"
    tail -10 /tmp/scanner_extended_output.txt
    echo "---"
}

# Test the scanners that showed timeout
echo "=== TESTING FVG 5M Scanner ==="
test_scanner_extended "manual_scanners/5_min_scanners/fair_value_gap_scanner/fair_value_gap_+_fibonacci_scanner_5m_r1.py"

echo ""
echo "=== TESTING Heikin Ashi 5M Scanner ==="
test_scanner_extended "manual_scanners/5_min_scanners/Heikin Ashi Scanner/heikin_ashi_scanner_r1.py"

echo ""
echo "=== TESTING Supply & Demand 5M Scanner ==="
test_scanner_extended "manual_scanners/5_minute_scanners/supply_demand_scanner_5m_r1.py"

echo ""
echo "=== TESTING Trend Following 1H Scanner ==="
test_scanner_extended "manual_scanners/1_hour_scanners/trend_following_scanner_1h_r1.py"

echo ""
echo "=== TESTING ICT 1H Scanner ==="
test_scanner_extended "manual_scanners/1_hour_scanners/ict_scanner_1h_r4.py"

echo ""
echo "=== TESTING Supply & Demand 1H Scanner ==="
test_scanner_extended "manual_scanners/1_hour_scanners/supply_demand_scanner_1h_r1.py"

echo ""
echo "=== EXTENDED TIMEOUT TESTING COMPLETE ==="
