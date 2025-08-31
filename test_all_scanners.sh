#!/bin/bash
# File: /opt/bb-screener/test_all_scanners.sh

cd /opt/bb-screener
export $(cat .env | xargs)
export PYTHONPATH=.

echo "=== COMPREHENSIVE SCANNER TEST ==="
echo "Testing all scanners systematically..."
echo ""

# Test 5-minute scanners
echo "=== 5-MINUTE SCANNERS ==="
for scanner in manual_scanners/5_min_scanners/*/*.py; do
    if [ -f "$scanner" ]; then
        echo "Testing: $scanner"
        timeout 15 python3 "$scanner" 2>&1 | tail -3
        echo "---"
    fi
done

# Test 15-minute scanners
echo "=== 15-MINUTE SCANNERS ==="
for scanner in manual_scanners/15_min_scanners/*/*.py; do
    if [ -f "$scanner" ]; then
        echo "Testing: $scanner"
        timeout 15 python3 "$scanner" 2>&1 | tail -3
        echo "---"
    fi
done

# Test 1-hour scanners
echo "=== 1-HOUR SCANNERS ==="
for scanner in manual_scanners/1_hour_scanners/*/*.py; do
    if [ -f "$scanner" ]; then
        echo "Testing: $scanner"
        timeout 15 python3 "$scanner" 2>&1 | tail -3
        echo "---"
    fi
done

# Test 4-hour scanners
echo "=== 4-HOUR SCANNERS ==="
for scanner in manual_scanners/4_hour_scanners/*/*.py; do
    if [ -f "$scanner" ]; then
        echo "Testing: $scanner"
        timeout 15 python3 "$scanner" 2>&1 | tail -3
        echo "---"
    fi
done

# Test daily scanners
echo "=== DAILY SCANNERS ==="
for scanner in manual_scanners/daily_scanners/*/*.py; do
    if [ -f "$scanner" ]; then
        echo "Testing: $scanner"
        timeout 15 python3 "$scanner" 2>&1 | tail -3
        echo "---"
    fi
done

# Test weekly scanners
echo "=== WEEKLY SCANNERS ==="
for scanner in manual_scanners/weekly_scanners/*/*.py; do
    if [ -f "$scanner" ]; then
        echo "Testing: $scanner"
        timeout 15 python3 "$scanner" 2>&1 | tail -3
        echo "---"
    fi
done

echo "=== SCANNER TEST COMPLETE ==="
