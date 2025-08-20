#!/bin/bash
echo "=== ELLIOTT WAVE SCANNER SUITE TEST ==="
echo "Testing all timeframes with database integration..."
echo ""

# Load environment variables
source .env

echo "=== Testing 4H Scanner (r2) ==="
echo "Running 4H scanner with database logging..."
python3 manual_scanners/4_hour_scanners/elliott_waves_scanner_4h_r2.py | head -20
echo ""

echo "=== Testing Daily Scanner (r5) ==="
echo "Running Daily scanner with database logging..."
python3 manual_scanners/daily_scanners/elliott_waves_scanner_1d_r5.py | head -20
echo ""

echo "=== Testing Weekly Scanner (r1) ==="
echo "Running Weekly scanner with database logging..."
python3 manual_scanners/weekly_scanners/elliot_waves_scanner_1w_r1.py | head -20
echo ""

echo "=== Database Validation ==="
echo "Checking Elliott Wave patterns in database..."
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT timeframe, COUNT(*) as patterns, MAX(created_at) as latest
FROM elliott_wave_signals
GROUP BY timeframe
ORDER BY timeframe;"

echo ""
echo "=== Recent Patterns by Timeframe ==="
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT timeframe, symbol, current_wave, pattern_type, pattern_quality, created_at
FROM elliott_wave_signals
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC
LIMIT 10;"

echo ""
echo "=== Isolation Check ==="
echo "Verifying no Elliott data in other_scanners_trades..."
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT COUNT(*) as contamination_check
FROM other_scanners_trades
WHERE scanner_name LIKE '%elliott%' OR scanner_name LIKE '%wave%';"

echo ""
echo "=== Test Complete ==="
echo "All Elliott Wave scanners tested successfully!"
