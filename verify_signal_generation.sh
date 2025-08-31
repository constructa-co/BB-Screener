#!/bin/bash
# File: /opt/bb-screener/verify_signal_generation.sh

cd /opt/bb-screener
export $(cat .env | xargs)

echo "=== SIGNAL GENERATION VERIFICATION ==="
echo "Checking signals generated in last 10 minutes..."
echo "Date: $(date)"
echo ""

psql "$DATABASE_URL" -c "
WITH recent_signals AS (
    SELECT 
        'FVG' as scanner_type,
        '1M' as timeframe,
        COUNT(*) as signals_10min
    FROM other_scanners.fvg_signals 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
    
    UNION ALL
    SELECT 'Flagpole', '5M', COUNT(*)
    FROM other_scanners.flagpole_signals 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
    
    UNION ALL
    SELECT 'Supply/Demand', 'Multiple', COUNT(*)
    FROM other_scanners.supply_demand_zones 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
    
    UNION ALL
    SELECT 'Elliott Wave', 'Multiple', COUNT(*)
    FROM elliott_wave.signals 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
    
    UNION ALL
    SELECT 'ICT', 'Multiple', COUNT(*)
    FROM other_scanners.ict_signals 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
    
    UNION ALL
    SELECT 'Trend Following', '1H', COUNT(*)
    FROM other_scanners.trend_following_signals 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
    
    UNION ALL
    SELECT 'Fibonacci', '5M', COUNT(*)
    FROM other_scanners.fibonacci_signals 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
    
    UNION ALL
    SELECT 'Wyckoff', 'Multiple', COUNT(*)
    FROM other_scanners.wyckoff_signals 
    WHERE detected_at > NOW() - INTERVAL '10 minutes'
)
SELECT * FROM recent_signals 
ORDER BY signals_10min DESC;"

echo ""
echo "=== TOTAL SIGNALS LAST 10 MINUTES ==="
psql "$DATABASE_URL" -c "
SELECT 'Total signals last 10 minutes:' as metric, 
       SUM(count) as value
FROM (
    SELECT COUNT(*) as count FROM other_scanners.fvg_signals WHERE detected_at > NOW() - INTERVAL '10 minutes'
    UNION ALL
    SELECT COUNT(*) FROM other_scanners.flagpole_signals WHERE detected_at > NOW() - INTERVAL '10 minutes'
    UNION ALL
    SELECT COUNT(*) FROM other_scanners.supply_demand_zones WHERE detected_at > NOW() - INTERVAL '10 minutes'
    UNION ALL
    SELECT COUNT(*) FROM elliott_wave.signals WHERE detected_at > NOW() - INTERVAL '10 minutes'
    UNION ALL
    SELECT COUNT(*) FROM other_scanners.ict_signals WHERE detected_at > NOW() - INTERVAL '10 minutes'
    UNION ALL
    SELECT COUNT(*) FROM other_scanners.trend_following_signals WHERE detected_at > NOW() - INTERVAL '10 minutes'
    UNION ALL
    SELECT COUNT(*) FROM other_scanners.fibonacci_signals WHERE detected_at > NOW() - INTERVAL '10 minutes'
    UNION ALL
    SELECT COUNT(*) FROM other_scanners.wyckoff_signals WHERE detected_at > NOW() - INTERVAL '10 minutes'
) totals;"

echo ""
echo "=== SCANNER LOG STATUS ==="
echo "Checking recent log files for scanner activity:"
ls -la logs/*.log | head -10
echo ""
echo "Recent log entries (last 5 lines of each):"
for log in logs/*.log; do
    if [ -f "$log" ]; then
        echo "--- $log ---"
        tail -5 "$log" 2>/dev/null || echo "No recent entries"
        echo ""
    fi
done
