#!/bin/bash
# File: /opt/bb-screener/check_active_scanners.sh

cd /opt/bb-screener
export $(cat .env | xargs)

echo "=== ACTIVE SCANNER CHECK (Last Hour) ==="
echo "Date: $(date)"
echo ""

psql "$DATABASE_URL" -c "
WITH recent_activity AS (
    SELECT 'FVG 1M' as scanner, COUNT(*) as signals_last_hour, MAX(detected_at) as last_signal
    FROM other_scanners.fvg_signals 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    SELECT 'FVG 5M', COUNT(*), MAX(detected_at)
    FROM other_scanners.fvg_signals 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    SELECT 'Flagpole 5M', COUNT(*), MAX(detected_at)
    FROM other_scanners.flagpole_signals 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    SELECT 'Supply Demand', COUNT(*), MAX(detected_at)
    FROM other_scanners.supply_demand_zones 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    SELECT 'Elliott Wave', COUNT(*), MAX(detected_at)
    FROM elliott_wave.signals 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    SELECT 'ICT Signals', COUNT(*), MAX(detected_at)
    FROM other_scanners.ict_signals 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    SELECT 'Fibonacci', COUNT(*), MAX(detected_at)
    FROM other_scanners.fibonacci_signals 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    SELECT 'Wyckoff', COUNT(*), MAX(detected_at)
    FROM other_scanners.wyckoff_signals 
    WHERE detected_at > NOW() - INTERVAL '1 hour'
)
SELECT * FROM recent_activity 
WHERE signals_last_hour > 0
ORDER BY signals_last_hour DESC;"

echo ""
echo "=== DATABASE TABLE STATISTICS ==="
psql "$DATABASE_URL" -c "
SELECT 
    schemaname || '.' || tablename as table_name,
    n_tup_ins as inserts_total,
    n_tup_upd as updates_total,
    n_tup_del as deletes_total,
    n_live_tup as live_rows
FROM pg_stat_user_tables 
WHERE schemaname IN ('elliott_wave', 'other_scanners')
AND n_tup_ins > 0
ORDER BY n_tup_ins DESC
LIMIT 10;"
