#!/bin/bash
# File: /opt/bb-screener/verify_signals_after_cron.sh

cd /opt/bb-screener
export $(cat .env | xargs)

echo "=== SIGNAL GENERATION VERIFICATION ==="
echo "Time: $(date)"
echo ""

# Check if cron jobs have run
echo "Last cron executions:"
grep CRON /var/log/syslog | tail -10
echo ""

# Check signals generated in last 15 minutes
echo "Signals generated in last 15 minutes:"
psql "$DATABASE_URL" -c "
WITH signal_counts AS (
    SELECT 
        'FVG 1M' as scanner,
        COUNT(*) as count_15min,
        MAX(detected_at) as last_signal
    FROM other_scanners.fvg_signals 
    WHERE detected_at > NOW() - INTERVAL '15 minutes'
    
    UNION ALL
    SELECT 'Flagpole 5M', COUNT(*), MAX(detected_at)
    FROM other_scanners.flagpole_signals 
    WHERE detected_at > NOW() - INTERVAL '15 minutes'
    
    UNION ALL
    SELECT 'Supply/Demand', COUNT(*), MAX(detected_at)
    FROM other_scanners.supply_demand_zones 
    WHERE detected_at > NOW() - INTERVAL '15 minutes'
    
    UNION ALL
    SELECT 'Heikin Ashi', COUNT(*), MAX(detected_at)
    FROM other_scanners.heikin_ashi_signals 
    WHERE detected_at > NOW() - INTERVAL '15 minutes'
    
    UNION ALL
    SELECT 'ICT', COUNT(*), MAX(detected_at)
    FROM other_scanners.ict_signals 
    WHERE detected_at > NOW() - INTERVAL '15 minutes'
    
    UNION ALL
    SELECT 'Elliott Wave', COUNT(*), MAX(detected_at)
    FROM elliott_wave.signals 
    WHERE detected_at > NOW() - INTERVAL '15 minutes'
)
SELECT * FROM signal_counts 
WHERE count_15min > 0 OR scanner IN ('FVG 1M', 'Flagpole 5M')
ORDER BY count_15min DESC;"

# Check log file activity
echo ""
echo "Recent log file updates:"
ls -lt /opt/bb-screener/logs/*.log | head -10

echo ""
echo "=== VERIFICATION COMPLETE ==="
echo "If signals are being generated but dashboard shows nothing,"
echo "then we need to fix the dashboard display layer specifically."
