#!/bin/bash
echo "=== Scanner Health Check ==="
echo ""

# Set up environment
source .env  # Load environment variables from .env file

echo "📊 Last 24h Scanner Activity:"
psql $DATABASE_URL -c "
SELECT 
    CASE 
        WHEN scanner_name LIKE 'ict_%' THEN 'ICT Scanners'
        WHEN scanner_name LIKE 'elliott%' THEN 'Elliott Wave'
        ELSE scanner_name 
    END as scanner_type,
    COUNT(*) as patterns_24h,
    MAX(created_at) as last_run
FROM (
    SELECT scanner_name, created_at 
    FROM other_scanners.other_scanners_trades 
    WHERE created_at > NOW() - INTERVAL '24 hours'
    UNION ALL
    SELECT scanner_name, created_at 
    FROM other_scanners.elliott_wave_signals 
    WHERE created_at > NOW() - INTERVAL '24 hours'
) combined
GROUP BY scanner_type
ORDER BY last_run DESC;"

echo ""
echo "🔧 Cron Jobs Status:"
crontab -l | grep -E "ict_|elliott" | head -10

echo ""
echo "📁 Recent Log Files:"
ls -lht logs/*.log 2>/dev/null | head -10

echo ""
echo "🎯 Scanner Status Summary:"
echo "  • BB Scanner: $(crontab -l | grep -c 'main_scanner.py') cron jobs"
echo "  • ICT Scanners: $(crontab -l | grep -c 'ict_') cron jobs"
echo "  • Elliott Wave: $(crontab -l | grep -c 'elliott') cron jobs"

echo ""
echo "✅ Health Check Complete!"
