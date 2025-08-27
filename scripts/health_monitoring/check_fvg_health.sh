#!/bin/bash
echo "=== FVG SCANNER HEALTH CHECK ==="
echo "Time: $(date)"

# Check cron
echo -e "\n🎯 Cron Status:"
crontab -l | grep fair_value_gap || echo "  ⚠️ No FVG cron jobs found"

# Check logs
LOG_FILE="/opt/bb-screener/logs/fvg/fvg_1m_$(date +%Y%m%d).log"
if [ -f "$LOG_FILE" ]; then
    echo -e "\n📊 Recent Log:"
    tail -5 "$LOG_FILE"
else
    echo -e "\n⚠️ No log file found for today"
fi

# Check database
echo -e "\n🗄️ Database Activity (24h):"
psql $DATABASE_URL -c "
    SELECT timeframe, COUNT(*) as signals, AVG(setup_score) as avg_score
    FROM other_scanners.fvg_signals
    WHERE detected_at > NOW() - INTERVAL '24 hours'
    GROUP BY timeframe;" 2>/dev/null || echo "  ⚠️ Database connection failed"

echo -e "\n✅ FVG health check completed"
