#!/bin/bash
# scripts/health_monitoring/check_wyckoff_health.sh
# Wyckoff Scanner Health Monitoring Script

set -euo pipefail

echo "=== WYCKOFF SCANNER HEALTH CHECK ==="
echo "Time: $(date)"
echo "====================================="

# Check cron job status
echo -e "\n🔍 Cron Status:"
crontab -l | grep wyckoff || echo "  ❌ No Wyckoff cron job found"

# Check recent logs
echo -e "\n📋 Recent Log Activity:"
LOG_DIR="${APP_HOME:-/root/TradingRobotPlug/BB_Screener}/logs/wyckoff"
if [ -d "$LOG_DIR" ]; then
    TODAY_LOG="$LOG_DIR/wyckoff_1h_$(date +%Y%m%d).log"
    if [ -f "$TODAY_LOG" ]; then
        echo "  📁 Today's log: $TODAY_LOG"
        echo "  📊 Recent entries:"
        tail -5 "$TODAY_LOG" | sed 's/^/    /'
    else
        echo "  ⚠️  No log file for today"
    fi
    
    # Check for recent log files
    RECENT_LOGS=$(ls -1t "$LOG_DIR"/wyckoff_1h_*.log 2>/dev/null | head -n 3 || true)
    if [ -n "$RECENT_LOGS" ]; then
        echo "  📚 Recent log files:"
        echo "$RECENT_LOGS" | sed 's/^/    /'
    fi
else
    echo "  ❌ Log directory not found: $LOG_DIR"
fi

# Check database activity (if DATABASE_URL is available)
echo -e "\n🗄️  Database Activity:"
if [ -n "${DATABASE_URL:-}" ]; then
    echo "  🔗 Database connection available"
    # Query for recent Wyckoff signals
    psql "$DATABASE_URL" -c "
        SELECT 
            COUNT(*) as total_signals,
            AVG(setup_score) as avg_score,
            MAX(computed_at) as last_detection,
            COUNT(CASE WHEN computed_at > NOW() - INTERVAL '24 hours' THEN 1 END) as last_24h
        FROM other_scanners.wyckoff_signals;" 2>/dev/null || echo "    ⚠️  Could not query database"
else
    echo "  ⚠️  DATABASE_URL not set - cannot check database activity"
fi

# Check scanner process status
echo -e "\n⚡ Scanner Process Status:"
if pgrep -f "wyckoff_1h_scanner_r1.py" > /dev/null; then
    echo "  ✅ Wyckoff scanner process is running"
    ps aux | grep "wyckoff_1h_scanner_r1.py" | grep -v grep | head -1 | sed 's/^/    /'
else
    echo "  ℹ️  No Wyckoff scanner process currently running (normal for scheduled execution)"
fi

# Check next scheduled run
echo -e "\n⏰ Next Scheduled Run:"
CRON_TIME=$(crontab -l | grep wyckoff | head -1 | awk '{print $1}' || echo "N/A")
if [ "$CRON_TIME" != "N/A" ]; then
    echo "  📅 Cron schedule: $CRON_TIME"
    echo "  🎯 Next execution: :05 and :35 past each hour"
else
    echo "  ❌ No cron schedule found"
fi

# Check for errors in recent logs
echo -e "\n🚨 Error Check:"
if [ -n "${TODAY_LOG:-}" ] && [ -f "$TODAY_LOG" ]; then
    ERROR_COUNT=$(grep -i "error\|exception\|traceback\|failed" "$TODAY_LOG" | wc -l || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  ⚠️  Found $ERROR_COUNT potential errors in today's log"
        echo "  📋 Recent errors:"
        grep -i "error\|exception\|traceback\|failed" "$TODAY_LOG" | tail -3 | sed 's/^/    /'
    else
        echo "  ✅ No errors detected in today's log"
    fi
else
    echo "  ℹ️  No log file available for error checking"
fi

# Health summary
echo -e "\n📊 Health Summary:"
CRON_EXISTS=$(crontab -l | grep -c wyckoff 2>/dev/null || echo "0")
LOG_DIR_EXISTS=$([ -d "$LOG_DIR" ] && echo "1" || echo "0")
TODAY_LOG_EXISTS=$([ -n "${TODAY_LOG:-}" ] && [ -f "$TODAY_LOG" ] && echo "1" || echo "0")

# Clean up variables to ensure they're single integers
CRON_EXISTS=$(echo "$CRON_EXISTS" | tr -d ' ' | head -1)
LOG_DIR_EXISTS=$(echo "$LOG_DIR_EXISTS" | tr -d ' ' | head -1)
TODAY_LOG_EXISTS=$(echo "$TODAY_LOG_EXISTS" | tr -d ' ' | head -1)

if [ "${CRON_EXISTS:-0}" -gt 0 ] && [ "${LOG_DIR_EXISTS:-0}" -eq 1 ]; then
    echo "  🟢 HEALTHY: Cron job configured and log directory exists"
    if [ "${TODAY_LOG_EXISTS:-0}" -eq 1 ]; then
        echo "  🟢 HEALTHY: Today's log file is being written"
    else
        echo "  🟡 WARNING: No log file for today (scanner may not have run yet)"
    fi
else
    echo "  🔴 UNHEALTHY: Cron job or log directory missing"
fi

echo -e "\n✅ Health check complete!"
