#!/bin/bash
# File: scripts/health_monitoring/check_trend_following_health.sh
# Trend Following Scanner Health Monitoring Script

set -e

echo "🔍 TREND FOLLOWING SCANNER HEALTH CHECK"
echo "========================================"

# Configuration
SCANNER_HOME="/opt/bb-screener"
SCANNER_NAME="trend_following_1h"
LOG_DIR="$SCANNER_HOME/logs"
DB_URL=""

# Source environment if available
if [ -f "$SCANNER_HOME/.env" ]; then
    source "$SCANNER_HOME/.env"
    DB_URL="$DATABASE_URL"
fi

# Check 1: Cron Job Status
echo "📋 1. CRON JOB STATUS:"
if crontab -l 2>/dev/null | grep -q "$SCANNER_NAME"; then
    echo "   ✅ Cron job found:"
    crontab -l 2>/dev/null | grep "$SCANNER_NAME" | sed 's/^/      /'
else
    echo "   ❌ Cron job not found!"
fi

# Check 2: Recent Log Files
echo ""
echo "📁 2. RECENT LOG FILES:"
if [ -d "$LOG_DIR" ]; then
    recent_logs=$(find "$LOG_DIR" -name "*${SCANNER_NAME}*" -type f -mtime -1 | head -5)
    if [ -n "$recent_logs" ]; then
        echo "   ✅ Recent log files found:"
        echo "$recent_logs" | sed 's/^/      /'
    else
        echo "   ⚠️  No recent log files found"
    fi
else
    echo "   ❌ Log directory not found: $LOG_DIR"
fi

# Check 3: Latest Log Content
echo ""
echo "📊 3. LATEST LOG CONTENT:"
latest_log=$(find "$LOG_DIR" -name "*${SCANNER_NAME}*" -type f -mtime -1 | head -1)
if [ -n "$latest_log" ] && [ -f "$latest_log" ]; then
    echo "   📄 Latest log: $(basename "$latest_log")"
    echo "   📅 Last modified: $(stat -c %y "$latest_log")"
    echo "   📏 File size: $(du -h "$latest_log" | cut -f1)"
    
    # Show last few lines
    echo "   📝 Last 5 lines:"
    tail -5 "$latest_log" | sed 's/^/      /'
else
    echo "   ❌ No log files found"
fi

# Check 4: Database Activity (if DB_URL available)
if [ -n "$DB_URL" ]; then
    echo ""
    echo "🗄️  4. DATABASE ACTIVITY:"
    
    # Check recent signals
    recent_signals=$(psql "$DB_URL" -t -c "
        SELECT COUNT(*) as total_signals,
               COUNT(CASE WHEN detected_at > NOW() - INTERVAL '24 hours' THEN 1 END) as last_24h,
               COUNT(CASE WHEN detected_at > NOW() - INTERVAL '1 hour' THEN 1 END) as last_1h,
               MAX(detected_at) as last_signal
        FROM other_scanners.trend_following_signals 
        WHERE timeframe = '1h';" 2>/dev/null | tr -d ' ')
    
    if [ -n "$recent_signals" ]; then
        echo "   ✅ Database connection successful"
        echo "   📊 Signals in database: $recent_signals"
    else
        echo "   ❌ Database connection failed"
    fi
else
    echo ""
    echo "🗄️  4. DATABASE ACTIVITY:"
    echo "   ⚠️  DATABASE_URL not available"
fi

# Check 5: Scanner Process Status
echo ""
echo "🔄 5. SCANNER PROCESS STATUS:"
if pgrep -f "trend_following_scanner_1h_r1.py" > /dev/null; then
    echo "   ✅ Scanner process running"
    pgrep -f "trend_following_scanner_1h_r1.py" | sed 's/^/      PID: /'
else
    echo "   ℹ️  No scanner process currently running (normal between scheduled runs)"
fi

# Check 6: Next Scheduled Run
echo ""
echo "⏰ 6. NEXT SCHEDULED RUN:"
current_minute=$(date +%M)
current_hour=$(date +%H)

# Calculate next run times
next_8=$(date -d "$(date +%Y-%m-%d) $current_hour:08:00" +"%H:%M")
next_38=$(date -d "$(date +%Y-%m-%d) $current_hour:38:00" +"%H:%M")

if [ "$current_minute" -lt 8 ]; then
    next_run="$next_8"
elif [ "$current_minute" -lt 38 ]; then
    next_run="$next_38"
else
    next_run="$next_8"  # Next hour
fi

echo "   🎯 Next execution: $next_run"
echo "   📅 Schedule: :08 and :38 past each hour"

# Summary
echo ""
echo "📋 HEALTH CHECK SUMMARY:"
echo "========================"

# Count issues
issues=0
if ! crontab -l 2>/dev/null | grep -q "$SCANNER_NAME"; then
    ((issues++))
fi

if [ ! -d "$LOG_DIR" ] || [ -z "$(find "$LOG_DIR" -name "*${SCANNER_NAME}*" -type f -mtime -1)" ]; then
    ((issues++))
fi

if [ -n "$DB_URL" ] && [ -z "$recent_signals" ]; then
    ((issues++))
fi

if [ $issues -eq 0 ]; then
    echo "   🟢 HEALTHY: All systems operational"
elif [ $issues -eq 1 ]; then
    echo "   🟡 WARNING: Minor issues detected"
else
    echo "   🔴 CRITICAL: Multiple issues detected"
fi

echo "   📊 Issues found: $issues"
echo ""
echo "💡 Monitor scanner with: tail -f $LOG_DIR/${SCANNER_NAME}_\$(date +%Y%m%d).log"
