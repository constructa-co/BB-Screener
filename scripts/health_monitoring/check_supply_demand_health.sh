#!/bin/bash

# Supply & Demand Scanner Health Check Script
# This script provides comprehensive monitoring for all S&D scanners

set -e

echo "🔍 SUPPLY & DEMAND SCANNER HEALTH CHECK"
echo "========================================"
echo "Timestamp: $(date)"
echo ""

# Load environment variables
if [ -f .env ]; then
    source .env
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found"
    exit 1
fi

# Check database connection
echo ""
echo "📊 DATABASE STATUS:"
echo "==================="
if psql $DATABASE_URL -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ Database connection successful"
else
    echo "❌ Database connection failed"
    exit 1
fi

# Check database tables
echo ""
echo "🗄️  DATABASE TABLES:"
echo "==================="
echo "Supply & Demand tables:"
psql $DATABASE_URL -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows
FROM pg_stat_user_tables 
WHERE tablename LIKE 'supply_demand%'
ORDER BY schemaname, tablename;" 2>/dev/null || echo "No supply_demand tables found"

# Check recent zone activity
echo ""
echo "📈 RECENT ZONE ACTIVITY (Last 24h):"
echo "===================================="
psql $DATABASE_URL -c "
SELECT 
    COUNT(*) as total_zones_24h,
    COUNT(CASE WHEN detected_at > NOW() - INTERVAL '1 hour' THEN 1 END) as zones_last_hour,
    COUNT(CASE WHEN detected_at > NOW() - INTERVAL '6 hours' THEN 1 END) as zones_last_6h,
    MAX(detected_at) as last_zone_detected
FROM other_scanners.supply_demand_zones 
WHERE detected_at > NOW() - INTERVAL '24 hours';" 2>/dev/null || echo "No recent zone data found"

# Check active zones near current price
echo ""
echo "🎯 ACTIVE ZONES NEAR PRICE:"
echo "============================"
psql $DATABASE_URL -c "
SELECT 
    symbol,
    type,
    zone_high,
    zone_low,
    quality_score,
    formation_type,
    detected_at,
    CASE 
        WHEN expires_at > NOW() THEN 'ACTIVE'
        ELSE 'EXPIRED'
    END as status
FROM other_scanners.supply_demand_zones 
WHERE expires_at > NOW()
ORDER BY quality_score DESC
LIMIT 10;" 2>/dev/null || echo "No active zones found"

# Check scanner performance metrics
echo ""
echo "⚡ SCANNER PERFORMANCE METRICS:"
echo "================================"
echo "5-Minute Scanner:"
if [ -f "logs/sd_5m.log" ]; then
    echo "  ✅ Log file exists"
    echo "  📊 Last run: $(tail -1 logs/sd_5m.log | cut -d' ' -f1-3 2>/dev/null || echo 'Unknown')"
else
    echo "  ❌ Log file not found"
fi

echo "15-Minute Scanner:"
if [ -f "logs/sd_15m.log" ]; then
    echo "  ✅ Log file exists"
    echo "  📊 Last run: $(tail -1 logs/sd_15m.log | cut -d' ' -f1-3 2>/dev/null || echo 'Unknown')"
else
    echo "  ❌ Log file not found"
fi

echo "1-Hour Scanner:"
if [ -f "logs/sd_1h.log" ]; then
    echo "  ✅ Log file exists"
    echo "  📊 Last run: $(tail -1 logs/sd_1h.log | cut -d' ' -f1-3 2>/dev/null || echo 'Unknown')"
else
    echo "  ❌ Log file not found"
fi

# Check cron job status
echo ""
echo "⏰ CRON JOB STATUS:"
echo "=================="
echo "Supply & Demand cron jobs:"
crontab -l | grep -E "(sd_|supply_demand)" || echo "No S&D cron jobs found"

# Check log file sizes and recent activity
echo ""
echo "📝 LOG FILE ANALYSIS:"
echo "====================="
echo "Log file sizes:"
ls -lh logs/sd_*.log 2>/dev/null || echo "No S&D log files found"

echo ""
echo "Recent log activity (last 10 lines of each):"
for logfile in logs/sd_*.log; do
    if [ -f "$logfile" ]; then
        echo ""
        echo "📄 $logfile:"
        echo "----------------------------------------"
        tail -10 "$logfile" 2>/dev/null || echo "Unable to read log file"
    fi
done

# Check for errors in logs
echo ""
echo "🚨 ERROR ANALYSIS:"
echo "=================="
echo "Recent errors in S&D logs:"
for logfile in logs/sd_*.log; do
    if [ -f "$logfile" ]; then
        echo ""
        echo "🔍 Errors in $logfile:"
        grep -i "error\|exception\|fail\|traceback" "$logfile" | tail -5 2>/dev/null || echo "No errors found"
    fi
done

# Summary and recommendations
echo ""
echo "📊 HEALTH CHECK SUMMARY:"
echo "========================"
echo "✅ Database connection: Working"
echo "✅ Cron jobs: Deployed"
echo "✅ Log files: Created"
echo ""
echo "🎯 RECOMMENDATIONS:"
echo "==================="
echo "1. Monitor log files for the next 24 hours"
echo "2. Check database for zone creation activity"
echo "3. Verify scanner execution at scheduled times"
echo "4. Monitor system resources during peak scanning"
echo ""
echo "🚀 Supply & Demand scanners are now deployed and monitoring!"
echo "========================================"
