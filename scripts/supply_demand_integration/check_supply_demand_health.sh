#!/bin/bash

# Supply & Demand Scanner Health Check Script
# This script provides comprehensive health monitoring for all S&D scanners

set -e

echo "🔍 SUPPLY & DEMAND SCANNER HEALTH CHECK"
echo "========================================"
echo "Timestamp: $(date)"
echo ""

# Check database connectivity and recent activity
echo "🔍 Database Status:"
source .env

# Check if tables exist
echo "📊 Checking database tables..."
psql $DATABASE_URL -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'other_scanners' 
AND tablename LIKE 'supply_demand%'
ORDER BY tablename;" 2>/dev/null || echo "❌ Database connection failed"

echo ""

# Check recent zone activity
echo "📈 Recent Zone Activity (Last 24h):"
psql $DATABASE_URL -c "
SELECT 
    COUNT(*) as total_zones_24h,
    COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as zones_last_hour,
    COUNT(CASE WHEN created_at > NOW() - INTERVAL '6 hours' THEN 1 END) as zones_last_6h
FROM other_scanners.supply_demand_zones 
WHERE created_at > NOW() - INTERVAL '24 hours';" 2>/dev/null || echo "❌ Query failed"

echo ""

# Check active zones near current price
echo "🎯 Active Zones Near Price:"
psql $DATABASE_URL -c "
SELECT 
    symbol,
    zone_type,
    zone_level,
    quality_score,
    created_at,
    expires_at
FROM other_scanners.supply_demand_zones 
WHERE expires_at > NOW()
ORDER BY quality_score DESC, created_at DESC
LIMIT 10;" 2>/dev/null || echo "❌ Query failed"

echo ""

# Check scanner performance metrics
echo "⚡ Scanner Performance Metrics:"
psql $DATABASE_URL -c "
SELECT 
    scanner_name,
    COUNT(*) as total_zones,
    AVG(quality_score) as avg_quality,
    MAX(created_at) as last_scan
FROM other_scanners.supply_demand_zones 
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY scanner_name
ORDER BY total_zones DESC;" 2>/dev/null || echo "❌ Query failed"

echo ""

# Check cron job status
echo "⏰ Cron Job Status:"
crontab -l | grep -E "supply_demand|sd_" || echo "❌ No S&D cron jobs found"

echo ""

# Check log files
echo "📝 Recent Log Files:"
ls -lht logs/sd_*.log 2>/dev/null | head -5 || echo "❌ No S&D log files found"

echo ""

# Check pending post-mortem analysis
echo "🔬 Pending Post-Mortem Analysis:"
psql $DATABASE_URL -c "
SELECT 
    COUNT(*) as zones_needing_analysis
FROM other_scanners.supply_demand_zones 
WHERE expires_at < NOW() 
AND post_mortem_analysis IS NULL;" 2>/dev/null || echo "❌ Query failed"

echo ""
echo "✅ Health check complete!"
echo "========================================"
