#!/bin/bash
echo "=== FIBONACCI SCANNER 24-HOUR MONITORING DASHBOARD ==="
echo "Generated: $(date)"
echo ""

# Load environment variables
source .env

echo "📊 HOURLY SIGNAL GENERATION RATE:"
echo "--------------------------------"
psql $DATABASE_URL -c "
SELECT 
    TO_CHAR(hour, 'HH24:MI') as hour,
    signals_per_hour,
    ROUND(avg_confidence::numeric, 3) as avg_conf,
    unique_symbols,
    high_conf_signals,
    ROUND((high_conf_signals::float / signals_per_hour * 100)::numeric, 1) as high_conf_pct
FROM other_scanners.fibonacci_monitor 
ORDER BY hour DESC 
LIMIT 6;"

echo ""
echo "🎯 CURRENT HOUR PERFORMANCE:"
echo "----------------------------"
psql $DATABASE_URL -c "
SELECT 
    COUNT(*) as total_signals_this_hour,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
    COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END) as high_confidence_signals,
    COUNT(DISTINCT symbol) as unique_symbols,
    MAX(detected_at) as latest_signal
FROM other_scanners.fibonacci_signals 
WHERE detected_at > NOW() - INTERVAL '1 hour';"

echo ""
echo "📈 24-HOUR SUMMARY STATISTICS:"
echo "------------------------------"
psql $DATABASE_URL -c "
SELECT 
    COUNT(*) as total_signals_24h,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence_24h,
    COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END) as high_conf_24h,
    ROUND((COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END)::float / COUNT(*) * 100)::numeric, 1) as high_conf_pct_24h,
    COUNT(DISTINCT symbol) as unique_symbols_24h,
    MIN(detected_at) as first_signal_24h,
    MAX(detected_at) as last_signal_24h
FROM other_scanners.fibonacci_signals 
WHERE detected_at > NOW() - INTERVAL '24 hours';"

echo ""
echo "🔍 SIGNAL TYPE DISTRIBUTION (Last 24h):"
echo "---------------------------------------"
psql $DATABASE_URL -c "
SELECT 
    signal_type,
    COUNT(*) as count,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
    ROUND((COUNT(*)::float / SUM(COUNT(*)) OVER() * 100)::numeric, 1) as percentage
FROM other_scanners.fibonacci_signals 
WHERE detected_at > NOW() - INTERVAL '24 hours'
GROUP BY signal_type
ORDER BY count DESC;"

echo ""
echo "📊 FIBONACCI LEVEL PERFORMANCE (Last 24h):"
echo "------------------------------------------"
psql $DATABASE_URL -c "
SELECT 
    fibonacci_level,
    COUNT(*) as count,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
    COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END) as high_conf_count
FROM other_scanners.fibonacci_signals 
WHERE detected_at > NOW() - INTERVAL '24 hours'
GROUP BY fibonacci_level
ORDER BY count DESC;"

echo ""
echo "🏆 TOP PERFORMING SYMBOLS (Last 24h):"
echo "------------------------------------"
psql $DATABASE_URL -c "
SELECT 
    symbol,
    COUNT(*) as signal_count,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
    COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END) as high_conf_signals
FROM other_scanners.fibonacci_signals 
WHERE detected_at > NOW() - INTERVAL '24 hours'
GROUP BY symbol
HAVING COUNT(*) >= 3
ORDER BY avg_confidence DESC
LIMIT 10;"

echo ""
echo "⚡ SYSTEM HEALTH CHECK:"
echo "----------------------"
echo "Cron Jobs Status:"
crontab -l | grep fibonacci | wc -l | xargs echo "  Active Fibonacci cron jobs:"

echo ""
echo "Recent Log Files:"
ls -lht logs/fibonacci_*.log 2>/dev/null | head -3 | while read line; do
    echo "  $line"
done

echo ""
echo "Process Status:"
ps aux | grep fibonacci | grep -v grep | wc -l | xargs echo "  Active Fibonacci processes:"

echo ""
echo "🔧 PERFORMANCE METRICS:"
echo "----------------------"
echo "Expected vs Actual:"
echo "  Expected signals per hour: 200-300"
echo "  Expected high-confidence rate: 25%+"
echo "  Expected unique symbols per hour: 20-50"

echo ""
echo "📋 MONITORING COMMANDS:"
echo "----------------------"
echo "Real-time monitoring:"
echo "  watch -n 60 './scripts/health_monitoring/fibonacci_24h_monitor.sh'"
echo ""
echo "Check for stuck processes:"
echo "  ps aux | grep fibonacci | grep -v grep"
echo ""
echo "Monitor logs in real-time:"
echo "  tail -f logs/fibonacci_5m_$(date +%Y%m%d).log"
echo ""
echo "Quick signal count:"
echo "  psql \$DATABASE_URL -c \"SELECT COUNT(*) FROM other_scanners.fibonacci_signals WHERE detected_at > NOW() - INTERVAL '1 hour';\""

echo ""
echo "✅ 24-HOUR MONITORING DASHBOARD COMPLETE"
echo "========================================"
