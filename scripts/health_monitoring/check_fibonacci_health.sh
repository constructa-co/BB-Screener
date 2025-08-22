#!/bin/bash
echo "=== Fibonacci Scanner Health Check ==="
source .env  # Load environment variables from .env file

echo "📊 Last 24h Fibonacci Activity:"
psql $DATABASE_URL -c "
SELECT
    'Fibonacci Scanners' as scanner_type,
    COUNT(*) as patterns_24h,
    MAX(detected_at) as last_run,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END) as high_confidence_signals
FROM other_scanners.fibonacci_signals
WHERE detected_at > NOW() - INTERVAL '24 hours';"

echo ""
echo "📈 Fibonacci Signal Types (Last 24h):"
psql $DATABASE_URL -c "
SELECT
    signal_type,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence,
    MAX(detected_at) as last_signal
FROM other_scanners.fibonacci_signals
WHERE detected_at > NOW() - INTERVAL '24 hours'
GROUP BY signal_type
ORDER BY count DESC;"

echo ""
echo "🎯 Top Fibonacci Levels (Last 24h):"
psql $DATABASE_URL -c "
SELECT
    fibonacci_level,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence
FROM other_scanners.fibonacci_signals
WHERE detected_at > NOW() - INTERVAL '24 hours'
GROUP BY fibonacci_level
ORDER BY count DESC
LIMIT 5;"

echo ""
echo "📊 Recent Fibonacci Signals (Last 10):"
psql $DATABASE_URL -c "
SELECT
    symbol,
    timeframe,
    signal_type,
    fibonacci_level,
    confidence_score,
    detected_at
FROM other_scanners.fibonacci_signals
WHERE detected_at > NOW() - INTERVAL '24 hours'
ORDER BY detected_at DESC
LIMIT 10;"

echo ""
echo "🔧 Fibonacci Scanner Status:"
echo "  • Database Table: other_scanners.fibonacci_signals"
echo "  • Scanner Version: R1 (Revision 1)"
echo "  • Integration: Complete isolation from other scanners"
echo "  • Circuit Breaker: 5-failure threshold, 300s recovery"
echo "  • Memory Management: 512MB limit, 30min cleanup"

echo ""
echo "📁 Recent Fibonacci Log Files:"
ls -lht logs/fibonacci_*.log 2>/dev/null | head -5

echo ""
echo "✅ Fibonacci Health Check Complete!"
