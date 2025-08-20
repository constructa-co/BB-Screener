#!/bin/bash
# Load environment variables
source .env

echo "=== ELLIOTT WAVE SCANNER SUITE STATUS ==="
echo ""

echo "📊 PATTERNS BY TIMEFRAME (Last 24h):"
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT timeframe, COUNT(*) as patterns, 
       MAX(created_at) as latest,
       AVG(pattern_quality) as avg_quality
FROM elliott_wave_signals 
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY timeframe
ORDER BY timeframe;"

echo ""
echo "🎯 RECENT HIGH-QUALITY PATTERNS:"
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT timeframe, symbol, current_wave, pattern_type, 
       pattern_quality, created_at
FROM elliott_wave_signals 
WHERE created_at > NOW() - INTERVAL '24 hours'
  AND pattern_quality >= 70
ORDER BY pattern_quality DESC, created_at DESC 
LIMIT 10;"

echo ""
echo "📈 TOTAL PATTERNS IN DATABASE:"
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT COUNT(*) as total_patterns,
       COUNT(DISTINCT symbol) as unique_symbols,
       COUNT(DISTINCT timeframe) as timeframes_active
FROM elliott_wave_signals;"

echo ""
echo "🔒 ISOLATION VERIFICATION:"
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT COUNT(*) as contamination_check
FROM other_scanners_trades 
WHERE scanner_name LIKE '%elliott%' OR scanner_name LIKE '%wave%';"

echo ""
echo "✅ Elliott Wave Scanner Suite Status Complete"
