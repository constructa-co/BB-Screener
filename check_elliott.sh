#!/bin/bash
# Load environment variables
source .env

echo "=== Elliott Wave Scanner Status ==="
echo "Patterns in last 24h:"
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT COUNT(*) as count, 
       MAX(created_at) as latest 
FROM elliott_wave_signals 
WHERE created_at > NOW() - INTERVAL '24 hours';"
echo ""
echo "Recent patterns:"
psql "$DATABASE_URL" -c "
SET search_path TO other_scanners;
SELECT symbol, current_wave, pattern_type, pattern_quality, created_at
FROM elliott_wave_signals 
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC 
LIMIT 5;"
