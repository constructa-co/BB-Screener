#!/bin/bash
echo "🔧 Fixing ICT Scanner Execution Issues..."

# Set up environment
cd /opt/bb-screener
source .env  # Load environment variables from .env file

echo "✅ Environment variables set"
echo "📊 Testing ICT 15M Scanner..."

# Test ICT 15M scanner
python3 manual_scanners/15_min_scanners/ict_scanner_15m_r4.py --once --symbols 10 --quality 50

echo "📊 Testing ICT 1H Scanner..."
# Test ICT 1H scanner  
python3 manual_scanners/1_hour_scanners/ict_scanner_1h_r4.py --once --symbols 10 --quality 50

echo "📊 Testing ICT 4H Scanner..."
# Test ICT 4H scanner
python3 manual_scanners/4_hour_scanners/ict_scanner_4h_r9.py --once --symbols 10 --quality 50

echo "✅ ICT Scanner tests completed"
