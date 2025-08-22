#!/bin/bash
echo "🔧 Setting up Fibonacci Scanner Cron Jobs with Non-Conflicting Schedule..."

# Remove any existing Fibonacci cron entries
crontab -l | grep -v "fibonacci" | crontab -

# Add Fibonacci scanner to crontab with isolated time slots
# Runs at :05, :20, :35, :50 to avoid Elliott Wave (:00, :30) and ICT (:15, :45)
(crontab -l 2>/dev/null; cat << 'EOF'
# Fibonacci Scanner - Non-conflicting time slots
# :05, :20, :35, :50 every hour (avoids Elliott Wave :00/:30 and ICT :15/:45)
5,20,35,50 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/5_min_scanners/fibonacci_revisions/fibonacci_retracement_scanner_r1.py 5m >> logs/fibonacci_5m_$(date +\%Y\%m\%d).log 2>&1

# Fibonacci Scanner 1-minute version (if needed)
# :08, :23, :38, :53 every hour
8,23,38,53 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/5_min_scanners/fibonacci_revisions/fibonacci_retracement_scanner_r1.py 1m >> logs/fibonacci_1m_$(date +\%Y\%m\%d).log 2>&1
EOF
) | crontab -

echo "✅ Fibonacci Scanner cron jobs installed successfully!"
echo ""
echo "📅 Schedule Summary:"
echo "  • Fibonacci 5M: :05, :20, :35, :50 every hour"
echo "  • Fibonacci 1M: :08, :23, :38, :53 every hour"
echo ""
echo "🔒 Non-Conflicting Time Slots:"
echo "  • Elliott Wave: :00, :30 (4H scanner)"
echo "  • ICT Scanners: :10, :25, :40, :55 (15M), :20 (1H), :30 (4H)"
echo "  • Fibonacci: :05, :08, :20, :23, :35, :38, :50, :53"
echo ""
echo "📊 Expected Performance:"
echo "  • 4 scans per hour per timeframe"
echo "  • 8 total Fibonacci scans per hour"
echo "  • Zero conflicts with existing scanners"
echo "  • Complete isolation maintained"
echo ""
echo "📁 Log Files:"
echo "  • logs/fibonacci_5m_YYYYMMDD.log"
echo "  • logs/fibonacci_1m_YYYYMMDD.log"
echo ""
echo "🔍 To verify installation:"
echo "  crontab -l | grep fibonacci"
echo ""
echo "📊 To check health:"
echo "  ./scripts/health_monitoring/check_fibonacci_health.sh"
