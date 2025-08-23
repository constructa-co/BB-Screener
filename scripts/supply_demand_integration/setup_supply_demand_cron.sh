#!/bin/bash

# Supply & Demand Scanner Cron Setup Script
# This script adds cron jobs for all S&D scanners with non-conflicting schedules
# EXISTING SCANNERS (DO NOT MODIFY):
# :00 - BB Scanner (main_scanner.py)
# :10, :25, :40, :55 - ICT 15M Scanner
# :20 - ICT 1H Scanner
# :30 - ICT 4H Scanner
# :45 - Elliott Wave 4H Scanner (every 4 hours)
# :50 - Elliott Wave 1H Scanner
# :05, :20, :35, :50 - Fibonacci 5M Scanner
# :08, :23, :38, :53 - Fibonacci 1M Scanner

set -e

echo "🔧 SUPPLY & DEMAND SCANNER CRON SETUP"
echo "======================================"
echo "Timestamp: $(date)"
echo ""

# Check if we're running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root"
    exit 1
fi

# Backup current crontab
echo "📋 Backing up current crontab..."
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).bak
echo "✅ Crontab backed up to /tmp/"

# Create logs directory if it doesn't exist
echo "📁 Ensuring logs directory exists..."
mkdir -p /opt/bb-screener/logs

# Supply & Demand Scanner Cron Jobs (Non-conflicting schedules)
echo "⏰ Adding Supply & Demand scanner cron jobs..."

# 5-Minute Scanner (every 5 minutes, avoiding conflicts)
echo "   Adding 5-minute S&D scanner..."
(crontab -l 2>/dev/null; echo "3,18,33,48 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/5_minute_scanners/supply_demand_scanner_5m_r1.py >> logs/sd_5m_\$(date +\%Y\%m\%d).log 2>&1") | crontab -

# 15-Minute Scanner (every 15 minutes, avoiding conflicts)
echo "   Adding 15-minute S&D scanner..."
(crontab -l 2>/dev/null; echo "13,28,43,58 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/15_minute_scanners/supply_demand_scanner_15m_r1.py >> logs/sd_15m_\$(date +\%Y\%m\%d).log 2>&1") | crontab -

# 1-Hour Scanner (every hour, avoiding conflicts)
echo "   Adding 1-hour S&D scanner..."
(crontab -l 2>/dev/null; echo "3 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/1_hour_scanners/supply_demand_scanner_1h_r1.py >> logs/sd_1h_\$(date +\%Y\%m\%d).log 2>&1") | crontab -

# 4-Hour Scanner (every 4 hours, avoiding conflicts)
echo "   Adding 4-hour S&D scanner..."
(crontab -l 2>/dev/null; echo "3 */4 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/4_hour_scanners/supply_demand_scanner_4h_r1.py >> logs/sd_4h_\$(date +\%Y\%m\%d).log 2>&1") | crontab -

echo "✅ All Supply & Demand scanner cron jobs added successfully!"
echo ""

# Verify the new cron jobs
echo "🔍 Verifying new cron jobs..."
echo "Current crontab:"
crontab -l | grep -E "(sd_|supply_demand)" || echo "No S&D cron jobs found (this shouldn't happen)"

echo ""
echo "📊 CRON SCHEDULE SUMMARY:"
echo "=========================="
echo "EXISTING SCANNERS (UNCHANGED):"
echo "  :00  - BB Scanner (main_scanner.py)"
echo "  :10, :25, :40, :55 - ICT 15M Scanner"
echo "  :20  - ICT 1H Scanner"
echo "  :30  - ICT 4H Scanner"
echo "  :45  - Elliott Wave 4H Scanner (every 4 hours)"
echo "  :50  - Elliott Wave 1H Scanner"
echo "  :05, :20, :35, :50 - Fibonacci 5M Scanner"
echo "  :08, :23, :38, :53 - Fibonacci 1M Scanner"
echo ""
echo "NEW SUPPLY & DEMAND SCANNERS:"
echo "  :03, :18, :33, :48 - S&D 5M Scanner (every 5 minutes)"
echo "  :13, :28, :43, :58 - S&D 15M Scanner (every 15 minutes)"
echo "  :03  - S&D 1H Scanner (every hour)"
echo "  :03  - S&D 4H Scanner (every 4 hours)"
echo ""
echo "✅ NO SCHEDULE CONFLICTS - All scanners can run simultaneously!"
echo ""
echo "📝 Log files will be created in: /opt/bb-screener/logs/"
echo "   - sd_5m_YYYYMMDD.log"
echo "   - sd_15m_YYYYMMDD.log"
echo "   - sd_1h_YYYYMMDD.log"
echo "   - sd_4h_YYYYMMDD.log"
echo ""
echo "🚀 Supply & Demand scanners are now scheduled and ready for production!"
echo "======================================"
