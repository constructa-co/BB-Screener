#!/bin/bash
echo "🔧 Updating Cron Jobs with New File Paths..."

# Remove old cron jobs
crontab -l | grep -v "ict_" | grep -v "elliott" | crontab -

# Add updated cron jobs with new script paths
(crontab -l 2>/dev/null; cat << 'EOF'
# BB Scanner - Every hour
0 * * * * cd /opt/bb-screener && source venv/bin/activate && python main_scanner.py >> /opt/bb-screener/scanner.log 2>&1 && python outputs/utility_scripts/fix_all_db_issues.py >> /opt/bb-screener/db_sync.log 2>&1

# ICT Scanners - Fixed to use system Python
# ICT 15M Scanner - Every 15 minutes at :10, :25, :40, :55
10,25,40,55 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/15_min_scanners/ict_scanner_15m_r4.py --once --symbols 500 --quality 75 >> logs/ict_15m_$(date +\%Y\%m\%d).log 2>&1

# ICT 1H Scanner - Every hour at :20
20 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/1_hour_scanners/ict_scanner_1h_r4.py --once --symbols 500 --quality 75 >> logs/ict_1h_$(date +\%Y\%m\%d).log 2>&1

# ICT 4H Scanner - Every hour at :30
30 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/4_hour_scanners/ict_scanner_4h_r9.py --once --symbols 500 --quality 75 >> logs/ict_4h_$(date +\%Y\%m\%d).log 2>&1

# Elliott Wave Scanner - Hourly at :50
50 * * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/1_hour_scanners/elliot_waves_scanner_1h_r1.py >> logs/elliott_1h_$(date +\%Y\%m\%d).log 2>&1

# Elliott Wave Scanner Suite - Multi-timeframe
45 */4 * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/4_hour_scanners/elliott_waves_scanner_4h_r2.py >> logs/elliott_4h_$(date +\%Y\%m\%d).log 2>&1

0 1 * * * cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/daily_scanners/elliott_waves_scanner_1d_r5.py >> logs/elliott_daily_$(date +\%Y\%m\%d).log 2>&1

0 2 * * 1 cd /opt/bb-screener && source .env && /usr/bin/python3 manual_scanners/weekly_scanners/elliot_waves_scanner_1w_r1.py >> logs/elliott_weekly_$(date +\%Y\%m\%d).log 2>&1
EOF
) | crontab -

echo "✅ Cron jobs updated successfully!"
echo "📊 Current cron schedule:"
crontab -l

echo ""
echo "🎯 Scanner Schedule Summary:"
echo "  • BB Scanner: Every hour at :00"
echo "  • ICT 15M: Every 15 minutes at :10, :25, :40, :55"
echo "  • ICT 1H: Every hour at :20"
echo "  • ICT 4H: Every hour at :30"
echo "  • Elliott 1H: Every hour at :50"
echo "  • Elliott 4H: Every 4 hours at :45"
echo "  • Elliott Daily: Daily at 01:00"
echo "  • Elliott Weekly: Mondays at 02:00"
