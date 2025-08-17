#!/bin/bash
# Setup scheduled sync for database - runs 5 minutes after scanner

echo "Setting up scheduled database sync..."

# Create the cron entries
(crontab -l 2>/dev/null; echo "# BB Scanner and Database Sync") | crontab -
(crontab -l 2>/dev/null; echo "0 */4 * * * cd /opt/bb-screener && /opt/bb-screener/venv/bin/python /opt/bb-screener/main_scanner.py >> /opt/bb-screener/logs/scanner.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "5 */4 * * * cd /opt/bb-screener && /opt/bb-screener/venv/bin/python /opt/bb-screener/fix_all_db_issues.py >> /opt/bb-screener/logs/db_sync.log 2>&1") | crontab -

echo "✅ Scheduled sync configured"
echo "Scanner runs at: 0:00, 4:00, 8:00, 12:00, 16:00, 20:00"
echo "DB sync runs at: 0:05, 4:05, 8:05, 12:05, 16:05, 20:05"

# Create log directory if it doesn't exist
mkdir -p /opt/bb-screener/logs

# Show current crontab
echo ""
echo "Current crontab:"
crontab -l
