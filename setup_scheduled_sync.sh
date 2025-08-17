#!/bin/bash
# Setup scheduled sync for database - runs AFTER scanner completes

echo "Setting up scheduled database sync..."

# Remove any existing BB scanner cron entries
(crontab -l 2>/dev/null | grep -v "bb-screener") | crontab -

# Create single cron entry that chains scanner and sync
(crontab -l 2>/dev/null; echo "# BB Scanner and Database Sync - Chained") | crontab -
(crontab -l 2>/dev/null; echo "0 */4 * * * cd /opt/bb-screener && /opt/bb-screener/venv/bin/python /opt/bb-screener/main_scanner.py >> /opt/bb-screener/logs/scanner.log 2>&1 && echo '---SYNC START---' >> /opt/bb-screener/logs/db_sync.log && /opt/bb-screener/venv/bin/python /opt/bb-screener/fix_all_db_issues.py >> /opt/bb-screener/logs/db_sync.log 2>&1") | crontab -

echo "✅ Scheduled sync configured"
echo "Scanner runs at: 0:00, 4:00, 8:00, 12:00, 16:00, 20:00"
echo "DB sync runs: Immediately after scanner completes"

# Create log directory if it doesn't exist
mkdir -p /opt/bb-screener/logs

# Show current crontab
echo ""
echo "Current crontab:"
crontab -l
