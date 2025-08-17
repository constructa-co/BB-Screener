#!/bin/bash
# Setup scheduled sync for database - runs AFTER scanner completes

echo "Setting up scheduled BB scanner and database sync..."

# Remove existing bb-screener cron entries
(crontab -l 2>/dev/null | grep -v "bb-screener") | crontab -

# Add new cron entry to run every hour (at minute 0)
# This chains the scanner and sync commands - sync only runs if scanner succeeds
(crontab -l 2>/dev/null; echo "0 * * * * cd /opt/bb-screener && source venv/bin/activate && python main_scanner.py >> /opt/bb-screener/scanner.log 2>&1 && python fix_all_db_issues.py >> /opt/bb-screener/db_sync.log 2>&1") | crontab -

echo "✅ Cron job set up successfully!"
echo "📅 Scanner and sync will run every hour at minute 0"
echo "📝 Logs will be written to:"
echo "   - /opt/bb-screener/scanner.log"
echo "   - /opt/bb-screener/db_sync.log"
echo ""
echo "To view current cron jobs: crontab -l"
echo "To view logs: tail -f /opt/bb-screener/scanner.log"
