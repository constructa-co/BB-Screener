#!/bin/bash
# File: scripts/flagpole_integration/setup_flagpole_cron.sh

echo "Setting up Flagpole Scanner cron jobs..."

PROJECT_DIR="/opt/bb-screener"
ENV_FILE="$PROJECT_DIR/.env"
LOG_DIR="$PROJECT_DIR/logs/flagpole"

# Create log directory
mkdir -p "$LOG_DIR"

# 5M Scanner: Run every 5 minutes
CRON_5M="*/5 * * * * cd $PROJECT_DIR && source $ENV_FILE && timeout 240 /usr/bin/python3 $PROJECT_DIR/manual_scanners/5_min_scanners/flagpole_scanner/flagpole_scanner_5m_r1.py >> $LOG_DIR/flagpole_5m_\$(date +\%Y\%m\%d).log 2>&1"

# Remove old entries and add new
(crontab -l 2>/dev/null | grep -v "flagpole_scanner.*r1.py"; echo "$CRON_5M") | crontab -

echo "✅ Flagpole cron job configured to run every 5 minutes"
crontab -l | grep flagpole || echo "⚠️ No flagpole jobs found"
