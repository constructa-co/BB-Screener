#!/bin/bash
# File: scripts/trend_following_integration/setup_trend_following_cron.sh
# Trend Following Scanner Cron Deployment Script

set -e

echo "🚀 Setting up Trend Following Scanner Cron Jobs..."

# Configuration
SCANNER_HOME="/opt/bb-screener"
SCANNER_SCRIPT="manual_scanners/1_hour_scanners/trend_following_scanner_1h_r1.py"
LOG_DIR="logs"
SCANNER_NAME="trend_following_1h"

# Ensure we're in the right directory
cd "$SCANNER_HOME" || exit 1

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$SCANNER_NAME"; then
    echo "⚠️  Cron job for $SCANNER_NAME already exists"
    echo "Current cron jobs:"
    crontab -l 2>/dev/null | grep "$SCANNER_NAME" || true
    echo ""
    read -p "Do you want to remove existing jobs and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing cron jobs..."
        (crontab -l 2>/dev/null | grep -v "$SCANNER_NAME") | crontab -
    else
        echo "❌ Aborting - existing cron jobs preserved"
        exit 1
    fi
fi

# Add new cron job (safe schedule: :08, :38 - avoids all conflicts)
echo "⏰ Adding Trend Following Scanner to cron at :08 and :38 past each hour..."
(crontab -l 2>/dev/null; echo "8,38 * * * * cd $SCANNER_HOME && source .env && timeout 240 python3 $SCANNER_SCRIPT >> $LOG_DIR/${SCANNER_NAME}_\$(date +\%Y\%m\%d).log 2>&1") | crontab -

# Verify cron job was added
echo "✅ Cron job added successfully!"
echo ""
echo "📋 Current cron jobs for Trend Following Scanner:"
crontab -l 2>/dev/null | grep "$SCANNER_NAME" || echo "No jobs found (this shouldn't happen)"

echo ""
echo "🔍 Verification:"
echo "   Scanner Home: $SCANNER_HOME"
echo "   Script: $SCANNER_SCRIPT"
echo "   Log Directory: $LOG_DIR"
echo "   Schedule: :08 and :38 past each hour"
echo "   Timeout: 4 minutes (240 seconds)"

echo ""
echo "📊 Next scheduled execution:"
echo "   $(date -d "$(date +%Y-%m-%d) $(date +%H):08:00" +"%Y-%m-%d %H:%M:%S")"
echo "   $(date -d "$(date +%Y-%m-%d) $(date +%H):38:00" +"%Y-%m-%d %H:%M:%S")"

echo ""
echo "🎯 Cron deployment complete! Trend Following Scanner will run automatically."
echo "💡 Monitor logs with: tail -f $LOG_DIR/${SCANNER_NAME}_\$(date +%Y%m%d).log"
