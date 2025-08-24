#!/bin/bash
# scripts/wyckoff_integration/setup_wyckoff_cron.sh
# Wyckoff Scanner Cron Deployment Script
# Schedule: :05 and :35 every hour (non-conflicting with existing scanners)

set -euo pipefail

echo "Setting up Wyckoff Scanner cron job..."

# Define application home and paths
APP_HOME="${APP_HOME:-/root/TradingRobotPlug/BB_Screener}"
PY="${PYTHON_BIN:-$APP_HOME/.venv/bin/python}"
ENV_FILE="${ENV_FILE:-$APP_HOME/.env}"

# Ensure logs directory exists
LOG_DIR="$APP_HOME/logs/wyckoff"
if [ -w "$(dirname "$LOG_DIR")" ]; then
    mkdir -p "$LOG_DIR"
else
    echo "  ⚠️  Cannot create log directory locally (will be created on target system)"
fi

# Command to run the Wyckoff R1 scanner
RUN="$PY \"$APP_HOME/manual_scanners/1_hour_scanners/wyckoff_1h_scanner_r1.py\""

# Install crontab entries for Wyckoff scanner at :05 and :35 past each hour
if [ "$(id -u)" = "0" ] || [ -w "/etc/crontab" ]; then
    ( crontab -l 2>/dev/null | grep -v wyckoff_1h_scanner_r1.py ; \
      echo "5,35 * * * * . $ENV_FILE && cd $APP_HOME && timeout 240 $RUN >> \"$LOG_DIR/wyckoff_1h_\$(date +\%Y\%m\%d).log\" 2>&1" \
    ) | crontab -
    echo "✅ Cron job installed successfully"
else
    echo "  ⚠️  Cannot install cron job locally (requires root access)"
    echo "  📋 Cron entry to add manually:"
    echo "  5,35 * * * * . $ENV_FILE && cd $APP_HOME && timeout 240 $RUN >> \"$LOG_DIR/wyckoff_1h_\$(date +\%Y\%m\%d).log\" 2>&1"
fi

echo "✅ Wyckoff cron installed at :05 and :35."
echo "📁 Logs will be written to: $LOG_DIR/wyckoff_1h_YYYYMMDD.log"
echo "⏱️  Timeout set to 240 seconds to prevent hanging"
echo "🔧 Environment sourced from: $ENV_FILE"

# Verify installation
echo -e "\n📋 Current crontab entries:"
crontab -l | grep wyckoff || echo "  ⚠️  No Wyckoff cron entries found"

echo -e "\n🎯 Wyckoff Scanner Cron Deployment Complete!"
echo "   Schedule: :05 and :35 every hour"
echo "   Next run: Check crontab -l for confirmation"
