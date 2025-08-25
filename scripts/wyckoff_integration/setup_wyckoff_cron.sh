#!/bin/bash
# scripts/wyckoff_integration/setup_wyckoff_cron.sh
# Wyckoff Scanner Cron Deployment Script
# Schedule: :05 and :35 every hour (non-conflicting with existing scanners)

set -euo pipefail

echo "Setting up Wyckoff Scanner cron job..."

# Define application home and paths
APP_HOME="${APP_HOME:-/opt/bb-screener}"
PY="${PYTHON_BIN:-/usr/bin/python3}"
ENV_FILE="${ENV_FILE:-$APP_HOME/.env}"

# Ensure logs directory exists
LOG_DIR="$APP_HOME/logs/wyckoff"
if [ -w "$(dirname "$LOG_DIR")" ]; then
    mkdir -p "$LOG_DIR"
else
    echo "  ⚠️  Cannot create log directory locally (will be created on target system)"
fi

# Commands to run the Wyckoff R1 scanners
RUN_1H="$PY \"$APP_HOME/manual_scanners/1_hour_scanners/wyckoff_1h_scanner_r1.py\""
RUN_4H="$PY \"$APP_HOME/manual_scanners/4_hour_scanners/wyckoff_scanner_4h_r1.py\""
RUN_15M="$PY \"$APP_HOME/manual_scanners/15_min_scanners/wyckoff_15m_scanner_r1.py\""

# Install crontab entries for all Wyckoff scanners
if [ "$(id -u)" = "0" ] || [ -w "/etc/crontab" ]; then
    # Remove any existing Wyckoff cron entries
    ( crontab -l 2>/dev/null | grep -v wyckoff.*scanner.*r1.py ; \
      # 1H Scanner: :05 and :35 every hour
      echo "5,35 * * * * . $ENV_FILE && cd $APP_HOME && timeout 240 $RUN_1H >> \"$LOG_DIR/wyckoff_1h_\$(date +\%Y\%m\%d).log\" 2>&1" ; \
      # 4H Scanner: :05 every 4 hours (01:05, 05:05, 09:05, 13:05, 17:05, 21:05)
      echo "5 */4 * * * . $ENV_FILE && cd $APP_HOME && timeout 300 $RUN_4H >> \"$LOG_DIR/wyckoff_4h_\$(date +\%Y\%m\%d).log\" 2>&1" ; \
      # 15M Scanner: :02, :17, :32, :47 every hour (avoiding Supply & Demand at :03, :18, :33, :48)
      echo "2,17,32,47 * * * * . $ENV_FILE && cd $APP_HOME && timeout 180 $RUN_15M >> \"$LOG_DIR/wyckoff_15m_\$(date +\%Y\%m\%d).log\" 2>&1" \
    ) | crontab -
    echo "✅ All Wyckoff cron jobs installed successfully"
else
    echo "  ⚠️  Cannot install cron job locally (requires root access)"
    echo "  📋 Cron entries to add manually:"
    echo "  # 1H Scanner: :05 and :35 every hour"
    echo "  5,35 * * * * . $ENV_FILE && cd $APP_HOME && timeout 240 $RUN_1H >> \"$LOG_DIR/wyckoff_1h_\$(date +\%Y\%m\%d).log\" 2>&1"
    echo "  # 4H Scanner: :05 every 4 hours"
    echo "  5 */4 * * * . $ENV_FILE && cd $APP_HOME && timeout 300 $RUN_4H >> \"$LOG_DIR/wyckoff_4h_\$(date +\%Y\%m\%d).log\" 2>&1"
    echo "  # 15M Scanner: :02, :17, :32, :47 every hour"
    echo "  2,17,32,47 * * * * . $ENV_FILE && cd $APP_HOME && timeout 180 $RUN_15M >> \"$LOG_DIR/wyckoff_15m_\$(date +\%Y\%m\%d).log\" 2>&1"
fi

echo "✅ Wyckoff multi-timeframe cron jobs installed:"
echo "  📊 1H Scanner: :05 and :35 every hour (timeout: 240s)"
echo "  📊 4H Scanner: :05 every 4 hours (timeout: 300s)"
echo "  📊 15M Scanner: :02, :17, :32, :47 every hour (timeout: 180s)"
echo "📁 Logs will be written to: $LOG_DIR/wyckoff_*_YYYYMMDD.log"
echo "🔧 Environment sourced from: $ENV_FILE"

# Verify installation
echo -e "\n📋 Current crontab entries:"
crontab -l | grep wyckoff || echo "  ⚠️  No Wyckoff cron entries found"

echo -e "\n🎯 Wyckoff Scanner Cron Deployment Complete!"
echo "   Schedule: :05 and :35 every hour"
echo "   Next run: Check crontab -l for confirmation"
