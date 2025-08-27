#!/bin/bash
# Setup cron job for 4H FVG Scanner

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
ENV_FILE="$PROJECT_ROOT/.env"

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "❌ .env file not found at $ENV_FILE"
    exit 1
fi

# Set up environment variables for cron
export DATABASE_URL="$DATABASE_URL"
export OTHER_SCANNERS_DATABASE_URL="$OTHER_SCANNERS_DATABASE_URL"

# Path to the R1 scanner
SCANNER_PATH="$PROJECT_ROOT/manual_scanners/4_hour_scanners/fair_value_gap_+_fibonacci_scanner_4h_r1.py"

# Create log directory if it doesn't exist
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Remove existing cron job if it exists
(crontab -l 2>/dev/null | grep -v "fair_value_gap_+_fibonacci_scanner_4h_r1.py") | crontab -

# Add new cron job (run every hour)
(crontab -l 2>/dev/null; echo "0 * * * * cd $PROJECT_ROOT && python3 $SCANNER_PATH >> $LOG_DIR/fvg_4h_scanner.log 2>&1") | crontab -

echo "✅ 4H FVG Scanner cron job set up successfully"
echo "📅 Schedule: Every hour (0 * * * *)"
echo "📁 Log file: $LOG_DIR/fvg_4h_scanner.log"
echo "🔧 Scanner: $SCANNER_PATH"

# List current cron jobs
echo ""
echo "📋 Current cron jobs:"
crontab -l
