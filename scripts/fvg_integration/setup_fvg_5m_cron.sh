#!/bin/bash
# Setup cron job for 5M FVG Scanners

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

# Paths to the R1 scanners
SCANNER_PATH_1="$PROJECT_ROOT/manual_scanners/5_min_scanners/fair_value_gap_scanner/fair_value_gap_+_fibonacci_scanner_5m_r1.py"
SCANNER_PATH_2="$PROJECT_ROOT/manual_scanners/5_min_scanners/fair_value_gap_scanner/fair_value_gap_scanner_5m_r1.py"

# Create log directory if it doesn't exist
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Remove existing cron jobs if they exist
(crontab -l 2>/dev/null | grep -v "fair_value_gap_+_fibonacci_scanner_5m_r1.py") | crontab -
(crontab -l 2>/dev/null | grep -v "fair_value_gap_scanner_5m_r1.py") | crontab -

# Add new cron jobs (run every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * cd $PROJECT_ROOT && python3 $SCANNER_PATH_1 >> $LOG_DIR/fvg_5m_fib_scanner.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "*/5 * * * * cd $PROJECT_ROOT && python3 $SCANNER_PATH_2 >> $LOG_DIR/fvg_5m_simple_scanner.log 2>&1") | crontab -

echo "✅ 5M FVG Scanner cron jobs set up successfully"
echo "📅 Schedule: Every 5 minutes (*/5 * * * *)"
echo "📁 Log files:"
echo "   - $LOG_DIR/fvg_5m_fib_scanner.log (Fibonacci scanner)"
echo "   - $LOG_DIR/fvg_5m_simple_scanner.log (Simple scanner)"
echo "🔧 Scanners:"
echo "   - $SCANNER_PATH_1"
echo "   - $SCANNER_PATH_2"

# List current cron jobs
echo ""
echo "📋 Current cron jobs:"
crontab -l
