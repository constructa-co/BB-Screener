#!/bin/bash
set -e

echo "Setting up Fair Value Gap Scanner cron jobs..."

# Configuration
PROJECT_DIR="/opt/bb-screener"
ENV_FILE="$PROJECT_DIR/.env"
LOG_DIR="$PROJECT_DIR/logs/fvg"

# Create log directory
mkdir -p "$LOG_DIR"

# 1M Scanner: :07, :22, :37, :52 (avoiding conflicts with existing scanners)
CRON_1M="7,22,37,52 * * * * cd $PROJECT_DIR && source $ENV_FILE && timeout 240 /usr/bin/python3 $PROJECT_DIR/manual_scanners/1_min_scanners/fair_value_gap_scanner_1m_r1.py >> $LOG_DIR/fvg_1m_\$(date +\%Y\%m\%d).log 2>&1"

# Remove old FVG entries and add new
(crontab -l 2>/dev/null | grep -v "fair_value_gap.*scanner.*r1.py"; echo "$CRON_1M") | crontab -

echo "✅ FVG cron jobs configured:"
echo "  1M: :07, :22, :37, :52"

# Verify
crontab -l | grep fair_value_gap || echo "⚠️ No FVG jobs found in crontab"
