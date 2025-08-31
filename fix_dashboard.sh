#!/bin/bash
# File: /opt/bb-screener/fix_dashboard.sh

cd /opt/bb-screener

echo "=== BB-SCREENER DASHBOARD FIX ==="
echo "Backing up existing dashboard..."

# Backup existing dashboard
cp -r dashboard dashboard_backup_$(date +%Y%m%d_%H%M%S)

echo "Installing missing dependencies..."

# Install missing dependencies
pip install streamlit plotly sqlalchemy pandas psycopg2-binary pyarrow python-dotenv

echo "Killing existing streamlit process..."

# Kill existing streamlit
pkill -f streamlit

echo "Starting dashboard with new configuration..."

# Start dashboard with new configuration
export $(cat /opt/bb-screener/.env | xargs)
nohup streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.maxUploadSize 10 \
    --theme.base "dark" \
    > logs/streamlit.log 2>&1 &

echo "Dashboard restarted. Check http://your-server-ip:8501"
echo "Check logs with: tail -f logs/streamlit.log"
