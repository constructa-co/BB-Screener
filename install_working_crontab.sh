#!/bin/bash
# File: /opt/bb-screener/install_working_crontab.sh

cd /opt/bb-screener

# Backup current crontab
crontab -l > crontab_backup_$(date +%Y%m%d_%H%M%S).txt
echo "Current crontab backed up"

# Create new crontab with all confirmed working scanners
cat > /tmp/working_crontab << 'EOF'
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
BB_DIR=/opt/bb-screener

# === 1-MINUTE SCANNERS ===
* * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_min_scanners/fair_value_gap_scanner_1m_r1.py >> logs/fvg_1m.log 2>&1

# === 5-MINUTE SCANNERS (Confirmed Working) ===
*/5 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/5_min_scanners/fair_value_gap_scanner/fair_value_gap_+_fibonacci_scanner_5m_r1.py >> logs/fvg_5m.log 2>&1
*/5 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/5_min_scanners/flagpole_scanner/flagpole_scanner_5m_r1.py >> logs/flagpole_5m.log 2>&1
*/5 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/5_min_scanners/Heikin\ Ashi\ Scanner/heikin_ashi_scanner_r1.py >> logs/ha_5m.log 2>&1
*/5 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/5_minute_scanners/supply_demand_scanner_5m_r1.py >> logs/sd_5m.log 2>&1
*/5 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/5_min_scanners/fibonacci_revisions/fibonacci_retracement_scanner_r1.py >> logs/fibonacci_5m.log 2>&1

# === 15-MINUTE SCANNERS ===
*/15 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/15_min_scanners/ict_scanner_15m_r4.py >> logs/ict_15m.log 2>&1
*/15 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/15_min_scanners/wyckoff_15m_scanner_r1.py >> logs/wyckoff_15m.log 2>&1
*/15 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/15_min_scanners/supply_&_demand_scanner_15m_r0.py >> logs/sd_15m.log 2>&1

# === HOURLY SCANNERS ===
0 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_hour_scanners/elliot_waves_scanner_1h_r1.py >> logs/elliott_1h.log 2>&1
0 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_hour_scanners/ict_scanner_1h_r4.py >> logs/ict_1h.log 2>&1
0 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_hour_scanners/supply_demand_scanner_1h_r1.py >> logs/sd_1h.log 2>&1
0 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_hour_scanners/trend_following_scanner_1h_r1.py >> logs/trend_1h.log 2>&1
0 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_hour_scanners/wyckoff_1h_scanner_r1.py >> logs/wyckoff_1h.log 2>&1
0 * * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 main_scanner.py >> logs/main_scanner.log 2>&1

# === 4-HOUR SCANNERS ===
0 */4 * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/4_hour_scanners/elliott_wave_scanner_4h_r1.py >> logs/elliott_4h.log 2>&1
0 */4 * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/4_hour_scanners/ict_scanner_4h_r1.py >> logs/ict_4h.log 2>&1
0 */4 * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/4_hour_scanners/supply_demand_scanner_4h_r1.py >> logs/sd_4h.log 2>&1
0 */4 * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/4_hour_scanners/wyckoff_4h_scanner_r1.py >> logs/wyckoff_4h.log 2>&1

# === DAILY SCANNERS ===
0 0 * * * cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/daily_scanners/elliott_wave_scanner_daily_r1.py >> logs/elliott_daily.log 2>&1

# === WEEKLY SCANNERS ===
0 0 * * 1 cd $BB_DIR && export $(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/weekly_scanners/elliott_wave_scanner_weekly_r1.py >> logs/elliott_weekly.log 2>&1
EOF

# Install the new crontab
crontab /tmp/working_crontab
echo "Crontab updated with all working scanners"

# Verify installation
echo ""
echo "Verifying crontab installation:"
crontab -l | grep -c "python3"
echo "scanner jobs installed"

echo ""
echo "=== CRONTAB INSTALLATION COMPLETE ==="
echo "All confirmed working scanners are now scheduled to run automatically"
echo "Check logs directory for scanner output files"
echo "Wait 5-10 minutes for first signals to be generated"
