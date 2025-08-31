#!/bin/bash
# File: /opt/bb-screener/update_crontab_working_only.sh

# Extract working scanners from test results
RESULTS_FILE=$(ls -t scanner_test_results_*.txt | head -1)

if [ ! -f "$RESULTS_FILE" ]; then
    echo "No test results file found. Run test_all_scanners.sh first."
    exit 1
fi

echo "Creating crontab with only working scanners from: $RESULTS_FILE"

# Start new crontab
cat > /tmp/verified_crontab << 'EOF'
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
BB_DIR=/opt/bb-screener

# VERIFIED WORKING SCANNERS ONLY
EOF

# Add working 1-minute scanners
echo "" >> /tmp/verified_crontab
echo "# 1-MINUTE SCANNERS" >> /tmp/verified_crontab
grep -B1 "✅ SUCCESS" $RESULTS_FILE | grep "1_min_scanners" | while read line; do
    scanner=$(echo $line | awk -F': ' '{print $2}')
    echo "* * * * * cd \$BB_DIR && export \$(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_min_scanners/*/$scanner >> logs/${scanner%.py}.log 2>&1" >> /tmp/verified_crontab
done

# Add working 5-minute scanners
echo "" >> /tmp/verified_crontab
echo "# 5-MINUTE SCANNERS" >> /tmp/verified_crontab
grep -B1 "✅ SUCCESS" $RESULTS_FILE | grep "5_min_scanners" | while read line; do
    scanner=$(echo $line | awk -F': ' '{print $2}')
    echo "*/5 * * * * cd \$BB_DIR && export \$(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/5_min_scanners/*/$scanner >> logs/${scanner%.py}.log 2>&1" >> /tmp/verified_crontab
done

# Add working 15-minute scanners
echo "" >> /tmp/verified_crontab
echo "# 15-MINUTE SCANNERS" >> /tmp/verified_crontab
grep -B1 "✅ SUCCESS" $RESULTS_FILE | grep "15_min_scanners" | while read line; do
    scanner=$(echo $line | awk -F': ' '{print $2}')
    echo "*/15 * * * * cd \$BB_DIR && export \$(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/15_min_scanners/*/$scanner >> logs/${scanner%.py}.log 2>&1" >> /tmp/verified_crontab
done

# Add working 1-hour scanners
echo "" >> /tmp/verified_crontab
echo "# 1-HOUR SCANNERS" >> /tmp/verified_crontab
grep -B1 "✅ SUCCESS" $RESULTS_FILE | grep "1_hour_scanners" | while read line; do
    scanner=$(echo $line | awk -F': ' '{print $2}')
    echo "0 * * * * cd \$BB_DIR && export \$(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/1_hour_scanners/*/$scanner >> logs/${scanner%.py}.log 2>&1" >> /tmp/verified_crontab
done

# Add working 4-hour scanners
echo "" >> /tmp/verified_crontab
echo "# 4-HOUR SCANNERS" >> /tmp/verified_crontab
grep -B1 "✅ SUCCESS" $RESULTS_FILE | grep "4_hour_scanners" | while read line; do
    scanner=$(echo $line | awk -F': ' '{print $2}')
    echo "0 */4 * * * cd \$BB_DIR && export \$(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/4_hour_scanners/*/$scanner >> logs/${scanner%.py}.log 2>&1" >> /tmp/verified_crontab
done

# Add working daily scanners
echo "" >> /tmp/verified_crontab
echo "# DAILY SCANNERS" >> /tmp/verified_crontab
grep -B1 "✅ SUCCESS" $RESULTS_FILE | grep "daily_scanners" | while read line; do
    scanner=$(echo $line | awk -F': ' '{print $2}')
    echo "0 0 * * * cd \$BB_DIR && export \$(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/daily_scanners/*/$scanner >> logs/${scanner%.py}.log 2>&1" >> /tmp/verified_crontab
done

# Add working weekly scanners
echo "" >> /tmp/verified_crontab
echo "# WEEKLY SCANNERS" >> /tmp/verified_crontab
grep -B1 "✅ SUCCESS" $RESULTS_FILE | grep "weekly_scanners" | while read line; do
    scanner=$(echo $line | awk -F': ' '{print $2}')
    echo "0 0 * * 1 cd \$BB_DIR && export \$(cat .env | xargs) && PYTHONPATH=. python3 manual_scanners/weekly_scanners/*/$scanner >> logs/${scanner%.py}.log 2>&1" >> /tmp/verified_crontab
done

echo "Preview of new crontab:"
cat /tmp/verified_crontab

echo ""
echo "Install this crontab? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    crontab /tmp/verified_crontab
    echo "Crontab updated with working scanners only"
else
    echo "Crontab not installed. You can review /tmp/verified_crontab manually."
fi
