#!/bin/bash
# File: /opt/bb-screener/fix_all_timezones.sh
# Universal Timezone Fix for All Dashboard Files

cd /opt/bb-screener

echo "=== APPLYING UNIVERSAL TIMEZONE FIX TO ALL DASHBOARD FILES ==="
echo "Date: $(date)"
echo ""

# Backup all dashboard files
echo "1. Creating backups..."
for file in supply_demand_analysis trend_following_analysis fvg_analysis; do
    if [ -f "dashboard/${file}.py" ]; then
        cp "dashboard/${file}.py" "dashboard/${file}.py.backup_universal_$(date +%Y%m%d_%H%M%S)"
        echo "   ✅ Backed up dashboard/${file}.py"
    fi
done
echo ""

# Update supply_demand_analysis.py
echo "2. Updating supply_demand_analysis.py..."
if [ -f "dashboard/supply_demand_analysis.py" ]; then
    # Add import at the top
    sed -i '1i from timezone_helper import smart_timezone_handler' dashboard/supply_demand_analysis.py
    
    # Replace timezone operations
    sed -i 's/\.dt\.tz_localize.*\.dt\.tz_convert/\.dt\.tz_convert/g' dashboard/supply_demand_analysis.py
    
    # Add smart timezone handling after data retrieval
    sed -i '/zones_data\[.detected_at.\] = pd.to_datetime/a\        zones_data = smart_timezone_handler(zones_data, "detected_at", "UTC")' dashboard/supply_demand_analysis.py
    sed -i '/zones_data\[.expires_at.\] = pd.to_datetime/a\        zones_data = smart_timezone_handler(zones_data, "expires_at", "UTC")' dashboard/supply_demand_analysis.py
    sed -i '/zones_data\[.formation_start.\] = pd.to_datetime/a\        zones_data = smart_timezone_handler(zones_data, "formation_start", "UTC")' dashboard/supply_demand_analysis.py
    sed -i '/zones_data\[.formation_end.\] = pd.to_datetime/a\        zones_data = smart_timezone_handler(zones_data, "formation_end", "UTC")' dashboard/supply_demand_analysis.py
    
    echo "   ✅ Updated supply_demand_analysis.py"
fi
echo ""

# Update trend_following_analysis.py
echo "3. Updating trend_following_analysis.py..."
if [ -f "dashboard/trend_following_analysis.py" ]; then
    # Add import at the top
    sed -i '1i from timezone_helper import smart_timezone_handler' dashboard/trend_following_analysis.py
    
    # Replace timezone operations
    sed -i 's/\.dt\.tz_localize.*\.dt\.tz_convert/\.dt\.tz_convert/g' dashboard/trend_following_analysis.py
    
    # Add smart timezone handling after data retrieval
    sed -i '/display_df\[.detected_at.\] = display_df\[.detected_at.\]/a\        display_df = smart_timezone_handler(display_df, "detected_at", "UTC")' dashboard/trend_following_analysis.py
    
    echo "   ✅ Updated trend_following_analysis.py"
fi
echo ""

# Update fvg_analysis.py (already has safe_tz_convert, but let's use the universal handler)
echo "4. Updating fvg_analysis.py..."
if [ -f "dashboard/fvg_analysis.py" ]; then
    # Add import at the top
    sed -i '1i from timezone_helper import smart_timezone_handler' dashboard/fvg_analysis.py
    
    # Replace safe_tz_convert calls with smart_timezone_handler
    sed -i 's/safe_tz_convert(df\[.detected_at.\], uae_tz)/smart_timezone_handler(df, "detected_at", "Asia\/Dubai")["detected_at"]/g' dashboard/fvg_analysis.py
    sed -i 's/safe_tz_convert(df\[.expires_at.\], uae_tz)/smart_timezone_handler(df, "expires_at", "Asia\/Dubai")["expires_at"]/g' dashboard/fvg_analysis.py
    
    echo "   ✅ Updated fvg_analysis.py"
fi
echo ""

echo "=== UNIVERSAL TIMEZONE FIX COMPLETED ==="
echo "All dashboard files updated with smart timezone handling"
echo ""

# Test the imports
echo "5. Testing imports..."
python3 -c "
try:
    from dashboard.timezone_helper import smart_timezone_handler
    print('✅ timezone_helper import successful')
except Exception as e:
    print(f'❌ timezone_helper import failed: {e}')
"

echo ""
echo "6. Ready to restart Streamlit with universal timezone fix"
echo "Run: pkill -f streamlit && nohup streamlit run dashboard/dashboard.py --server.port 8501 --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &"
