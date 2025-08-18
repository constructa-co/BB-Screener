#!/usr/bin/env python3
"""
Scanner Integration Checklist
Systematically integrate all 22 remaining manual scanners
"""

import os
import subprocess
from datetime import datetime

# Define all scanners to integrate (grouped by timeframe)
SCANNERS_TO_INTEGRATE = {
    '1m': [
        'fair_value_gap_fibonacci_scanner_1m_r0.py',
        'supply_demand_scanner_1m_r0.py'
    ],
    '5m': [
        'fair_value_gap_scanner_5m.py',
        'flagpole_scanner_5m.py',
        'heikin_ashi_scanner_5m.py',
        'support_resistance_scanner_5m.py',
        'fibonacci_retracement_scanner_5m.py',
        'supply_demand_scanner_5m_r0.py'
    ],
    '15m': [
        'ict_scanner_15m_r3.py',
        'supply_demand_scanner_15m_r0.py',
        'wyckoff_scanner_15m_r0.py'
    ],
    '1h': [
        'elliott_waves_scanner_1h_r0.py',
        'ict_scanner_1h_r3.py',
        'supply_demand_scanner_1h_r0.py',
        'trend_following_scanner.py',
        'wyckoff_1h_scanner_r0.py'
    ],
    '4h': [
        'elliott_waves_scanner_4h_r1.py',
        # 'ict_scanner_4h_r8.py',  # Already done as r9
        'supply_demand_scanner_4h_r0.py',
        'fair_value_gap_fibonacci_scanner_4h_r0.py',
        'wyckoff_scanner_4h_r0.py'
    ],
    '1d': [
        'elliott_waves_scanner_1d_r4.py'
    ],
    '1w': [
        'elliott_waves_scanner_1w_r0.py'
    ]
}

def check_scanner_exists(timeframe, scanner_name):
    """Check if scanner file exists"""
    base_path = f"/opt/bb-screener/manual_scanners/{timeframe}_scanners"
    if timeframe == '1m':
        base_path = "/opt/bb-screener/manual_scanners/1_minute_scanners"
    elif timeframe == '5m':
        base_path = "/opt/bb-screener/manual_scanners/5_minute_scanners"
    elif timeframe == '15m':
        base_path = "/opt/bb-screener/manual_scanners/15_minute_scanners"
    elif timeframe == '1h':
        base_path = "/opt/bb-screener/manual_scanners/1_hour_scanners"
    elif timeframe == '4h':
        base_path = "/opt/bb-screener/manual_scanners/4_hour_scanners"
    elif timeframe == '1d':
        base_path = "/opt/bb-screener/manual_scanners/daily_scanners"
    elif timeframe == '1w':
        base_path = "/opt/bb-screener/manual_scanners/weekly_scanners"
    
    file_path = os.path.join(base_path, scanner_name)
    exists = os.path.exists(file_path)
    return exists, file_path

def create_integration_report():
    """Generate integration status report"""
    print("\n" + "="*60)
    print("MANUAL SCANNERS INTEGRATION CHECKLIST")
    print("="*60)
    print(f"Generated: {datetime.now()}")
    print("\n✅ COMPLETED: ict_scanner_4h_r9.py")
    
    total = 0
    found = 0
    missing = []
    
    for timeframe, scanners in SCANNERS_TO_INTEGRATE.items():
        print(f"\n[{timeframe.upper()} TIMEFRAME]")
        for scanner in scanners:
            exists, path = check_scanner_exists(timeframe, scanner)
            total += 1
            if exists:
                found += 1
                print(f"  ✅ {scanner}")
            else:
                missing.append(f"{timeframe}/{scanner}")
                print(f"  ❌ {scanner} - NOT FOUND")
    
    print(f"\n\nSUMMARY:")
    print(f"  Total scanners to integrate: {total}")
    print(f"  Files found: {found}")
    print(f"  Files missing: {len(missing)}")
    
    if missing:
        print(f"\n⚠️ MISSING FILES:")
        for m in missing:
            print(f"  - {m}")
    
    print(f"\n📋 INTEGRATION ORDER (by timeframe):")
    print("  1. Start with daily/weekly (low frequency)")
    print("  2. Then 4h scanners")
    print("  3. Then 1h scanners")
    print("  4. Then 15m scanners")
    print("  5. Then 5m scanners")
    print("  6. Finally 1m scanners (highest frequency)")
    
    return found, total

def generate_integration_plan():
    """Generate detailed integration plan"""
    print("\n" + "="*60)
    print("DETAILED INTEGRATION PLAN")
    print("="*60)
    
    # Integration template
    template = """
INTEGRATION TEMPLATE FOR EACH SCANNER:

1. CREATE NEW VERSION:
   cp {scanner_path} {scanner_path.replace('.py', '_r{next_version}.py')}

2. ADD UNIVERSAL LOGGER INTEGRATION:
   - Add import: from database_and_logging.universal_scanner_logger import UniversalScannerLogger
   - Initialize logger: logger = UniversalScannerLogger('{scanner_name}', 'r{next_version}')
   - Map trade data to universal logger format
   - Add logger.log_trade(trade_data) call
   - Add proper cleanup: logger.connection_pool.closeall()

3. TEST MANUALLY:
   python3 {new_scanner_path} --once --symbols 10 --quality 50

4. VERIFY DATABASE:
   psql $OTHER_SCANNERS_DATABASE_URL -c "SET search_path TO other_scanners; SELECT COUNT(*) FROM other_scanners_trades WHERE scanner_name = '{scanner_name}';"

5. ADD TO CRON (staggered timing):
   # Example for 4h scanners (every 4 hours at different minutes)
   {cron_minute} */4 * * * cd /opt/bb-screener && source .venv/bin/activate && export OTHER_SCANNERS_DATABASE_URL="..." && python3 {new_scanner_path} --once --symbols 500 --quality 75 >> logs/{scanner_name}.log 2>&1

6. MONITOR:
   python3 database_and_logging/monitor_other_scanners.py
"""
    
    print(template)
    
    # Generate specific plans for each timeframe
    cron_minutes = {
        '1w': 0,    # Sunday at 00:00
        '1d': 5,    # Daily at 00:05
        '4h': 15,   # Every 4h at :15 (already used by ICT)
        '1h': 30,   # Every hour at :30
        '15m': 45,  # Every 15m at :45
        '5m': 10,   # Every 5m at :10
        '1m': 20    # Every minute at :20
    }
    
    print("\nSPECIFIC INTEGRATION PLANS:")
    
    for timeframe, scanners in SCANNERS_TO_INTEGRATE.items():
        if timeframe == '4h':
            # Skip ICT scanner as it's already done
            scanners = [s for s in scanners if 'ict_scanner_4h' not in s]
        
        if scanners:
            print(f"\n[{timeframe.upper()} SCANNERS]")
            minute = cron_minutes[timeframe]
            
            for i, scanner in enumerate(scanners):
                exists, path = check_scanner_exists(timeframe, scanner)
                if exists:
                    scanner_name = scanner.replace('.py', '').replace('_r0', '').replace('_r1', '').replace('_r3', '').replace('_r4', '')
                    next_version = get_next_version(scanner)
                    new_scanner_path = path.replace('.py', f'_r{next_version}.py')
                    
                    print(f"\n  {scanner_name}:")
                    print(f"    - Current: {path}")
                    print(f"    - New: {new_scanner_path}")
                    print(f"    - Cron: {minute + i} */{get_cron_interval(timeframe)} * * *")
                    print(f"    - Logger: UniversalScannerLogger('{scanner_name}', 'r{next_version}')")

def get_next_version(scanner_name):
    """Get next version number for scanner"""
    if '_r0.py' in scanner_name:
        return 1
    elif '_r1.py' in scanner_name:
        return 2
    elif '_r3.py' in scanner_name:
        return 4
    elif '_r4.py' in scanner_name:
        return 5
    else:
        return 1

def get_cron_interval(timeframe):
    """Get cron interval for timeframe"""
    intervals = {
        '1w': 168,  # 7 days
        '1d': 24,   # 24 hours
        '4h': 4,    # 4 hours
        '1h': 1,    # 1 hour
        '15m': 0,   # Every 15 minutes (special handling)
        '5m': 0,    # Every 5 minutes (special handling)
        '1m': 0     # Every minute (special handling)
    }
    return intervals.get(timeframe, 1)

if __name__ == "__main__":
    found, total = create_integration_report()
    
    if found == total:
        print("\n✅ ALL SCANNER FILES FOUND - READY FOR INTEGRATION!")
        generate_integration_plan()
    else:
        print(f"\n⚠️ Some scanner files missing. Please locate them first.")
        print("\nNEXT STEPS:")
        print("1. Locate missing scanner files")
        print("2. Copy them to appropriate directories")
        print("3. Run this script again")
        print("4. Begin integration process")
