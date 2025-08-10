#!/usr/bin/env python3
"""
Verification script to check if BB scanner is now capturing comprehensive Excel data
"""

import pandas as pd
import json
from datetime import datetime
import sys

def verify_comprehensive_data():
    """Check if comprehensive data is being captured correctly"""
    
    try:
        # Run database health check
        print("🔍 VERIFYING COMPREHENSIVE DATA CAPTURE")
        print("=" * 60)
        
        import subprocess
        result = subprocess.run([
            'ssh', 'root@165.232.160.52', 
            'cd /opt/bb-screener && source venv/bin/activate && python database_health_check.py'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Database health check completed")
            print(result.stdout)
        else:
            print("❌ Database health check failed")
            print(result.stderr)
            
        # Download latest database export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_filename = f"database_verification_{timestamp}.xlsx"
        
        download_result = subprocess.run([
            'scp', f'root@165.232.160.52:/opt/bb-screener/outputs/database_check_*.xlsx', 
            f'./{export_filename}'
        ], capture_output=True, text=True)
        
        if download_result.returncode == 0:
            print(f"✅ Downloaded latest database export: {export_filename}")
            
            # Analyze the export
            df = pd.read_excel(export_filename, sheet_name='All_Data')
            
            # Find BB scanner trades
            bb_trades = df[df['scan_type'] == 'bb_scanner']
            print(f"\n📊 BB SCANNER TRADE ANALYSIS:")
            print(f"   • Total BB trades: {len(bb_trades)}")
            
            if len(bb_trades) > 0:
                latest_trade = bb_trades.iloc[0]
                print(f"   • Latest trade: {latest_trade['symbol']}")
                print(f"   • BB Score: {latest_trade['bb_score']}")
                print(f"   • Probability: {latest_trade['probability']}%")
                
                # Check comprehensive data
                if pd.notna(latest_trade['scanner_specific_data']):
                    try:
                        json_data = json.loads(latest_trade['scanner_specific_data'])
                        print(f"   • 🎉 COMPREHENSIVE FIELDS: {len(json_data)} additional fields!")
                        
                        # Check for specific Excel fields that should now be present
                        excel_fields_to_check = [
                            'historical_probability', 'market_health', 'sentiment_confidence',
                            'technical_confidence', 'pattern_boost', 'confluence_score',
                            'market_regime', 'btc_health_score', 'alt_season_indicator'
                        ]
                        
                        found_fields = [field for field in excel_fields_to_check if field in json_data]
                        print(f"   • Excel fields found: {len(found_fields)}/{len(excel_fields_to_check)}")
                        print(f"   • Sample fields: {found_fields[:5]}")
                        
                        if len(found_fields) >= 5:
                            print(f"   • ✅ COMPREHENSIVE DATA CAPTURE WORKING!")
                        else:
                            print(f"   • ❌ Still missing comprehensive fields")
                            
                    except Exception as e:
                        print(f"   • ❌ JSON parsing failed: {e}")
                else:
                    print(f"   • ❌ No comprehensive data found")
            else:
                print(f"   • ⏳ No BB scanner trades found yet")
                
        else:
            print("❌ Could not download database export")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    verify_comprehensive_data()