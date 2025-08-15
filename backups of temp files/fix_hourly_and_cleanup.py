#!/usr/bin/env python3
"""
Fix Hourly Scanning and Clean Up Scanner Data
"""

import subprocess
import os

def setup_hourly_scanning():
    """Set up hourly scanning for both scanners"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('/opt/bb-screener/logs', exist_ok=True)
    
    # Hourly scanning for both scanners
    hourly_jobs = [
        # BB Scanner - Every hour
        "0 * * * * cd /opt/bb-screener && source venv/bin/activate && timeout 1800 python main_scanner.py >> /opt/bb-screener/logs/bb_scanner_$(date +\\%Y\\%m\\%d).log 2>&1",
        
        # ICT Scanner - Every hour (offset by 30 minutes)
        "30 * * * * cd /opt/bb-screener && source venv/bin/activate && timeout 900 python ict_scanner_4h_r8.py >> /opt/bb-screener/logs/ict_scanner_$(date +\\%Y\\%m\\%d).log 2>&1"
    ]
    
    # Write to temporary file
    with open('/tmp/hourly_crontab', 'w') as f:
        for job in hourly_jobs:
            f.write(job + '\n')
    
    # Install new crontab
    result = subprocess.run(['crontab', '/tmp/hourly_crontab'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Hourly scanning set up successfully!")
        print("\n📅 NEW Scanner Schedule:")
        print("  • BB Scanner: Every hour (00:00, 01:00, 02:00, etc.)")
        print("  • ICT Scanner: Every hour (00:30, 01:30, 02:30, etc.)")
        print("\n🎯 Benefits:")
        print("  • Catch opportunities ASAP when they appear")
        print("  • Monitor 4h candles throughout their development")
        print("  • Get alerts quickly when conditions are met")
    else:
        print(f"❌ Failed to set up hourly scanning: {result.stderr}")
    
    # Clean up
    os.remove('/tmp/hourly_crontab')

def cleanup_scanner_data():
    """Clean up scanner data in database"""
    
    # Create a separate cleanup script file
    cleanup_script_content = '''#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def cleanup_scanner_data():
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        print("🧹 Cleaning up scanner data...")
        
        # 1. Delete BB Backtest data (not a scanner)
        logger.cursor.execute("""
            DELETE FROM trade_opportunities 
            WHERE scan_id IN (
                SELECT id FROM scan_results 
                WHERE scan_type = 'BB_Backtest_R10'
            )
        """)
        bb_backtest_deleted = logger.cursor.rowcount
        
        # 2. Delete bb_scanner_test data
        logger.cursor.execute("""
            DELETE FROM trade_opportunities 
            WHERE scan_id IN (
                SELECT id FROM scan_results 
                WHERE scan_type = 'bb_scanner_test'
            )
        """)
        bb_test_deleted = logger.cursor.rowcount
        
        # 3. Merge bb_scanner and bb_scanner_4h into bb_scanner_4h
        logger.cursor.execute("""
            UPDATE scan_results 
            SET scan_type = 'bb_scanner_4h' 
            WHERE scan_type = 'bb_scanner'
        """)
        bb_merged = logger.cursor.rowcount
        
        logger.connection.commit()
        
        print(f"✅ Cleanup completed:")
        print(f"  • Deleted {bb_backtest_deleted} BB Backtest records")
        print(f"  • Deleted {bb_test_deleted} bb_scanner_test records")
        print(f"  • Merged {bb_merged} bb_scanner records into bb_scanner_4h")
        
        # Show final scanner counts
        logger.cursor.execute("""
            SELECT scan_type, COUNT(*) as count
            FROM scan_results 
            GROUP BY scan_type
            ORDER BY scan_type
        """)
        
        results = logger.cursor.fetchall()
        print(f"\\n📊 Final Scanner Counts:")
        for row in results:
            print(f"  • {row['scan_type']}: {row['count']} scans")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
    
    logger.close()

if __name__ == "__main__":
    cleanup_scanner_data()
'''
    
    # Write cleanup script to file
    with open('/tmp/cleanup_scanner_data.py', 'w') as f:
        f.write(cleanup_script_content)
    
    # Run cleanup
    result = subprocess.run(['python3', '/tmp/cleanup_scanner_data.py'], 
                          cwd='/opt/bb-screener', capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"⚠️  Warnings: {result.stderr}")
    
    # Clean up temp file
    os.remove('/tmp/cleanup_scanner_data.py')
    
    # Write cleanup script
    with open('/tmp/cleanup_scanner_data.py', 'w') as f:
        f.write(cleanup_script)
    
    # Run cleanup
    result = subprocess.run(['python3', '/tmp/cleanup_scanner_data.py'], 
                          cwd='/opt/bb-screener', capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"⚠️  Warnings: {result.stderr}")
    
    # Clean up temp file
    os.remove('/tmp/cleanup_scanner_data.py')

def show_current_cron():
    """Show current cron jobs"""
    print("\n🔍 Current Cron Jobs:")
    result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  • {line}")
    else:
        print("  No cron jobs found")

if __name__ == "__main__":
    print("🚀 Setting up hourly scanning and cleaning up scanner data...")
    setup_hourly_scanning()
    cleanup_scanner_data()
    show_current_cron()
    print("\n🎉 Complete! Both scanners now run hourly and data is cleaned up.")
