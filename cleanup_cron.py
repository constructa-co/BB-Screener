#!/usr/bin/env python3
"""
Clean Up Cron Jobs - Remove backtest and organize scanner jobs
"""

import subprocess
import os

def cleanup_cron_jobs():
    """Clean up cron jobs and remove backtest entry"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('/opt/bb-screener/logs', exist_ok=True)
    
    # New clean cron jobs (removing backtest)
    clean_jobs = [
        # BB Scanner - Every 4 hours (core scanning)
        "0 */4 * * * cd /opt/bb-screener && source venv/bin/activate && timeout 1800 python main_scanner.py >> /opt/bb-screener/logs/bb_scanner_$(date +\\%Y\\%m\\%d).log 2>&1",
        
        # ICT Scanner - Every 4 hours (offset by 30 mins)
        "30 */4 * * * cd /opt/bb-screener && source venv/bin/activate && timeout 900 python ict_scanner_4h_r8.py >> /opt/bb-screener/logs/ict_scanner_$(date +\\%Y\\%m\\%d).log 2>&1",
        
        # BB Scanner Market Hours - Every hour during trading (day trading)
        "0 9-17 * * 1-5 cd /opt/bb-screener && source venv/bin/activate && timeout 1800 python main_scanner.py >> /opt/bb-screener/logs/bb_market_hours_$(date +\\%Y\\%m\\%d).log 2>&1"
    ]
    
    # Write to temporary file
    with open('/tmp/clean_crontab', 'w') as f:
        for job in clean_jobs:
            f.write(job + '\n')
    
    # Install new crontab
    result = subprocess.run(['crontab', '/tmp/clean_crontab'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Cron jobs cleaned up successfully!")
        print("\n📅 New Scanner Schedule:")
        print("  • BB Scanner (Core): Every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)")
        print("  • ICT Scanner: Every 4 hours (00:30, 04:30, 08:30, 12:30, 16:30, 20:30)")
        print("  • BB Scanner (Market): Every hour 9:00-17:00, Mon-Fri")
        print("\n🗑️  Removed: Backtest cron job (not a scanner)")
    else:
        print(f"❌ Failed to clean up cron jobs: {result.stderr}")
    
    # Clean up
    os.remove('/tmp/clean_crontab')

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
    print("🧹 Cleaning up cron jobs...")
    cleanup_cron_jobs()
    show_current_cron()
    print("\n🎉 Cleanup complete!")
