#!/usr/bin/env python3
"""
Fix Cron Jobs - Correct Understanding of Scanners
"""

import subprocess
import os

def fix_cron_jobs():
    """Fix cron jobs with correct scanner understanding"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('/opt/bb-screener/logs', exist_ok=True)
    
    # CORRECT cron jobs (only 2 scanners)
    correct_jobs = [
        # BB Scanner (main_scanner.py) - Every 4 hours
        "0 */4 * * * cd /opt/bb-screener && source venv/bin/activate && timeout 1800 python main_scanner.py >> /opt/bb-screener/logs/bb_scanner_$(date +\\%Y\\%m\\%d).log 2>&1",
        
        # ICT Scanner (ict_scanner_4h_r8.py) - Every 4 hours (offset by 30 mins)
        "30 */4 * * * cd /opt/bb-screener && source venv/bin/activate && timeout 900 python ict_scanner_4h_r8.py >> /opt/bb-screener/logs/ict_scanner_$(date +\\%Y\\%m\\%d).log 2>&1"
    ]
    
    # Write to temporary file
    with open('/tmp/correct_crontab', 'w') as f:
        for job in correct_jobs:
            f.write(job + '\n')
    
    # Install new crontab
    result = subprocess.run(['crontab', '/tmp/correct_crontab'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Cron jobs fixed successfully!")
        print("\n📅 CORRECT Scanner Schedule:")
        print("  • BB Scanner (main_scanner.py): Every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)")
        print("  • ICT Scanner (ict_scanner_4h_r8.py): Every 4 hours (00:30, 04:30, 08:30, 12:30, 16:30, 20:30)")
        print("\n🗑️  Removed: Duplicate BB Scanner job (was running same file twice)")
        print("📊 Total Scanners: 2 (BB + ICT)")
    else:
        print(f"❌ Failed to fix cron jobs: {result.stderr}")
    
    # Clean up
    os.remove('/tmp/correct_crontab')

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
    print("🔧 Fixing cron jobs with correct understanding...")
    fix_cron_jobs()
    show_current_cron()
    print("\n🎉 Fix complete!")
