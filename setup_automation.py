#!/usr/bin/env python3
"""
Setup Automated Scanner Scheduling
"""

import subprocess
import os

def setup_cron_jobs():
    """Set up cron jobs for automated scanning"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('/opt/bb-screener/logs', exist_ok=True)
    
    # Current cron jobs
    current_cron = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    current_jobs = current_cron.stdout.strip().split('\n') if current_cron.stdout else []
    
    # New cron jobs to add
    new_jobs = [
        # BB Scanner - Every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
        "0 */4 * * * cd /opt/bb-screener && source venv/bin/activate && timeout 1800 python main_scanner.py >> /opt/bb-screener/logs/bb_scanner_$(date +\\%Y\\%m\\%d).log 2>&1",
        
        # ICT Scanner - Every 4 hours (offset by 30 mins)
        "30 */4 * * * cd /opt/bb-screener && source venv/bin/activate && timeout 900 python ict_scanner_4h_r8.py >> /opt/bb-screener/logs/ict_scanner_$(date +\\%Y\\%m\\%d).log 2>&1",
        
        # Optional: More frequent for day trading (every hour during market hours)
        "0 9-17 * * 1-5 cd /opt/bb-screener && source venv/bin/activate && timeout 1800 python main_scanner.py >> /opt/bb-screener/logs/bb_hourly_$(date +\\%Y\\%m\\%d).log 2>&1"
    ]
    
    # Combine existing and new jobs
    all_jobs = current_jobs + new_jobs
    
    # Write to temporary file
    with open('/tmp/new_crontab', 'w') as f:
        for job in all_jobs:
            if job.strip():  # Skip empty lines
                f.write(job + '\n')
    
    # Install new crontab
    result = subprocess.run(['crontab', '/tmp/new_crontab'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Cron jobs installed successfully!")
        print("\n📅 Scheduled Jobs:")
        for job in new_jobs:
            print(f"  • {job}")
    else:
        print(f"❌ Failed to install cron jobs: {result.stderr}")
    
    # Clean up
    os.remove('/tmp/new_crontab')

def check_scanner_status():
    """Check if scanners are working"""
    print("\n🔍 Checking Scanner Status...")
    
    # Check if main_scanner.py exists
    if os.path.exists('/opt/bb-screener/main_scanner.py'):
        print("✅ main_scanner.py found")
    else:
        print("❌ main_scanner.py not found")
    
    # Check if ict_scanner_4h_r8.py exists
    if os.path.exists('/opt/bb-screener/ict_scanner_4h_r8.py'):
        print("✅ ict_scanner_4h_r8.py found")
    else:
        print("❌ ict_scanner_4h_r8.py not found")
    
    # Check logs directory
    if os.path.exists('/opt/bb-screener/logs'):
        print("✅ logs directory exists")
    else:
        print("❌ logs directory not found")

if __name__ == "__main__":
    print("🚀 Setting up Automated Scanner Scheduling...")
    setup_cron_jobs()
    check_scanner_status()
    print("\n🎉 Setup complete! Scanners will now run automatically.")
