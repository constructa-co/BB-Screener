#!/usr/bin/env python3
"""
Test Telegram Notification Setup
Run this script to verify your Telegram bot is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telegram_alerts import TelegramNotifier, test_telegram_setup
import config

def main():
    print("🔔 Testing Telegram Notification Setup")
    print("=" * 50)
    
    # Check if Telegram settings are configured
    if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Telegram bot token not configured!")
        print("Please add your bot token to config.py")
        return False
    
    if not config.TELEGRAM_CHAT_ID or config.TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("❌ Telegram chat ID not configured!")
        print("Please add your chat ID to config.py")
        return False
    
    print(f"✅ Bot Token: {config.TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"✅ Chat ID: {config.TELEGRAM_CHAT_ID}")
    print()
    
    # Test the setup
    print("📱 Sending test notification...")
    success = test_telegram_setup(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    
    if success:
        print("✅ Telegram setup is working!")
        print("📱 Check your Telegram app for the test message")
        
        # Test additional features
        print("\n🧪 Testing additional features...")
        notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        
        # Test heartbeat
        print("💚 Testing heartbeat...")
        if notifier.send_heartbeat():
            print("✅ Heartbeat sent successfully")
        else:
            print("❌ Heartbeat failed")
        
        # Test error alert
        print("⚠️ Testing error alert...")
        if notifier.send_error_alert("Test error message", "Test Scanner"):
            print("✅ Error alert sent successfully")
        else:
            print("❌ Error alert failed")
        
        # Test summary report
        print("📊 Testing summary report...")
        test_summary = {
            'period': 'Test Period',
            'total_scans': 5,
            'opportunities': 3,
            'high_prob': 1,
            'by_scanner': {
                'BB Scanner': {'opportunities': 2},
                'ICT Scanner': {'opportunities': 1}
            },
            'top_opportunities': [
                {'symbol': 'BTC/USDT', 'probability': 85.0, 'risk_reward_ratio': 2.5},
                {'symbol': 'ETH/USDT', 'probability': 78.0, 'risk_reward_ratio': 1.8}
            ]
        }
        
        if notifier.send_summary_report(test_summary):
            print("✅ Summary report sent successfully")
        else:
            print("❌ Summary report failed")
        
        print("\n🎉 All Telegram tests completed!")
        print("Your notification system is ready to use.")
        
    else:
        print("❌ Telegram setup failed!")
        print("Please check:")
        print("1. Your bot token is correct")
        print("2. Your chat ID is correct")
        print("3. You've started a chat with your bot")
        print("4. Your bot has permission to send messages")
    
    return success

if __name__ == "__main__":
    main() 