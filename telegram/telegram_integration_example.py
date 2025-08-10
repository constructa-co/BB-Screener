#!/usr/bin/env python3
"""
Example: How to Integrate Telegram Notifications into Your Scanners

This shows how to add Telegram alerts to your existing scanner code.
Copy the relevant parts into your scanner files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telegram_alerts import TelegramNotifier
import config

# Initialize Telegram notifier (add this to your scanner)
telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

def example_trade_alert():
    """
    Example: Send a trade alert when a high-probability setup is found
    """
    # This would be in your scanner's trade detection logic
    probability = 85.5
    symbol = "BTC/USDT"
    
    if probability >= 75:  # Your threshold
        trade_data = {
            'symbol': symbol,
            'scanner_type': 'bb_scanner',
            'timeframe': '4H',
            'probability': probability,
            'risk_reward_ratio': 3.2,
            'entry_price': 43250.50,
            'stop_loss': 42800.00,
            'target_1': 44500.00,
            'target_2': 45200.00,
            'rsi': 32.5,
            'mfi': 28.3,
            'pattern_type': 'Bollinger Bounce'
        }
        
        # Send the alert
        success = telegram.send_trade_alert(trade_data)
        if success:
            print(f"✅ Telegram alert sent for {symbol}")
        else:
            print(f"❌ Failed to send Telegram alert for {symbol}")

def example_error_handling():
    """
    Example: Send error alerts when scanner fails
    """
    try:
        # Your scanner code here
        # ... scanner logic ...
        pass
        
    except Exception as e:
        # Send error alert
        error_message = str(e)
        scanner_name = "BB Scanner 4H"
        
        success = telegram.send_error_alert(error_message, scanner_name)
        if success:
            print(f"✅ Error alert sent for {scanner_name}")
        else:
            print(f"❌ Failed to send error alert")

def example_summary_report():
    """
    Example: Send summary report after scan completion
    """
    # This would be called after your scanner completes
    summary_data = {
        'period': 'Last 4 Hours',
        'total_scans': 150,
        'opportunities': 12,
        'high_prob': 3,
        'by_scanner': {
            'BB Scanner': {'opportunities': 8},
            'ICT Scanner': {'opportunities': 4}
        },
        'top_opportunities': [
            {'symbol': 'BTC/USDT', 'probability': 85.0, 'risk_reward_ratio': 2.5},
            {'symbol': 'ETH/USDT', 'probability': 78.0, 'risk_reward_ratio': 1.8},
            {'symbol': 'SOL/USDT', 'probability': 72.0, 'risk_reward_ratio': 2.1}
        ]
    }
    
    success = telegram.send_summary_report(summary_data)
    if success:
        print("✅ Summary report sent")
    else:
        print("❌ Failed to send summary report")

def example_heartbeat():
    """
    Example: Send periodic heartbeat to confirm system is running
    """
    # This could be called every hour or so
    success = telegram.send_heartbeat()
    if success:
        print("✅ Heartbeat sent")
    else:
        print("❌ Failed to send heartbeat")

# Integration with main_scanner.py example:
"""
# Add to main_scanner.py:

from telegram_alerts import TelegramNotifier
import config

# Initialize at the top of your scanner
telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

# In your trade detection loop:
for symbol in symbols:
    # ... your existing scanner logic ...
    
    if probability >= 75:  # High probability threshold
        trade_data = {
            'symbol': symbol,
            'scanner_type': 'bb_scanner',
            'timeframe': '4H',
            'probability': probability,
            'risk_reward_ratio': risk_reward,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'rsi': rsi_value,
            'mfi': mfi_value,
            'pattern_type': pattern_type
        }
        
        # Send alert
        telegram.send_trade_alert(trade_data)

# Error handling:
try:
    # Your scanner code
    pass
except Exception as e:
    telegram.send_error_alert(str(e), "BB Scanner 4H")

# At the end of scan:
telegram.send_summary_report(summary_data)
"""

if __name__ == "__main__":
    print("🔔 Telegram Integration Examples")
    print("=" * 40)
    
    # Test each example
    print("\n1. Testing trade alert...")
    example_trade_alert()
    
    print("\n2. Testing error alert...")
    example_error_handling()
    
    print("\n3. Testing summary report...")
    example_summary_report()
    
    print("\n4. Testing heartbeat...")
    example_heartbeat()
    
    print("\n✅ All examples completed!")
    print("Copy the relevant code into your scanner files.") 