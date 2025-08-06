"""
Telegram Alert System for Crypto Scanners
Add this as a new file: telegram_alerts.py

SETUP INSTRUCTIONS:
1. Install Telegram app on your phone
2. Search for @BotFather in Telegram
3. Send /newbot and follow instructions
4. Choose a name like "CryptoScannerBot"
5. You'll receive a bot token like: 1234567890:ABCdefGHIjklmNOPqrstUVwxyz
6. Start a chat with your bot
7. Send any message to your bot
8. Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
9. Find your chat_id in the response
10. Add token and chat_id to config.py
"""

import requests
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Your bot token from BotFather
            chat_id: Your personal chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_trade_alert(self, trade_data: Dict) -> bool:
        """
        Send a formatted trade alert to Telegram
        """
        # Determine urgency emoji based on probability
        if trade_data.get('probability', 0) >= 85:
            emoji = "🔥🔥🔥"
            urgency = "HIGH PROBABILITY"
        elif trade_data.get('probability', 0) >= 75:
            emoji = "🔥🔥"
            urgency = "GOOD SETUP"
        else:
            emoji = "🔥"
            urgency = "OPPORTUNITY"
        
        # Format the message
        message = f"""
{emoji} <b>{urgency} TRADE ALERT</b> {emoji}

<b>Symbol:</b> {trade_data.get('symbol', 'N/A')}
<b>Scanner:</b> {trade_data.get('scanner_type', 'N/A').replace('_', ' ').title()}
<b>Timeframe:</b> {trade_data.get('timeframe', 'N/A')}
<b>Pattern:</b> {trade_data.get('pattern_type', 'N/A')}

<b>📊 Trade Metrics:</b>
• Probability: <b>{trade_data.get('probability', 0):.1f}%</b>
• Risk/Reward: <b>{trade_data.get('risk_reward_ratio', 0):.2f}:1</b>

<b>💰 Price Levels:</b>
• Entry: <code>${trade_data.get('entry_price', 0):.6f}</code>
• Stop Loss: <code>${trade_data.get('stop_loss', 0):.6f}</code>
• Target 1: <code>${trade_data.get('target_1', 0):.6f}</code>
• Target 2: <code>${trade_data.get('target_2', 0):.6f}</code>

<b>📈 Indicators:</b>
• RSI: {trade_data.get('rsi', 0):.1f}
• MFI: {trade_data.get('mfi', 0):.1f}

<b>⏰ Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

<a href="https://www.tradingview.com/chart/?symbol=BINANCE:{trade_data.get('symbol', '').replace('/', '')}">View Chart</a>
"""
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': False
                }
            )
            
            if response.status_code == 200:
                logging.info(f"Telegram alert sent for {trade_data.get('symbol')}")
                return True
            else:
                logging.error(f"Telegram error: {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def send_summary_report(self, summary_data: Dict) -> bool:
        """
        Send daily/hourly summary report
        """
        message = f"""
📊 <b>SCANNER SUMMARY REPORT</b> 📊

<b>Period:</b> {summary_data.get('period', 'Last 24 Hours')}
<b>Total Scans:</b> {summary_data.get('total_scans', 0)}
<b>Opportunities Found:</b> {summary_data.get('opportunities', 0)}
<b>High Probability (≥80%):</b> {summary_data.get('high_prob', 0)}

<b>By Scanner:</b>
{self._format_scanner_summary(summary_data.get('by_scanner', {}))}

<b>Top Opportunities:</b>
{self._format_top_opportunities(summary_data.get('top_opportunities', []))}
"""
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
            )
            return response.status_code == 200
        except:
            return False
    
    def send_error_alert(self, error_message: str, scanner_name: str = "Unknown") -> bool:
        """
        Send error notifications
        """
        message = f"""
⚠️ <b>SCANNER ERROR ALERT</b> ⚠️

<b>Scanner:</b> {scanner_name}
<b>Error:</b> {error_message}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

Please check the logs for more details.
"""
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
            )
            return response.status_code == 200
        except:
            return False
    
    def send_heartbeat(self) -> bool:
        """
        Send periodic heartbeat to confirm system is running
        """
        message = "💚 Scanner system is running normally"
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message
                }
            )
            return response.status_code == 200
        except:
            return False
    
    def _format_scanner_summary(self, scanner_data: Dict) -> str:
        """Format scanner summary for message"""
        lines = []
        for scanner, data in scanner_data.items():
            lines.append(f"• {scanner}: {data['opportunities']} opportunities")
        return '\n'.join(lines) if lines else "No data"
    
    def _format_top_opportunities(self, opportunities: List[Dict]) -> str:
        """Format top opportunities for message"""
        lines = []
        for i, opp in enumerate(opportunities[:5], 1):
            lines.append(
                f"{i}. {opp['symbol']} - {opp['probability']:.0f}% "
                f"(R:R {opp['risk_reward_ratio']:.1f}:1)"
            )
        return '\n'.join(lines) if lines else "No opportunities"


# Integration code for scanners:
"""
# Add to config.py:
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# Add to each scanner (e.g., main_scanner.py):
from telegram_alerts import TelegramNotifier
import config

# Initialize notifier
telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

# After finding a high-probability trade:
if probability >= 75:  # Your threshold
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
        'pattern_type': 'Bollinger Bounce'
    }
    
    # Send alert
    telegram.send_trade_alert(trade_data)

# For error handling:
try:
    # Scanner code
except Exception as e:
    telegram.send_error_alert(str(e), "BB Scanner 4H")
"""

# Test function to verify setup:
def test_telegram_setup(bot_token: str, chat_id: str):
    """
    Test your Telegram setup
    """
    notifier = TelegramNotifier(bot_token, chat_id)
    
    test_trade = {
        'symbol': 'BTC/USDT',
        'scanner_type': 'test_scanner',
        'timeframe': '4H',
        'probability': 85.5,
        'risk_reward_ratio': 3.2,
        'entry_price': 43250.50,
        'stop_loss': 42800.00,
        'target_1': 44500.00,
        'target_2': 45200.00,
        'rsi': 32.5,
        'mfi': 28.3,
        'pattern_type': 'Test Pattern'
    }
    
    print("Sending test alert...")
    if notifier.send_trade_alert(test_trade):
        print("✅ Success! Check your Telegram app")
    else:
        print("❌ Failed! Check your bot token and chat ID") 