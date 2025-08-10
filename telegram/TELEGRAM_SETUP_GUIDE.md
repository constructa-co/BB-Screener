# 🔔 Telegram Notification Setup Guide

## Step-by-Step Setup Instructions

### 1. Create Your Telegram Bot

1. **Install Telegram** on your phone if you haven't already
2. **Search for @BotFather** in Telegram
3. **Send `/newbot`** to BotFather
4. **Choose a name** for your bot (e.g., "CryptoScannerBot")
5. **Choose a username** (must end in 'bot', e.g., "crypto_scanner_bot")
6. **Save the bot token** that BotFather gives you (looks like: `1234567890:ABCdefGHIjklmNOPqrstUVwxyz`)

### 2. Get Your Chat ID

1. **Start a chat** with your new bot
2. **Send any message** to your bot (e.g., "Hello")
3. **Visit this URL** in your browser (replace with your bot token):
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
4. **Find your chat_id** in the response (it will be a number like `123456789`)

### 3. Update Configuration

1. **Edit config.py** and update these lines:
   ```python
   TELEGRAM_BOT_TOKEN = "YOUR_ACTUAL_BOT_TOKEN"
   TELEGRAM_CHAT_ID = "YOUR_ACTUAL_CHAT_ID"
   ```

### 4. Test the Setup

Run the test script:
```bash
python test_telegram.py
```

### 5. Common Issues & Solutions

#### ❌ "chat not found" Error
- **Solution**: Make sure you've sent a message to your bot first
- **Solution**: Double-check your chat ID is correct

#### ❌ "Unauthorized" Error  
- **Solution**: Check your bot token is correct
- **Solution**: Make sure you copied the full token from BotFather

#### ❌ Bot not responding
- **Solution**: Make sure you've started a chat with your bot
- **Solution**: Try sending `/start` to your bot

### 6. Integration with Scanners

Once setup is working, the Telegram notifications will be automatically integrated into your scanners:

- **High-probability trades** (≥75%) will send alerts
- **Error notifications** when scanners fail
- **Summary reports** with scan results
- **Heartbeat messages** to confirm system is running

### 7. Example Alert Format

You'll receive alerts like this:

```
🔥🔥 HIGH PROBABILITY TRADE ALERT 🔥🔥

Symbol: BTC/USDT
Scanner: BB Scanner
Timeframe: 4H
Pattern: Bollinger Bounce

📊 Trade Metrics:
• Probability: 85.5%
• Risk/Reward: 3.20:1

💰 Price Levels:
• Entry: $43250.500000
• Stop Loss: $42800.000000
• Target 1: $44500.000000
• Target 2: $45200.000000

📈 Indicators:
• RSI: 32.5
• MFI: 28.3

⏰ Time: 2025-08-06 16:30:45 UTC

View Chart
```

### 8. Next Steps

1. **Complete the setup** using the steps above
2. **Test the notifications** with the test script
3. **Deploy to Digital Ocean** when ready
4. **Monitor your phone** for trade alerts!

---

**Need Help?** If you're still having issues, check:
- Your bot token is correct and complete
- Your chat ID is the right number
- You've sent at least one message to your bot
- Your bot has permission to send messages 