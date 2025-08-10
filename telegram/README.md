# Telegram Module

This directory contains Telegram-related files for the BB Scanner.

## 📱 Files

### `telegram_alerts.py`
- **Purpose:** Main Telegram alert system
- **Usage:** Sends trade alerts and notifications via Telegram
- **Description:** Core functionality for sending BB Scanner alerts to Telegram channels

### `telegram_integration_example.py`
- **Purpose:** Example implementation of Telegram integration
- **Usage:** Reference for implementing Telegram features
- **Description:** Shows how to integrate Telegram alerts into other parts of the system

### `TELEGRAM_SETUP_GUIDE.md`
- **Purpose:** Setup guide for Telegram integration
- **Usage:** Follow the guide to configure Telegram alerts
- **Description:** Step-by-step instructions for setting up Telegram bot and channels

## 🚀 Setup

1. Follow the setup guide: `TELEGRAM_SETUP_GUIDE.md`
2. Configure your bot token in `config.py`
3. Import and use `telegram_alerts.py` in your main scanner

## 📋 Usage

```python
from telegram.telegram_alerts import send_trade_alert

# Send a trade alert
send_trade_alert(trade_data)
```

## 🔧 Configuration

Make sure your Telegram bot token is configured in `config.py`:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
```
