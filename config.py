# config.py - Configuration file for BB Scanner
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === EXCHANGE API KEYS (from environment) ===
CMC_API_KEY = os.getenv('CMC_API_KEY')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY')
KUCOIN_SECRET = os.getenv('KUCOIN_SECRET')
KUCOIN_PASSPHRASE = os.getenv('KUCOIN_PASSPHRASE')
OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_SECRET = os.getenv('OKX_SECRET')
OKX_PASSPHRASE = os.getenv('OKX_PASSPHRASE')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_SECRET = os.getenv('BYBIT_SECRET')
LUNAR_API_KEY = os.getenv('LUNAR_API_KEY')
TOKENMETRICS_API_KEY = os.getenv('TOKENMETRICS_API_KEY')

# === SCANNER CONFIGURATION ===
SCANNER_CONFIG = {
    "top_coins_limit": 500,
    "min_volume_24h": 1_000_000,
    "primary_timeframe": "4h",
    "confirmation_timeframes": ["1h", "1d"],
    "min_score_for_alert": 6,
    "max_alerts_per_run": 20,
    "exchanges_to_use": ["binance", "bybit", "okx", "kucoin"],
    "rate_limit_delay": 6,  # Seconds between API calls
}

# Bollinger Band Settings
BB_CONFIG = {
    "period": 20,
    "std_dev": 2,
    "lower_touch_threshold": 1.02,
    "upper_touch_threshold": 0.98,
    "min_candles_required": 100,
    "atr_stop_multiplier": 2.0,
}

# Risk Management Settings
RISK_CONFIG = {
    "default_risk_per_trade": 1.0,
    "max_concurrent_trades": 10,
    "max_correlated_positions": 3,
    "min_score_threshold": 5,
    "max_risk_percent": 15.0,
    "min_risk_reward": 0.3,
    "price_drift_tolerance": 0.015,  # 1.5%
    "tier_leverage": {
        "tier1": 3,
        "tier2": 2,
        "tier3": 1
    },
    "stop_loss_atr_multiplier": 0.5,
    "require_market_regime": True,
}

# Score Thresholds
SCORE_TIERS = {
    "tier1": 8,
    "tier2": 6,
    "tier3": 4
}

# Notification Settings
NOTIFICATION_CONFIG = {
    "telegram_enabled": False,
    "discord_enabled": False,
    "email_enabled": False,
    "alert_cooldown_minutes": 240,
}

# === TELEGRAM SETTINGS (Optional) ===
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', "YOUR_CHAT_ID_HERE")

# === 3COMMAS SETTINGS (Optional) ===
THREECOMMAS_API_KEY = os.getenv('THREECOMMAS_API_KEY', "YOUR_3COMMAS_API_KEY")
THREECOMMAS_SECRET = os.getenv('THREECOMMAS_SECRET', "YOUR_3COMMAS_SECRET")
THREECOMMAS_EMAIL_TOKEN = os.getenv('THREECOMMAS_EMAIL_TOKEN', "YOUR_EMAIL_TOKEN")
THREECOMMAS_BOT_IDS = {
    "tier1_long": "YOUR_TIER1_LONG_BOT_ID",
    "tier1_short": "YOUR_TIER1_SHORT_BOT_ID",
    "tier2_long": "YOUR_TIER2_LONG_BOT_ID",
    "tier2_short": "YOUR_TIER2_SHORT_BOT_ID"
}

# Cloud Deployment Settings
DEPLOYMENT_CONFIG = {
    "scan_interval_minutes": 240,
    "timezone": "UTC",
    "log_level": "INFO",
    "max_log_size_mb": 100,
    "keep_results_days": 30,
}

# File Export Settings
EXPORT_CONFIG = {
    "excel_enabled": True,
    "csv_enabled": True,
    "json_enabled": False,
    "output_directory": "./results",
    "filename_prefix": "bb_bounce",
    "include_timestamp": True,
}
