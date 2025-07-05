# config.py - Configuration file for BB Scanner
# Store your API keys and settings here

# === EXCHANGE API KEYS ===

# CoinMarketCap
CMC_API_KEY = "7eaf27ab-6af5-436c-8df2-469d7dce91e7"

# Binance
BINANCE_API_KEY = "SZobyT8HMXQkmuk1jrm1bvTRmmPrU4RPka6CcLgGsTeqj3OvaUyCN8tkmTtcfgjI"
BINANCE_SECRET_KEY = "07V5knOaokx5JjL6epHWV2YfTZ2uxCwCtrFDxtGMvMJLBpP0PGObrhoN56069S1a"

# KuCoin
KUCOIN_API_KEY = "685034bb35f9a40001e243a6"
KUCOIN_SECRET = "9097f861-63b4-4bd6-98be-15f0f0040c0b"
KUCOIN_PASSPHRASE = "buildwithconcrete"

# OKX
OKX_API_KEY = "22beacfa-7f05-4efd-806b-1e0ca23e32be"
OKX_SECRET = "6B933F3F390ED47481F49692406224BF"
OKX_PASSPHRASE = "Buildwithconcrete100!"

# ByBit
BYBIT_API_KEY = "VeCksFQzz7nLssX9Hu"
BYBIT_SECRET = "yRNHLYCskyBAciRGqWanS413LdC8BBKra73M"

# LunarCrush (for sentiment analysis)
LUNAR_API_KEY = "cnj4irl0msoxh98ebhbdtkczcfmbz8vj1bogxefw"

# TokenMetrics (for sentiment analysis)
TOKENMETRICS_API_KEY = "tm-9ca2ced1-cf5c-43f6-89dd-b355cd7bfb67"

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
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# === 3COMMAS SETTINGS (Optional) ===
THREECOMMAS_API_KEY = "YOUR_3COMMAS_API_KEY"
THREECOMMAS_SECRET = "YOUR_3COMMAS_SECRET"
THREECOMMAS_EMAIL_TOKEN = "YOUR_EMAIL_TOKEN"

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