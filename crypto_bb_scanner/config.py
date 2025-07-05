# config.py - Configuration Module
# Store your API keys and secrets here. Do NOT share this file publicly.

# === EXCHANGE API KEYS ===

# CoinMarketCap
CMC_API_KEY = "7eaf27ab-6af5-436c-8df2-469d7dce91e7"

# Binance
BINANCE_API_KEY = "GalzTyZjmB0PdvlQkakT5IhG0r1khjeINySDs32Hqtv0PEB7MPWJW6K92dBldM6D"
BINANCE_SECRET_KEY = "yYQddhCnUC66etaM8to0TI67nO7sxE1obeSqzbo6bmyhIFptrVKc2WSvbiaXshDf"

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

# LunarCrush
LUNAR_API_KEY = "cnj4irl0msoxh98ebhbdtkczcfmbz8vj1bogxefw"

# TokenMetrics
TOKENMETRICS_API_KEY = "tm-9ca2ced1-cf5c-43f6-89dd-b355cd7bfb67"

# === SCANNER SETTINGS ===

SCANNER_CONFIG = {
    "top_coins_limit": 100,
    "min_volume_24h": 10_000_000,
    "primary_timeframe": "4h",
    "confirmation_timeframe": "1h",
    "rate_limit_delay": 6,  # seconds between API calls
    "exchanges_to_use": ["binance", "bybit", "okx", "kucoin"],
}

# BB Detection Settings
BB_CONFIG = {
    "period": 20,
    "std_dev": 2,
    "atr_stop_multiplier": 2.0,
    "min_candles_required": 100,
}

# Risk Management
RISK_CONFIG = {
    "min_score_threshold": 5,
    "max_risk_percent": 15.0,
    "min_risk_reward": 0.3,
    "price_drift_tolerance": 0.015,  # 1.5%
}