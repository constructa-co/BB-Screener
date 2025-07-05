# data_fetcher.py - Market Data Collection Module
import ccxt
import requests
import pandas as pd
import pandas_ta as ta
import logging
from typing import Optional, Dict, List
from config import *

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self):
        self.exchanges = self._setup_exchanges()
        self.top_coins = []
        
    def _setup_exchanges(self) -> Dict[str, ccxt.Exchange]:
        """Initialize exchanges with error handling"""
        exchanges = {}
        try:
            exchanges['binance'] = ccxt.binance({
                "apiKey": BINANCE_API_KEY,
                "secret": BINANCE_SECRET_KEY,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "timeout": 30000
            })
            
            exchanges['bybit'] = ccxt.bybit({
                "apiKey": BYBIT_API_KEY,
                "secret": BYBIT_SECRET,
                "enableRateLimit": True,
                "timeout": 30000
            })
            
            exchanges['kucoin'] = ccxt.kucoin({
                "apiKey": KUCOIN_API_KEY,
                "secret": KUCOIN_SECRET,
                "password": KUCOIN_PASSPHRASE,
                "enableRateLimit": True,
                "timeout": 30000
            })
            
            try:
                exchanges['okx'] = ccxt.okx({
                    "apiKey": OKX_API_KEY,
                    "secret": OKX_SECRET,
                    "password": OKX_PASSPHRASE,
                    "enableRateLimit": True,
                    "timeout": 30000,
                    "sandbox": False,
                    "rateLimit": 100,
                    "options": {"defaultType": "spot"}
                })
                logger.info("OKX exchange initialized successfully")
            except Exception as e:
                logger.warning(f"OKX initialization failed: {e} - continuing without OKX")
            
            # Load markets for all exchanges
            for name, exchange in list(exchanges.items()):
                try:
                    exchange.load_markets()
                    logger.info(f"Loaded {len(exchange.markets)} markets for {name}")
                except Exception as e:
                    logger.error(f"Error loading markets for {name}: {e}")
                    exchanges.pop(name, None)
                    
            logger.info(f"Successfully initialized {len(exchanges)} exchanges")
        except Exception as e:
            logger.error(f"Error setting up exchanges: {e}")
        return exchanges

    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchange names"""
        return list(self.exchanges.keys())

    def fetch_top_coins(self, limit: int = 200) -> List[str]:
        """Fetch top coins from CoinMarketCap"""
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
            params = {
                "limit": 300,
                "sort": "volume_24h",
                "cryptocurrency_type": "coins"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            excluded_coins = {
                "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDD", "FDUSD", "PYUSD",
                "WBTC", "WETH", "stETH", "BETH"
            }
            
            quality_coins = []
            for item in data['data']:
                symbol = item['symbol']
                volume_24h = item['quote']['USD']['volume_24h']
                market_cap = item['quote']['USD']['market_cap']
                
                if (symbol not in excluded_coins and 
                    volume_24h > SCANNER_CONFIG["min_volume_24h"] and
                    market_cap > 1_000_000 and
                    len(symbol) <= 8):
                    
                    quality_coins.append(symbol)
            
            # Ensure BTC and ETH are always included
            for priority_coin in ['BTC', 'ETH']:
                if priority_coin not in quality_coins:
                    quality_coins.insert(0, priority_coin)
            
            self.top_coins = quality_coins[:limit]
            logger.info(f"Selected {len(self.top_coins)} coins for analysis")
            return self.top_coins
            
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    def fetch_ohlcv(self, exchange: str, symbol: str, timeframe: str = '4h') -> Optional[pd.DataFrame]:
        """Fetch OHLCV with basic technical indicators"""
        try:
            if exchange not in self.exchanges:
                return None
                
            ex = self.exchanges[exchange]
            market = f"{symbol}/USDT"
            
            if market not in ex.markets:
                alt_markets = [f"{symbol}/USD", f"{symbol}/BUSD"]
                market_found = False
                for alt_market in alt_markets:
                    if alt_market in ex.markets:
                        market = alt_market
                        market_found = True
                        break
                if not market_found:
                    return None
            
            candles = ex.fetch_ohlcv(market, timeframe=timeframe, limit=200)
            if not candles or len(candles) < BB_CONFIG["min_candles_required"]:
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Basic technical indicators
            bb = ta.bbands(df['close'], length=BB_CONFIG["period"], std=BB_CONFIG["std_dev"])
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Momentum indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
            df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
            df['macd_histogram'] = ta.macd(df['close'])['MACDh_12_26_9']
            
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # Volume analysis
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # Volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_pct'] = df['atr'] / df['close']
            
            # Momentum
            df['momentum'] = df['close'].pct_change(periods=5)
            
            return df
            
        except Exception as e:
            logger.debug(f"Error fetching {symbol} from {exchange}: {e}")
            return None

    def validate_current_price(self, exchange: str, symbol: str, historical_entry: float) -> tuple[bool, float]:
        """Validate setup is still valid at current price"""
        try:
            ex = self.exchanges[exchange]
            market = f"{symbol}/USDT"
            
            if market not in ex.markets:
                alt_markets = [f"{symbol}/USD", f"{symbol}/BUSD"]
                for alt_market in alt_markets:
                    if alt_market in ex.markets:
                        market = alt_market
                        break
            
            ticker = ex.fetch_ticker(market)
            current_price = ticker['last']
            
            price_change = abs(current_price - historical_entry) / historical_entry
            is_valid = price_change <= RISK_CONFIG["price_drift_tolerance"]
            
            return is_valid, current_price
            
        except Exception as e:
            logger.debug(f"Error validating current price for {symbol}: {e}")
            return False, historical_entry

    def get_all_market_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch all market data for analysis"""
        all_data = {}
        
        for symbol in self.top_coins:
            symbol_data = {}
            for exchange_name in SCANNER_CONFIG["exchanges_to_use"]:
                if exchange_name in self.exchanges:
                    df = self.fetch_ohlcv(exchange_name, symbol, SCANNER_CONFIG["primary_timeframe"])
                    if df is not None:
                        symbol_data[exchange_name] = df
            
            if symbol_data:
                all_data[symbol] = symbol_data
                
        return all_data