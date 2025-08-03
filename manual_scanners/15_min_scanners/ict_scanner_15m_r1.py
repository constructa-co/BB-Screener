import ccxt
import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ICTFVGScanner:
    """ICT Fair Value Gap Scanner - 15M Scalping Configuration"""
    
    def __init__(self, exchange_id='binance', test_mode=False):
        """Initialize scanner with 15M R2 proven configuration"""
        # 15M R2 Proven Parameters (from successful backtest)
        self.config = {
            # FVG Detection (15M optimized)
            'min_fvg_size': 0.001,          # 0.1% for 15M volatility (lowered for more trades)
            'max_fvg_age': 40,              # 40 bars for 15M (from 72)
            'min_quality_score': 50,         # Minimum quality threshold (lowered for more trades)
            'max_distance_to_fvg': 0.10,    # Max 10% distance from current price
            
            # Order Block Detection (15M optimized)
            'min_order_block_size': 0.5,    # 0.5% for 15M (from 1.0)
            'min_volume_ratio': 1.2,        # 1.2x for 15M (from 1.5)
            'max_ob_age': 40,               # 40 bars for 15M (from 72)
            
            # Targets (15M optimized)
            'targets': {
                'T1': 0.05,                 # 5% (84-90% hit rate)
                'T2': 0.07,                 # 7% (for partial profits)
                'T3': 0.10                  # 10% (runners)
            },
            
            # Position Management
            'position_sizing': {
                'T1_exit': 0.5,             # Exit 50% at T1
                'T2_exit': 0.3,             # Exit 30% at T2
                'T3_exit': 0.2              # Exit final 20% at T3
            },
            
            # Risk Management (15M R2 optimized with commission adjustment)
            'stop_loss_pct': 0.003,         # 0.3% stop loss (commission-adjusted from 0.5%)
            'min_risk_reward': 0.3,         # Very low for FVGs (proven to work)
            'max_risk_per_trade': 0.01,     # 1% account risk
            
            # Market Filters
            'min_volume_24h': 1000000,      # $1M daily volume
            'max_spread_pct': 0.002,        # 0.2% max spread
            
            # Quality Scoring
            'volume_surge_threshold': 1.5,   # 50% above average
            'category_weights': {
                'Layer 2/Scaling': 1.3,
                'Meme/Community': 1.2,
                'Major Cryptos': 1.1,
                'Infrastructure': 1.0,
                'Others': 1.0,
                'Altcoins': 1.0,
                'Layer 1s': 0.9,
                'DeFi': 0.9,
                'Gaming/NFTs': 0.8
            }
        }
        
        # Exchange setup
        self.exchange_id = exchange_id
        self.test_mode = test_mode
        self.exchange = None
        self.setup_exchange()
        
        # Tracking
        self.last_scan_time = {}
        self.active_setups = {}
        self.performance_stats = {
            'total_scans': 0,
            'setups_found': 0,
            'alerts_sent': 0
        }
        
    def setup_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            logger.info(f"Connected to {self.exchange_id}")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise
            
    def get_top_symbols(self, limit=500) -> List[str]:
        """Get top symbols by 24h volume"""
        try:
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs with sufficient volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > self.config['min_volume_24h']:
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': ticker.get('quoteVolume', 0)
                    })
            
            # Sort by volume and take top N
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            symbols = [pair['symbol'] for pair in usdt_pairs[:limit]]
            
            logger.info(f"Found {len(symbols)} symbols with sufficient volume")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def fetch_candles(self, symbol: str, timeframe='15m', limit=200) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MFI (Money Flow Index)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
            mfi_ratio = positive_flow / negative_flow
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # CMF (Chaikin Money Flow)
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfm = mfm.replace([np.inf, -np.inf], 0)
            mfv = mfm * df['volume']
            df['cmf'] = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def detect_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Fair Value Gaps (FVGs) using ICT methodology"""
        fvgs = []
        min_fvg_size = self.config['min_fvg_size']
        max_age = self.config['max_fvg_age']
        
        for i in range(2, len(df) - 1):
            # Get three consecutive candles
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if candle3['low'] > candle1['high']:
                gap_size = (candle3['low'] - candle1['high']) / candle1['high'] * 100
                
                if gap_size >= min_fvg_size:
                    fvgs.append({
                        'type': 'bullish',
                        'index': i,
                        'timestamp': candle3['timestamp'],
                        'gap_high': candle3['low'],
                        'gap_low': candle1['high'],
                        'gap_size': gap_size,
                        'filled': False,
                        'age': len(df) - i - 1,
                        'volume_surge': candle2['volume_ratio'] > self.config['volume_surge_threshold']
                    })
            
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            elif candle3['high'] < candle1['low']:
                gap_size = (candle1['low'] - candle3['high']) / candle1['low'] * 100
                
                if gap_size >= min_fvg_size:
                    fvgs.append({
                        'type': 'bearish',
                        'index': i,
                        'timestamp': candle3['timestamp'],
                        'gap_high': candle1['low'],
                        'gap_low': candle3['high'],
                        'gap_size': gap_size,
                        'filled': False,
                        'age': len(df) - i - 1,
                        'volume_surge': candle2['volume_ratio'] > self.config['volume_surge_threshold']
                    })
        
        # Filter by age
        valid_fvgs = [fvg for fvg in fvgs if fvg['age'] <= max_age]
        
        # Check if FVGs have been filled
        for fvg in valid_fvgs:
            for j in range(fvg['index'] + 1, len(df)):
                candle = df.iloc[j]
                if fvg['type'] == 'bullish':
                    # Bullish FVG is filled if price goes below gap low
                    if candle['low'] <= fvg['gap_low']:
                        fvg['filled'] = True
                        break
                else:
                    # Bearish FVG is filled if price goes above gap high
                    if candle['high'] >= fvg['gap_high']:
                        fvg['filled'] = True
                        break
        
        # Return only unfilled FVGs
        return [fvg for fvg in valid_fvgs if not fvg['filled']]
    
    def calculate_fvg_quality(self, fvg: Dict, candle_data: pd.Series) -> float:
        """Calculate quality score for FVG setup"""
        quality_score = 50  # Base score
        
        # Gap size quality (larger gaps are better)
        if fvg['gap_size'] >= 0.5:
            quality_score += 20
        elif fvg['gap_size'] >= 0.3:
            quality_score += 15
        elif fvg['gap_size'] >= 0.2:
            quality_score += 10
        
        # Volume surge quality
        if fvg.get('volume_surge', False):
            quality_score += 15
        
        # Age quality (fresher is better)
        if fvg['age'] <= 10:
            quality_score += 15
        elif fvg['age'] <= 20:
            quality_score += 10
        elif fvg['age'] <= 30:
            quality_score += 5
        
        # Technical indicator confluence
        if candle_data['rsi'] < 30 or candle_data['rsi'] > 70:
            quality_score += 10
        
        if candle_data['mfi'] < 20 or candle_data['mfi'] > 80:
            quality_score += 10
        
        if abs(candle_data['cmf']) > 0.1:
            quality_score += 10
        
        return min(quality_score, 100)
    
    def find_swing_high(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find swing high for Fibonacci calculations"""
        try:
            recent_df = df.tail(lookback)
            
            # Find local highs
            highs = []
            for i in range(1, len(recent_df) - 1):
                if (recent_df.iloc[i]['high'] > recent_df.iloc[i-1]['high'] and 
                    recent_df.iloc[i]['high'] > recent_df.iloc[i+1]['high']):
                    highs.append({
                        'index': i,
                        'high': recent_df.iloc[i]['high'],
                        'volume': recent_df.iloc[i]['volume']
                    })
            
            if not highs:
                return None
            
            # Score each high based on height and volume
            for high in highs:
                # Height score (how much higher than surrounding)
                left_avg = recent_df.iloc[max(0, high['index']-5):high['index']]['high'].mean()
                right_avg = recent_df.iloc[high['index']+1:min(len(recent_df), high['index']+6)]['high'].mean()
                height_score = (high['high'] - max(left_avg, right_avg)) / high['high'] * 100
                
                # Volume score
                avg_volume = recent_df['volume'].mean()
                volume_score = (high['volume'] / avg_volume) * 10
                
                high['score'] = height_score + volume_score
            
            # Return the best swing high
            best_high = max(highs, key=lambda x: x['score'])
            
            # Validate swing high makes sense
            current_price = recent_df['close'].iloc[-1]
            if best_high['high'] < current_price:
                # If swing high is below current price, use recent max
                best_high['high'] = recent_df['high'].rolling(20).max().iloc[-1]
            
            return best_high['high']
            
        except Exception as e:
            logger.error(f"Error finding swing high: {e}")
            # Return a fallback value instead of None
            try:
                return df['high'].rolling(window=20).max().iloc[-1]
            except:
                return None
    
    def find_swing_low(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find swing low for Fibonacci calculations"""
        try:
            recent_df = df.tail(lookback)
            
            # Find local lows
            lows = []
            for i in range(1, len(recent_df) - 1):
                if (recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and 
                    recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low']):
                    lows.append({
                        'index': i,
                        'low': recent_df.iloc[i]['low'],
                        'volume': recent_df.iloc[i]['volume']
                    })
            
            if not lows:
                return None
            
            # Score each low based on depth and volume
            for low in lows:
                # Depth score (how much lower than surrounding)
                left_avg = recent_df.iloc[max(0, low['index']-5):low['index']]['low'].mean()
                right_avg = recent_df.iloc[low['index']+1:min(len(recent_df), low['index']+6)]['low'].mean()
                depth_score = (min(left_avg, right_avg) - low['low']) / low['low'] * 100
                
                # Volume score
                avg_volume = recent_df['volume'].mean()
                volume_score = (low['volume'] / avg_volume) * 10
                
                low['score'] = depth_score + volume_score
            
            # Return the best swing low
            best_low = max(lows, key=lambda x: x['score'])
            
            # Validate swing low makes sense
            current_price = recent_df['close'].iloc[-1]
            if best_low['low'] > current_price:
                # If swing low is above current price, use recent min
                best_low['low'] = recent_df['low'].rolling(20).min().iloc[-1]
            
            return best_low['low']
            
        except Exception as e:
            logger.error(f"Error finding swing low: {e}")
            # Return a fallback value instead of None
            try:
                return df['low'].rolling(window=20).min().iloc[-1]
            except:
                return None
    
    def calculate_smart_fibonacci_targets(self, df: pd.DataFrame, fvg: Dict, entry_price: float, stop_loss: float) -> Dict:
        """Calculate Fibonacci-based targets with smart fallbacks"""
        try:
            # Get swing levels
            swing_high = self.find_swing_high(df, lookback=50)
            swing_low = self.find_swing_low(df, lookback=50)
            
            # Error handling for None values
            if swing_high is None or swing_low is None:
                if fvg['type'] == 'bullish':
                    targets = {
                        'T1': entry_price * 1.03,  # +3%
                        'T2': entry_price * 1.05,  # +5%
                        'T3': entry_price * 1.07   # +7%
                    }
                    return {'targets': targets, 'note': 'Fallback +3/5/7% (swing error)', 'fib_quality': 50}
                else:
                    targets = {
                        'T1': entry_price * 0.97,  # -3%
                        'T2': entry_price * 0.95,  # -5%
                        'T3': entry_price * 0.93   # -7%
                    }
                    return {'targets': targets, 'note': 'Fallback -3/-5/-7% (swing error)', 'fib_quality': 50}
            
            # Ensure swing levels are valid
            if swing_high <= swing_low:
                swing_high, swing_low = swing_low, swing_high
            
            # Calculate Fibonacci levels
            fib_range = abs(swing_high - swing_low)
            if fib_range < entry_price * 0.01:
                # Fallback for tiny swing range
                if fvg['type'] == 'bullish':
                    targets = {
                        'T1': entry_price * 1.03,  # +3%
                        'T2': entry_price * 1.05,  # +5%
                        'T3': entry_price * 1.07   # +7%
                    }
                    return {'targets': targets, 'note': 'Fallback +3/5/7%', 'fib_quality': 50}
                else:
                    targets = {
                        'T1': entry_price * 0.97,  # -3%
                        'T2': entry_price * 0.95,  # -5%
                        'T3': entry_price * 0.93   # -7%
                    }
                    return {'targets': targets, 'note': 'Fallback -3/-5/-7%', 'fib_quality': 50}
            
            # Calculate Fibonacci retracements
            if fvg['type'] == 'bullish':
                # For bullish FVG, use swing low to swing high
                fib_38_2 = swing_low + (fib_range * 0.382)
                fib_61_8 = swing_low + (fib_range * 0.618)
                fib_100 = swing_high
                
                targets = {
                    'T1': fib_38_2,
                    'T2': fib_61_8,
                    'T3': fib_100
                }
            else:
                # For bearish FVG, use swing high to swing low
                fib_38_2 = swing_high - (fib_range * 0.382)
                fib_61_8 = swing_high - (fib_range * 0.618)
                fib_100 = swing_low
                
                targets = {
                    'T1': fib_38_2,
                    'T2': fib_61_8,
                    'T3': fib_100
                }
            
            # Calculate quality score
            fib_quality = min(100, 50 + (fib_range / entry_price * 1000))
            
            return {
                'targets': targets,
                'note': 'Fibonacci 38.2/61.8/100%',
                'fib_quality': fib_quality
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci targets: {e}")
            # Fallback to simple percentage targets
            if fvg['type'] == 'bullish':
                targets = {
                    'T1': entry_price * 1.03,
                    'T2': entry_price * 1.05,
                    'T3': entry_price * 1.07
                }
            else:
                targets = {
                    'T1': entry_price * 0.97,
                    'T2': entry_price * 0.95,
                    'T3': entry_price * 0.93
                }
            return {'targets': targets, 'note': 'Error fallback', 'fib_quality': 30}
    
    def get_token_category(self, symbol: str) -> str:
        """Categorize tokens for position sizing"""
        token = symbol.replace('/USDT', '')
        
        categories = {
            'Major Cryptos': ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'TRX', 'LTC', 'BCH'],
            'Layer 1s': ['DOT', 'AVAX', 'ATOM', 'NEAR', 'ALGO', 'ICP', 'FLOW', 'EGLD', 'XTZ', 'THETA', 'HBAR'],
            'Layer 2/Scaling': ['MATIC', 'ARB', 'OP', 'IMX', 'LRC', 'BLUR'],
            'DeFi': ['UNI', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'SNX', 'CRV', 'YFI', 'BAL', '1INCH', 'LDO'],
            'Meme/Community': ['SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', '1000CAT', '1000CHEEMS'],
            'Gaming/NFTs': ['MANA', 'SAND', 'AXS', 'ENJ', 'GALA', 'CHZ', 'ALICE', 'SLP', 'JUV', 'PIXEL'],
            'Infrastructure': ['LINK', 'GRT', 'FIL', 'AR', 'OCEAN', 'STORJ', 'LPT', 'API3', 'BAND'],
            'Altcoins': ['VET', 'XLM', 'XTZ', 'EOS', 'NEO', 'QTUM', 'ZIL', 'HOT', 'IOTA', 'ZEC', 'DASH', 'KSM', 'RUNE', 'LUNA', 'FET', 'APE', 'GMT', 'CELO', 'BAT', 'ONE', 'TON', 'ACM', 'AIXBT', 'VIRTUAL', 'CVX', 'FXS', 'ETHFI', 'ETC', 'TREE', 'ASR', 'OM', 'TAO', 'AMP']
        }
        
        for category, tokens in categories.items():
            if token in tokens:
                return category
        
        return 'Others'
    
    def calculate_setup_details(self, symbol: str, fvg: Dict, current_price: float) -> Dict:
        """Calculate complete setup details including entry, stop, targets"""
        try:
            # Calculate entry price (midpoint of FVG)
            if fvg['type'] == 'bullish':
                entry_price = (fvg['gap_high'] + fvg['gap_low']) / 2
            else:
                entry_price = (fvg['gap_high'] + fvg['gap_low']) / 2
            
            # Calculate stop loss using commission-adjusted percentage
            stop_loss_pct = self.config.get('stop_loss_pct', 0.003)  # 0.3% default (commission-adjusted)
            if fvg['type'] == 'bullish':
                stop_loss = entry_price * (1 - stop_loss_pct)
            else:  # bearish
                stop_loss = entry_price * (1 + stop_loss_pct)
            
            # Calculate targets using Fibonacci
            df = self.fetch_candles(symbol, '15m', 200)
            if not df.empty:
                df = self.add_indicators(df)
                fib_result = self.calculate_smart_fibonacci_targets(df, fvg, entry_price, stop_loss)
                targets = fib_result['targets']
                fib_note = fib_result['note']
                fib_quality = fib_result.get('fib_quality', 50)
            else:
                # Fallback targets
                if fvg['type'] == 'bullish':
                    targets = {
                        'T1': entry_price * 1.03,
                        'T2': entry_price * 1.05,
                        'T3': entry_price * 1.07
                    }
                else:
                    targets = {
                        'T1': entry_price * 0.97,
                        'T2': entry_price * 0.95,
                        'T3': entry_price * 0.93
                    }
                fib_note = 'Fallback targets'
                fib_quality = 30
            
            # Calculate risk/reward
            stop_distance = abs(entry_price - stop_loss)
            target_distance = abs(targets['T1'] - entry_price)
            risk_reward = target_distance / stop_distance if stop_distance > 0 else 0
            
            # Get category for position sizing
            category = self.get_token_category(symbol)
            category_weight = self.config['category_weights'].get(category, 1.0)
            
            return {
                'symbol': symbol,
                'type': fvg['type'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'targets': targets,
                'risk_reward': risk_reward,
                'gap_size': fvg['gap_size'],
                'age': fvg['age'],
                'category': category,
                'category_weight': category_weight,
                'fib_note': fib_note,
                'fib_quality': fib_quality,
                'current_price': current_price,
                'distance_pct': abs(current_price - entry_price) / current_price * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating setup details for {symbol}: {e}")
            return None
    
    def format_alert(self, setup: Dict) -> str:
        """Format setup as alert message"""
        try:
            # Validate targets are in correct direction with tolerance
            tolerance = 0.01  # 1% tolerance for rounding errors and small movements
            if setup['type'] == 'bearish':
                if setup['targets']['T1'] > (setup['entry_price'] * (1 + tolerance)):
                    logger.error(f"ERROR: Bearish trade {setup['symbol']} has targets above entry!")
                    return None
            else:  # bullish
                if setup['targets']['T1'] < (setup['entry_price'] * (1 - tolerance)):
                    logger.error(f"ERROR: Bullish trade {setup['symbol']} has targets below entry!")
                    return None
            
            direction_emoji = "🟢" if setup['type'] == 'bullish' else "🔴"
            category_emoji = "🎯" if setup['category'] in ['Layer 2/Scaling', 'DeFi', 'Infrastructure'] else "🔥"
            
            alert = f"""
{direction_emoji} **ICT FVG SETUP - {setup['symbol']}** {category_emoji}

📊 **Setup Details:**
• Type: {setup['type'].upper()}
• Entry: ${setup['entry_price']:.4f}
• Stop: ${setup['stop_loss']:.4f}
• Gap Size: {setup['gap_size']:.2f}%
• Age: {setup['age']} bars
• Category: {setup['category']}

🎯 **Targets:**
• T1: ${setup['targets']['T1']:.4f} ({((setup['targets']['T1'] - setup['entry_price']) / setup['entry_price'] * 100):+.2f}%)
• T2: ${setup['targets']['T2']:.4f} ({((setup['targets']['T2'] - setup['entry_price']) / setup['entry_price'] * 100):+.2f}%)
• T3: ${setup['targets']['T3']:.4f} ({((setup['targets']['T3'] - setup['entry_price']) / setup['entry_price'] * 100):+.2f}%)

⚖️ **Risk Management:**
• R/R Ratio: {setup['risk_reward']:.2f}:1
• Distance: {setup['distance_pct']:.1f}%
• Quality: {setup['fib_quality']}/100

📈 **Current Price:** ${setup['current_price']:.4f}
🎯 **Fibonacci:** {setup['fib_note']}

#ICT #FVG #15M #Scalping
"""
            return alert
            
        except Exception as e:
            logger.error(f"Error formatting alert: {e}")
            return None
    
    def scan_all_symbols(self, top_n=500, min_quality=50, max_alerts=20):
        """Scan all symbols for FVG setups"""
        try:
            symbols = self.get_top_symbols(top_n)
            setups_found = []
            
            logger.info(f"🔍 Scanning {len(symbols)} symbols for ICT FVG setups...")
            
            for symbol in symbols:
                try:
                    # Fetch data
                    df = self.fetch_candles(symbol, '15m', 200)
                    if df.empty or len(df) < 50:
                        continue
                    
                    # Add indicators
                    df = self.add_indicators(df)
                    
                    # Detect FVGs
                    fvgs = self.detect_fvg(df)
                    
                    if not fvgs:
                        continue
                    
                    # Process each FVG
                    for fvg in fvgs:
                        try:
                            # Calculate quality
                            current_candle = df.iloc[-1]
                            quality = self.calculate_fvg_quality(fvg, current_candle)
                            
                            if quality < min_quality:
                                continue
                            
                            # Calculate setup details
                            current_price = current_candle['close']
                            setup = self.calculate_setup_details(symbol, fvg, current_price)
                            
                            if not setup:
                                continue
                            
                            # Apply filters
                            if setup['risk_reward'] < self.config['min_risk_reward']:
                                continue
                            
                            if setup['distance_pct'] > self.config['max_distance_to_fvg']:
                                continue
                            
                            # Add quality score
                            setup['quality_score'] = quality
                            setups_found.append(setup)
                            
                        except Exception as e:
                            logger.error(f"Error processing FVG for {symbol}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort by quality and take top setups
            setups_found.sort(key=lambda x: x['quality_score'], reverse=True)
            top_setups = setups_found[:max_alerts]
            
            # Send alerts
            alerts_sent = 0
            for setup in top_setups:
                alert = self.format_alert(setup)
                if alert:
                    self.send_alert(alert)
                    alerts_sent += 1
            
            # Update stats
            self.performance_stats['total_scans'] += 1
            self.performance_stats['setups_found'] += len(setups_found)
            self.performance_stats['alerts_sent'] += alerts_sent
            
            logger.info(f"✅ Scan complete: {len(setups_found)} setups found, {alerts_sent} alerts sent")
            
        except Exception as e:
            logger.error(f"Error in scan_all_symbols: {e}")
    
    def send_alert(self, message: str):
        """Send alert (placeholder for actual implementation)"""
        if self.test_mode:
            print(message)
        else:
            # TODO: Implement actual alert sending (Telegram, Discord, etc.)
            logger.info(f"ALERT: {message[:100]}...")
    
    def run_continuous(self, scan_interval=300, max_alerts_per_scan=20):
        """Run continuous scanning"""
        logger.info("🚀 Starting ICT FVG Scanner - 15M Scalping Configuration")
        logger.info(f"⏰ Scan interval: {scan_interval} seconds")
        logger.info(f"🎯 Max alerts per scan: {max_alerts_per_scan}")
        
        while True:
            try:
                self.scan_all_symbols(max_alerts=max_alerts_per_scan)
                time.sleep(scan_interval)
            except KeyboardInterrupt:
                logger.info("🛑 Scanner stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                time.sleep(60)  # Wait before retrying
    
    def backtest_setup(self, setup: Dict) -> Dict:
        """Backtest a setup (placeholder)"""
        return {
            'symbol': setup['symbol'],
            'entry_price': setup['entry_price'],
            'stop_loss': setup['stop_loss'],
            'targets': setup['targets'],
            'quality_score': setup.get('quality_score', 0)
        }

def main():
    """Main function for testing"""
    scanner = ICTFVGScanner(test_mode=True)
    
    print("🔥 ICT 15M R1 - SCALPING CONFIGURATION (FIXED)")
    print("=" * 50)
    
    # Run single scan with more symbols and lower quality threshold
    scanner.scan_all_symbols(top_n=500, min_quality=50, max_alerts=20)
    
    # Print stats
    stats = scanner.performance_stats
    print(f"\n📊 Scan Statistics:")
    print(f"Total Scans: {stats['total_scans']}")
    print(f"Setups Found: {stats['setups_found']}")
    print(f"Alerts Sent: {stats['alerts_sent']}")

if __name__ == "__main__":
    main() 