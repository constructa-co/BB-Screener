#!/usr/bin/env python3
"""
ICT FVG Scanner 4H - Enhanced Version with Universal Enrichment (R8)
Based on R7 with added database integration and rich data capture

Key Features:
- Fair Value Gap detection (proven 80%+ win rate)
- Real-time scanning of top 500 crypto pairs
- Smart Fibonacci targets with safety constraints
- Partial profit management (50% T1, 30% T2, 20% T3)
- Quality scoring based on R21 backtest results
- Universal enrichment for consistent database output
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
import logging
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the universal enrichment module
from universal_enrichment import UniversalEnrichment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ict_fvg_scanner_4h_r8.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ICTFVGScanner:
    """Enhanced FVG Scanner with universal enrichment for database integration"""
    
    def __init__(self, exchange_id='binance', test_mode=False):
        """Initialize scanner with R21 winning configuration and enrichment"""
        # R21 Proven Parameters (unchanged from R7)
        self.config = {
            # FVG Detection (from R21)
            'min_fvg_size': 0.007,          # 0.7% minimum gap (tightened from 0.6%)
            'max_fvg_age': 30,              # Maximum age in bars
            'min_quality_score': 70,         # Minimum quality threshold
            'max_distance_to_fvg': 0.10,    # Max 10% distance from current price
            
            # Targets (R21 optimized)
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
            
            # Risk Management
            'stop_loss': 0.02,              # 2% stop loss
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
        
        # Initialize universal enrichment
        self.enricher = UniversalEnrichment(exchange_id)
        
        # Tracking
        self.last_scan_time = {}
        self.active_setups = {}
        self.performance_stats = {
            'total_scans': 0,
            'setups_found': 0,
            'alerts_sent': 0
        }
        
        # Database integration
        self.db_logger = None
        try:
            from trade_logger import TradeLogger
            self.db_logger = TradeLogger()
            logger.info("✅ Database logger initialized")
        except Exception as e:
            logger.warning(f"❌ Database logger not available: {e}")
    
    def setup_exchange(self):
        """Initialize exchange connection"""
        try:
            self.exchange = getattr(ccxt, self.exchange_id)()
            logger.info(f"✅ Connected to {self.exchange_id}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to {self.exchange_id}: {e}")
            raise
    
    def get_top_symbols(self, limit=500) -> List[str]:
        """Get top trading symbols by volume"""
        try:
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs and sort by volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > self.config['min_volume_24h']:
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': ticker.get('quoteVolume', 0)
                    })
            
            # Sort by volume and return top symbols
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            return [pair['symbol'] for pair in usdt_pairs[:limit]]
            
        except Exception as e:
            logger.error(f"❌ Error fetching symbols: {e}")
            return []
    
    def fetch_candles(self, symbol: str, timeframe='4h', limit=200) -> pd.DataFrame:
        """Fetch OHLCV data with error handling"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        if df.empty:
            return df
        
        try:
            # RSI
            df['rsi'] = self.calculate_rsi(df['close'])
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_surge'] = df['volume'] / df['volume_ma']
            
            # Price levels
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            
            return df
        except Exception as e:
            logger.error(f"❌ Error adding indicators: {e}")
            return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Fair Value Gaps in the data"""
        fvgs = []
        
        if len(df) < 3:
            return fvgs
        
        try:
            for i in range(1, len(df) - 1):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                next_candle = df.iloc[i+1]
                
                # Bullish FVG (gap up)
                if (previous['high'] < next_candle['low'] and 
                    current['low'] > previous['high'] and 
                    current['high'] < next_candle['low']):
                    
                    gap_size = (next_candle['low'] - previous['high']) / previous['high']
                    
                    if gap_size >= self.config['min_fvg_size']:
                        fvg = {
                            'type': 'bullish',
                            'start_price': previous['high'],
                            'end_price': next_candle['low'],
                            'gap_size': gap_size,
                            'bar_index': i,
                            'timestamp': current['timestamp']
                        }
                        fvgs.append(fvg)
                
                # Bearish FVG (gap down)
                elif (previous['low'] > next_candle['high'] and 
                      current['high'] < previous['low'] and 
                      current['low'] > next_candle['high']):
                    
                    gap_size = (previous['low'] - next_candle['high']) / next_candle['high']
                    
                    if gap_size >= self.config['min_fvg_size']:
                        fvg = {
                            'type': 'bearish',
                            'start_price': next_candle['high'],
                            'end_price': previous['low'],
                            'gap_size': gap_size,
                            'bar_index': i,
                            'timestamp': current['timestamp']
                        }
                        fvgs.append(fvg)
        
        except Exception as e:
            logger.error(f"❌ Error detecting FVG: {e}")
        
        return fvgs
    
    def calculate_fvg_quality(self, fvg: Dict, candle_data: pd.Series) -> float:
        """Calculate quality score for FVG setup"""
        quality_score = 70  # Base score
        
        try:
            # Gap size bonus
            gap_size = fvg['gap_size']
            if gap_size > 0.02:  # 2%+ gap
                quality_score += 10
            elif gap_size > 0.015:  # 1.5%+ gap
                quality_score += 5
            
            # Volume surge bonus
            volume_surge = candle_data.get('volume_surge', 1)
            if volume_surge > 2.0:
                quality_score += 10
            elif volume_surge > 1.5:
                quality_score += 5
            
            # RSI confluence
            rsi = candle_data.get('rsi', 50)
            if fvg['type'] == 'bullish' and rsi < 40:
                quality_score += 5
            elif fvg['type'] == 'bearish' and rsi > 60:
                quality_score += 5
            
            # Age penalty (older FVGs get penalized)
            age_bars = len(candle_data) - fvg['bar_index']
            if age_bars > 20:
                quality_score -= 10
            elif age_bars > 10:
                quality_score -= 5
            
            return max(quality_score, 0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating FVG quality: {e}")
            return 70
    
    def find_swing_high(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find recent swing high for Fibonacci calculations"""
        if len(df) < lookback:
            return df['high'].max()
        
        recent_data = df.tail(lookback)
        return recent_data['high'].max()
    
    def find_swing_low(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find recent swing low for Fibonacci calculations"""
        if len(df) < lookback:
            return df['low'].min()
        
        recent_data = df.tail(lookback)
        return recent_data['low'].min()
    
    def calculate_smart_fibonacci_targets(self, df: pd.DataFrame, fvg: Dict, entry_price: float, stop_loss: float) -> Dict:
        """Calculate Fibonacci-based targets with safety constraints"""
        try:
            swing_high = self.find_swing_high(df)
            swing_low = self.find_swing_low(df)
            
            if fvg['type'] == 'bullish':
                # Bullish setup - targets above entry
                fib_levels = [0.236, 0.382, 0.618, 0.786, 1.0, 1.618]
                range_size = swing_high - swing_low
                
                targets = {}
                for i, level in enumerate(fib_levels, 1):
                    target_price = entry_price + (range_size * level)
                    
                    # Safety constraints
                    if target_price > entry_price * 1.15:  # Max 15% above entry
                        target_price = entry_price * 1.15
                    
                    targets[f'T{i}'] = round(target_price, 6)
                
                return targets
                
            else:
                # Bearish setup - targets below entry
                fib_levels = [0.236, 0.382, 0.618, 0.786, 1.0, 1.618]
                range_size = swing_high - swing_low
                
                targets = {}
                for i, level in enumerate(fib_levels, 1):
                    target_price = entry_price - (range_size * level)
                    
                    # Safety constraints
                    if target_price < entry_price * 0.85:  # Max 15% below entry
                        target_price = entry_price * 0.85
                    
                    targets[f'T{i}'] = round(target_price, 6)
                
                return targets
                
        except Exception as e:
            logger.error(f"❌ Error calculating Fibonacci targets: {e}")
            # Fallback to simple targets
            return {
                'T1': round(entry_price * (1.05 if fvg['type'] == 'bullish' else 0.95), 6),
                'T2': round(entry_price * (1.07 if fvg['type'] == 'bullish' else 0.93), 6),
                'T3': round(entry_price * (1.10 if fvg['type'] == 'bullish' else 0.90), 6)
            }
    
    def get_token_category(self, symbol: str) -> str:
        """Determine token category for weighting"""
        symbol_upper = symbol.upper()
        
        # Major cryptos
        if symbol_upper in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
            return 'Major Cryptos'
        
        # Layer 2/Scaling
        if any(x in symbol_upper for x in ['ARB/USDT', 'OP/USDT', 'MATIC/USDT', 'IMX/USDT']):
            return 'Layer 2/Scaling'
        
        # Infrastructure
        if any(x in symbol_upper for x in ['LINK/USDT', 'DOT/USDT', 'ATOM/USDT', 'SOL/USDT']):
            return 'Infrastructure'
        
        # DeFi
        if any(x in symbol_upper for x in ['UNI/USDT', 'AAVE/USDT', 'COMP/USDT', 'CRV/USDT']):
            return 'DeFi'
        
        # Gaming/NFTs
        if any(x in symbol_upper for x in ['AXS/USDT', 'SAND/USDT', 'MANA/USDT', 'ENJ/USDT']):
            return 'Gaming/NFTs'
        
        # Meme/Community
        if any(x in symbol_upper for x in ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']):
            return 'Meme/Community'
        
        return 'Altcoins'
    
    def calculate_setup_details(self, symbol: str, fvg: Dict, current_price: float) -> Dict:
        """Calculate complete setup details with enrichment"""
        try:
            # Basic setup calculation
            entry_price = current_price
            stop_loss = entry_price * (0.98 if fvg['type'] == 'bullish' else 1.02)
            
            # Get OHLCV for Fibonacci calculations
            df = self.fetch_candles(symbol, '4h', 100)
            if df.empty:
                return {}
            
            df = self.add_indicators(df)
            
            # Calculate targets
            targets = self.calculate_smart_fibonacci_targets(df, fvg, entry_price, stop_loss)
            
            # Calculate quality score
            quality_score = self.calculate_fvg_quality(fvg, df.iloc[-1])
            
            # Apply category weighting
            category = self.get_token_category(symbol)
            category_weight = self.config['category_weights'].get(category, 1.0)
            final_score = quality_score * category_weight
            
            # Create base output for enrichment
            base_output = {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_1': targets.get('T1', entry_price),
                'target_2': targets.get('T2', entry_price),
                'target_3': targets.get('T3', entry_price),
                'timeframe': '4h',
                'quality_score': final_score,
                'scanner_specific_data': {
                    'fvg_type': fvg['type'],
                    'gap_size': fvg['gap_size'],
                    'fvg_age_bars': len(df) - fvg['bar_index'],
                    'category': category,
                    'category_weight': category_weight,
                    'base_quality_score': quality_score,
                    'final_quality_score': final_score,
                    'targets': targets
                }
            }
            
            # Enrich with universal data
            enriched_output = self.enricher.enrich_scanner_output('ict', base_output)
            
            return enriched_output
            
        except Exception as e:
            logger.error(f"❌ Error calculating setup details for {symbol}: {e}")
            return {}
    
    def format_alert(self, setup: Dict) -> str:
        """Format setup for alert/display"""
        try:
            symbol = setup['symbol']
            entry = setup['entry_price']
            stop = setup['stop_loss']
            target1 = setup['target_1']
            probability = setup.get('probability', 0)
            risk_reward = setup.get('risk_reward_ratio', 0)
            
            alert = f"""
🎯 ICT FVG Setup Found: {symbol}
📊 Probability: {probability:.1f}%
⚖️ Risk/Reward: {risk_reward:.2f}:1
💰 Entry: ${entry:.6f}
🛑 Stop: ${stop:.6f}
🎯 Target 1: ${target1:.6f}
📈 Market Regime: {setup.get('market_regime', 'Unknown')}
📊 RSI: {setup.get('rsi', 0):.1f}
📈 Volume Surge: {setup.get('volume_surge', 0):.1f}x
            """
            
            return alert.strip()
            
        except Exception as e:
            logger.error(f"❌ Error formatting alert: {e}")
            return f"ICT Setup: {setup.get('symbol', 'Unknown')}"
    
    def scan_all_symbols(self, top_n=500, min_quality=75, max_alerts=20):
        """Scan all symbols for FVG setups with database logging"""
        logger.info(f"🔍 Starting ICT FVG scan of top {top_n} symbols...")
        
        symbols = self.get_top_symbols(top_n)
        setups_found = 0
        alerts_sent = 0
        
        # Initialize database scan
        scan_id = None
        if self.db_logger:
            try:
                scan_id = self.db_logger.start_scan('ict_scanner_4h')
                logger.info(f"✅ Database scan started: {scan_id}")
            except Exception as e:
                logger.error(f"❌ Failed to start database scan: {e}")
        
        for symbol in symbols:
            try:
                # Fetch data
                df = self.fetch_candles(symbol, '4h', 100)
                if df.empty:
                    continue
                
                # Detect FVGs
                fvgs = self.detect_fvg(df)
                
                for fvg in fvgs:
                    # Check if FVG is still valid (not too old)
                    age_bars = len(df) - fvg['bar_index']
                    if age_bars > self.config['max_fvg_age']:
                        continue
                    
                    # Get current price
                    current_price = df['close'].iloc[-1]
                    
                    # Check distance to FVG
                    fvg_mid = (fvg['start_price'] + fvg['end_price']) / 2
                    distance = abs(current_price - fvg_mid) / current_price
                    
                    if distance > self.config['max_distance_to_fvg']:
                        continue
                    
                    # Calculate setup details with enrichment
                    setup = self.calculate_setup_details(symbol, fvg, current_price)
                    
                    if not setup:
                        continue
                    
                    # Check quality threshold
                    if setup.get('probability', 0) < min_quality:
                        continue
                    
                    setups_found += 1
                    
                    # Log to database
                    if self.db_logger and scan_id:
                        try:
                            # Prepare trade data for database
                            trade_data = {
                                'symbol': setup['symbol'],
                                'exchange': 'Binance',
                                'timeframe': '4H',
                                'bb_score': setup.get('quality_score', 0),
                                'probability': setup.get('probability', 0),
                                'risk_reward_ratio': setup.get('risk_reward_ratio', 0),
                                'current_price': setup.get('current_price', 0),
                                'entry_price': setup['entry_price'],
                                'stop_loss': setup['stop_loss'],
                                'target_1': setup['target_1'],
                                'target_2': setup['target_2'],
                                'target_3': setup['target_3'],
                                'rsi': setup.get('rsi', 0),
                                'mfi': setup.get('mfi', 0),
                                'stochastic_k': setup.get('stochastic_k', 0),
                                'volume_surge': setup.get('volume_surge', 0),
                                'macd_signal': setup.get('macd_signal', 'neutral'),
                                'pattern_type': f"ICT FVG {setup['scanner_specific_data']['fvg_type']}",
                                'pattern_quality': 'GOOD' if setup.get('probability', 0) > 80 else 'FAIR',
                                'confluence_score': setup.get('quality_score', 0),
                                'historical_win_rate': setup.get('historical_win_rate', 0),
                                'category_win_rate': setup.get('category_win_rate', 0),
                                'similar_setups_count': setup.get('similar_setups_count', 0),
                                'market_cap': setup.get('market_cap', 0),
                                'volume_24h': setup.get('volume_24h', 0),
                                'price_change_24h': setup.get('price_change_24h', 0),
                                'scanner_type': 'ict_scanner_4h'
                            }
                            
                            success = self.db_logger.log_trade_opportunity(scan_id, trade_data)
                            if success:
                                logger.info(f"✅ Logged ICT setup: {symbol} -> {setup.get('probability', 0):.1f}%")
                            else:
                                logger.warning(f"❌ Failed to log ICT setup: {symbol}")
                                
                        except Exception as e:
                            logger.error(f"❌ Database logging error for {symbol}: {e}")
                    
                    # Send alert if within limit
                    if alerts_sent < max_alerts:
                        alert = self.format_alert(setup)
                        logger.info(alert)
                        alerts_sent += 1
                    
                    # Store active setup
                    self.active_setups[symbol] = {
                        'setup': setup,
                        'timestamp': datetime.now()
                    }
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {e}")
                continue
        
        # Complete database scan
        if self.db_logger and scan_id:
            try:
                self.db_logger.complete_scan(scan_id, len(symbols), setups_found, 120)
                logger.info(f"✅ Database scan completed: {setups_found} setups found")
            except Exception as e:
                logger.error(f"❌ Failed to complete database scan: {e}")
        
        # Update stats
        self.performance_stats['total_scans'] += 1
        self.performance_stats['setups_found'] += setups_found
        self.performance_stats['alerts_sent'] += alerts_sent
        
        logger.info(f"📊 Scan complete: {setups_found} setups found, {alerts_sent} alerts sent")
        return setups_found
    
    def send_alert(self, message: str):
        """Send alert (placeholder for Telegram integration)"""
        logger.info(f"🔔 ALERT: {message}")
    
    def run_continuous(self, scan_interval=900, max_alerts_per_scan=20):
        """Run continuous scanning"""
        logger.info(f"🚀 Starting continuous ICT FVG scanning (interval: {scan_interval}s)")
        
        while True:
            try:
                self.scan_all_symbols(max_alerts=max_alerts_per_scan)
                logger.info(f"⏰ Next scan in {scan_interval} seconds...")
                time.sleep(scan_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Scanning stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous scanning: {e}")
                time.sleep(60)  # Wait before retrying
    
    def backtest_setup(self, setup: Dict) -> Dict:
        """Backtest a setup (placeholder)"""
        return {
            'win_rate': 0.85,
            'avg_return': 0.04,
            'max_drawdown': 0.02
        }


def main():
    """Main execution function"""
    scanner = ICTFVGScanner()
    
    # Run single scan
    setups_found = scanner.scan_all_symbols(top_n=500, min_quality=75, max_alerts=10)
    print(f"Found {setups_found} ICT FVG setups")


if __name__ == "__main__":
    main() 