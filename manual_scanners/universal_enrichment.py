"""
Universal Data Enrichment Module
Shared by all scanners for consistent data capture
"""

import pandas as pd
import numpy as np
from datetime import datetime
import ccxt
import logging
from typing import Dict, List, Optional
import requests
import json

class UniversalEnrichment:
    """
    Provides common enrichment functions for all scanner types
    """
    
    def __init__(self, exchange='binance'):
        """Initialize with exchange and APIs"""
        self.exchange = getattr(ccxt, exchange)()
        self.logger = logging.getLogger(__name__)
        
        # Initialize APIs if available
        self.cmc_api_key = None  # Set from config
        self.lunarcrush_key = None  # Set from config
        
    def enrich_basic_data(self, symbol: str, timeframe: str, entry_price: float, 
                         stop_loss: float, targets: List[float]) -> Dict:
        """
        Add basic enrichment data that applies to all scanners
        """
        try:
            # Get current market data
            ticker = self.exchange.fetch_ticker(symbol)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate basic indicators
            df['rsi'] = self.calculate_rsi(df['close'])
            df['volume_ma'] = df['volume'].rolling(20).mean()
            
            current_rsi = df['rsi'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume_ma'].iloc[-1]
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Market cap and volume (24h)
            market_cap = ticker.get('info', {}).get('market_cap', 0)
            volume_24h = ticker.get('quoteVolume', 0)
            
            # Calculate risk metrics
            risk_percent = abs((entry_price - stop_loss) / entry_price * 100)
            reward_percent = abs((targets[0] - entry_price) / entry_price * 100)
            risk_reward_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
            
            # Price change
            price_change_24h = ticker.get('percentage', 0)
            
            # Calculate spread safely
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            spread_percent = abs((ask - bid) / bid * 100) if bid > 0 else 0
            
            return {
                'current_price': ticker['last'],
                'rsi': round(current_rsi, 2),
                'volume_surge': round(volume_surge, 2),
                'volume_24h': volume_24h,
                'market_cap': market_cap,
                'price_change_24h': price_change_24h,
                'risk_percent': round(risk_percent, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'bid': bid,
                'ask': ask,
                'spread_percent': spread_percent
            }
            
        except Exception as e:
            self.logger.error(f"Error enriching basic data for {symbol}: {e}")
            return {}
    
    def add_market_regime(self, symbol: str) -> Dict:
        """
        Determine current market regime
        """
        try:
            # Get BTC data as market proxy
            btc_ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1d', limit=50)
            btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate trend
            btc_df['sma_20'] = btc_df['close'].rolling(20).mean()
            btc_df['sma_50'] = btc_df['close'].rolling(50).mean()
            
            current_price = btc_df['close'].iloc[-1]
            sma_20 = btc_df['sma_20'].iloc[-1]
            sma_50 = btc_df['sma_50'].iloc[-1]
            
            # Determine regime
            if current_price > sma_20 > sma_50:
                regime = "Bullish"
                strength = min(((current_price - sma_50) / sma_50 * 100), 100)
            elif current_price < sma_20 < sma_50:
                regime = "Bearish"
                strength = min(((sma_50 - current_price) / sma_50 * 100), 100)
            else:
                regime = "Neutral"
                strength = 50
            
            return {
                'market_regime': regime,
                'regime_strength': round(strength, 1),
                'btc_trend': 'Up' if current_price > sma_20 else 'Down'
            }
            
        except Exception as e:
            self.logger.error(f"Error determining market regime: {e}")
            return {
                'market_regime': 'Unknown',
                'regime_strength': 0,
                'btc_trend': 'Unknown'
            }
    
    def add_pattern_recognition(self, df: pd.DataFrame) -> Dict:
        """
        Add basic pattern recognition (simplified from main scanner)
        """
        patterns = {
            'candlestick_patterns': [],
            'chart_patterns': []
        }
        
        try:
            # Simple candlestick patterns
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # Hammer
            body = abs(last_candle['close'] - last_candle['open'])
            lower_wick = last_candle['open'] - last_candle['low'] if last_candle['close'] > last_candle['open'] else last_candle['close'] - last_candle['low']
            
            if lower_wick > body * 2 and last_candle['high'] - max(last_candle['close'], last_candle['open']) < body * 0.3:
                patterns['candlestick_patterns'].append('Hammer')
            
            # Doji
            if body < (last_candle['high'] - last_candle['low']) * 0.1:
                patterns['candlestick_patterns'].append('Doji')
            
            # Engulfing
            if (last_candle['close'] > last_candle['open'] and 
                prev_candle['close'] < prev_candle['open'] and
                last_candle['open'] < prev_candle['close'] and
                last_candle['close'] > prev_candle['open']):
                patterns['candlestick_patterns'].append('Bullish Engulfing')
            
        except Exception as e:
            self.logger.error(f"Error in pattern recognition: {e}")
        
        return patterns
    
    def add_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Find nearest support and resistance levels
        """
        try:
            # Find recent highs and lows
            highs = df['high'].rolling(10).max()
            lows = df['low'].rolling(10).min()
            
            # Get unique levels
            resistance_levels = sorted(set(highs.dropna().values))
            support_levels = sorted(set(lows.dropna().values))
            
            # Find nearest levels
            nearest_resistance = None
            nearest_support = None
            
            for level in resistance_levels:
                if level > current_price:
                    nearest_resistance = level
                    break
            
            for level in reversed(support_levels):
                if level < current_price:
                    nearest_support = level
                    break
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance_percent': abs((current_price - nearest_support) / current_price * 100) if nearest_support else None,
                'resistance_distance_percent': abs((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {e}")
            return {}
    
    def calculate_probability_score(self, scanner_type: str, scanner_score: float, 
                                  enrichment_data: Dict) -> float:
        """
        Calculate enhanced probability based on scanner type and enrichment
        """
        base_probability = scanner_score
        
        # Add bonuses based on confluence
        if enrichment_data.get('market_regime') == 'Bullish':
            base_probability += 5
        
        if enrichment_data.get('volume_surge', 1) > 2:
            base_probability += 3
        
        if enrichment_data.get('risk_reward_ratio', 0) > 3:
            base_probability += 5
        
        # Scanner-specific adjustments
        if scanner_type == 'wyckoff' and enrichment_data.get('volume_surge', 1) > 2.5:
            base_probability += 5  # Volume is critical for Wyckoff
        
        if scanner_type == 'ict' and enrichment_data.get('spread_percent', 1) < 0.1:
            base_probability += 3  # Tight spread good for ICT
        
        return min(base_probability, 95)  # Cap at 95%
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_unified_output(self, base_data: Dict, enrichment_data: Dict, 
                            scanner_specific_data: Dict) -> Dict:
        """
        Create unified output structure for database
        """
        # Combine all data
        unified_data = {
            **base_data,  # Symbol, entry, targets, etc.
            **enrichment_data,  # RSI, volume, market cap, etc.
            'scanner_specific_data': scanner_specific_data,  # JSON field in DB
            'scan_timestamp': datetime.now(),
            'data_version': '2.0'  # Track data structure version
        }
        
        return unified_data
    
    def enrich_scanner_output(self, scanner_type: str, base_output: Dict) -> Dict:
        """
        Main enrichment function called by any scanner
        
        Args:
            scanner_type: 'ict', 'wyckoff', 'elliott', etc.
            base_output: Dictionary with symbol, entry, stop, targets, and scanner-specific data
        
        Returns:
            Fully enriched data ready for database
        """
        symbol = base_output['symbol']
        timeframe = base_output.get('timeframe', '4h')
        
        # Get basic enrichment
        basic_enrichment = self.enrich_basic_data(
            symbol, 
            timeframe,
            base_output['entry_price'],
            base_output['stop_loss'],
            base_output.get('targets', [base_output.get('target_1')])
        )
        
        # Add market regime
        market_data = self.add_market_regime(symbol)
        
        # Get OHLCV for pattern analysis
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Add patterns
        patterns = self.add_pattern_recognition(df)
        
        # Add support/resistance
        sr_levels = self.add_support_resistance(df, basic_enrichment.get('current_price', 0))
        
        # Combine all enrichment
        all_enrichment = {
            **basic_enrichment,
            **market_data,
            **patterns,
            **sr_levels
        }
        
        # Calculate enhanced probability
        base_score = base_output.get('quality_score', 70)
        enhanced_probability = self.calculate_probability_score(
            scanner_type, 
            base_score,
            all_enrichment
        )
        
        all_enrichment['probability'] = enhanced_probability
        
        # Create unified output
        return self.create_unified_output(
            base_output,
            all_enrichment,
            base_output.get('scanner_specific_data', {})
        )


# Usage in any scanner:
"""
# In wyckoff_scanner.py:
from universal_enrichment import UniversalEnrichment

enricher = UniversalEnrichment()

# After finding a trade setup:
base_output = {
    'symbol': 'BTC/USDT',
    'entry_price': 42150,
    'stop_loss': 41500,
    'target_1': 43200,
    'target_2': 44100,
    'timeframe': '4h',
    'quality_score': 82,
    'scanner_specific_data': {
        'phase': 'Spring',
        'volume_surge': 2.5,
        'spring_quality': 85
    }
}

# Enrich the data
enriched_output = enricher.enrich_scanner_output('wyckoff', base_output)

# Send to database
logger.log_trade_opportunity(scan_id, enriched_output)
""" 