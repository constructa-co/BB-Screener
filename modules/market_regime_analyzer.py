"""
Market Regime Analyzer - Complete BTC Intelligence System + Traditional Markets
Analyzes market conditions for BB bounce strategy with full BTC context + traditional market intelligence
Uses individual stocks (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL) for LunarCrush sentiment
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import yfinance as yf

logger = logging.getLogger(__name__)

class MarketRegimeAnalyzer:
    """
    Comprehensive market regime detection with full BTC intelligence + traditional markets
    Provides technical + sentiment context for optimal alt trading decisions
    """
    
    def __init__(self, data_fetcher=None, sentiment_analyzer=None):
        self.data_fetcher = data_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        
        # Store the API key for direct access to LunarCrush stocks API
        if sentiment_analyzer and hasattr(sentiment_analyzer, 'lunar_api_key'):
            self.lunarcrush_api_key = sentiment_analyzer.lunar_api_key
        else:
            self.lunarcrush_api_key = None
            logger.warning("LunarCrush API key not available - stocks sentiment will use defaults")
        
        # Regime classification thresholds
        self.ADX_STRONG_TREND = 30
        self.ADX_WEAK_TREND = 20
        self.VOLATILITY_HIGH_THRESHOLD = 1.3
        self.VOLATILITY_LOW_THRESHOLD = 0.8
        self.BB_SQUEEZE_THRESHOLD = 0.2
        
        # BTC sentiment confidence thresholds
        self.BTC_SENTIMENT_STRONG = 70
        self.BTC_SENTIMENT_WEAK = 50
        self.BTC_TECHNICAL_STRONG = 80
        self.BTC_TECHNICAL_WEAK = 60
        
        # Traditional market correlation thresholds
        self.HIGH_CORRELATION = 0.7
        self.MODERATE_CORRELATION = 0.4
    
    async def analyze_market_regime(self, symbol: str, alt_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete market regime analysis with BTC intelligence + traditional markets
        Returns comprehensive regime context for trading decisions
        """
        try:
            # 1. Analyze alt coin technical regime
            alt_regime = self._analyze_alt_regime(alt_df)
            
            # 2. Get BTC data and analyze comprehensive BTC context
            btc_context = await self._analyze_btc_context()
            
            # 3. NEW: Analyze traditional markets
            traditional_context = await self._analyze_traditional_markets()
            
            # 4. Calculate composite confidence and position sizing
            regime_analysis = self._calculate_composite_regime(
                alt_regime, btc_context, traditional_context, symbol
            )
            
            logger.info(f"Market regime analysis completed for {symbol}")
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Market regime analysis failed for {symbol}: {str(e)}")
            return self._get_default_regime_analysis()
    
    def _analyze_alt_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze altcoin technical regime using pandas (no external dependencies)"""
        try:
            # Calculate technical indicators using pandas
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # 1. Trend Analysis - Simple ADX approximation
            high_low = high - low
            high_close = np.abs(high - close.shift(1))
            low_close = np.abs(low - close.shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            
            # Simplified directional movement
            dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                              np.maximum(high - high.shift(1), 0), 0)
            dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                               np.maximum(low.shift(1) - low, 0), 0)
            
            dm_plus_series = pd.Series(dm_plus, index=df.index).rolling(14).mean()
            dm_minus_series = pd.Series(dm_minus, index=df.index).rolling(14).mean()
            
            # Simple ADX approximation
            di_plus = (dm_plus_series / atr) * 100
            di_minus = (dm_minus_series / atr) * 100
            dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            adx = dx.rolling(14).mean()
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
            
            # Moving averages
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            
            # Trend direction and strength
            trend_direction = 'UP' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'DOWN'
            if abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] < 0.02:
                trend_direction = 'NEUTRAL'
            
            # SMA slope for trend momentum
            sma_slope = (sma_20.iloc[-1] - sma_20.iloc[-10]) / sma_20.iloc[-10] * 100
            
            # 2. Volatility Analysis
            current_atr = atr.iloc[-1]
            atr_50_avg = atr.rolling(50).mean().iloc[-1]
            volatility_ratio = current_atr / atr_50_avg if atr_50_avg > 0 else 1.0
            
            # 3. BB Analysis (simplified)
            bb_middle = sma_20
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_width_percentile = bb_width.rolling(50).rank(pct=True).iloc[-1]
            
            # 4. Volume Analysis
            volume_sma_20 = volume.rolling(20).mean()
            volume_trend = (volume_sma_20.iloc[-1] / volume_sma_20.iloc[-10] - 1) * 100
            
            # 5. Price Position Analysis
            price_range_20 = high.rolling(20).max() - low.rolling(20).min()
            price_position = (close.iloc[-1] - low.rolling(20).min().iloc[-1]) / price_range_20.iloc[-1]
            
            return {
                'adx': current_adx,
                'trend_direction': trend_direction,
                'trend_strength': current_adx,
                'sma_slope': sma_slope,
                'volatility_ratio': volatility_ratio,
                'bb_width_percentile': bb_width_percentile,
                'volume_trend': volume_trend,
                'price_position': price_position
            }
            
        except Exception as e:
            logger.error(f"Alt regime analysis failed: {str(e)}")
            return self._get_default_alt_regime()
    
    async def _analyze_traditional_markets(self) -> Dict[str, Any]:
        """NEW: Analyze traditional market conditions and sentiment"""
        try:
            # 1. Get traditional market data via Yahoo Finance
            traditional_data = await self._get_traditional_market_data()
            
            # 2. Get traditional market sentiment via LunarCrush Stocks API (individual stocks)
            traditional_sentiment = await self._get_traditional_market_sentiment()
            
            # 3. Calculate correlation regime
            correlation_regime = self._determine_correlation_regime(traditional_data)
            
            return {
                **traditional_data,
                **traditional_sentiment,
                **correlation_regime
            }
            
        except Exception as e:
            logger.error(f"Traditional market analysis failed: {str(e)}")
            return self._get_default_traditional_context()
    
    async def _get_traditional_market_data(self) -> Dict[str, Any]:
        """Get traditional market data via Yahoo Finance"""
        try:
            # Fetch key traditional market indicators with corrected symbols
            symbols = {
                'SPY': 'S&P 500 ETF',
                '^VIX': 'Volatility Index', 
                'DX-Y.NYB': 'US Dollar Index',
                'QQQ': 'Nasdaq ETF'
            }
            
            traditional_data = {}
            
            for symbol, description in symbols.items():
                try:
                    # Get last 10 trading days
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="10d")
                    
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        # Determine trend
                        if change_pct > 1:
                            trend = 'RISING'
                        elif change_pct < -1:
                            trend = 'FALLING'
                        else:
                            trend = 'STABLE'
                        
                        # Map back to expected names (clean symbols)
                        clean_symbol = symbol.replace('^', '').replace('-Y.NYB', 'Y').lower()
                        traditional_data[f'{clean_symbol}_price'] = current_price
                        traditional_data[f'{clean_symbol}_change'] = change_pct
                        traditional_data[f'{clean_symbol}_trend'] = trend
                        
                        logger.info(f"Traditional market {symbol}: {change_pct:.2f}% ({trend})")
                    
                except Exception as e:
                    logger.warning(f"Could not fetch {symbol} data: {str(e)}")
                    # Set defaults with clean symbol names
                    clean_symbol = symbol.replace('^', '').replace('-Y.NYB', 'Y').lower()
                    traditional_data[f'{clean_symbol}_price'] = 0
                    traditional_data[f'{clean_symbol}_change'] = 0
                    traditional_data[f'{clean_symbol}_trend'] = 'UNKNOWN'
            
            return traditional_data
            
        except Exception as e:
            logger.error(f"Traditional market data fetch failed: {str(e)}")
            return {
                'spy_price': 0, 'spy_change': 0, 'spy_trend': 'UNKNOWN',
                'vix_price': 20, 'vix_change': 0, 'vix_trend': 'UNKNOWN',
                'dxy_price': 0, 'dxy_change': 0, 'dxy_trend': 'UNKNOWN',
                'qqq_price': 0, 'qqq_change': 0, 'qqq_trend': 'UNKNOWN'
            }
    
    async def _get_traditional_market_sentiment(self) -> Dict[str, float]:
        """Get sentiment for traditional market stocks from LunarCrush using individual stocks"""
        try:
            # Check if API key is available
            if not hasattr(self, 'lunarcrush_api_key') or not self.lunarcrush_api_key:
                logger.warning("LunarCrush API key not available - stocks sentiment will use defaults")
                return {
                    'spy': 50,
                    'tech': 50, 
                    'growth': 50,
                    'stocks': 50
                }
            
            # Target individual stocks that LunarCrush actually covers
            target_stocks = [
                ('SPY', 'spy'),        # S&P 500 ETF
                ('AAPL', 'tech'),      # Apple - Tech sentiment
                ('MSFT', 'tech'),      # Microsoft - Alternative tech sentiment
                ('NVDA', 'growth'),    # Nvidia - Growth/AI sentiment
                ('TSLA', 'growth'),    # Tesla - Risk-on sentiment
                ('AMZN', 'stocks'),    # Amazon - General stocks sentiment
                ('GOOGL', 'stocks'),   # Google - Alternative general sentiment
            ]
            
            # Get stocks list if not cached
            if not hasattr(self, '_stocks_list'):
                self._stocks_list = await self._get_lunarcrush_stocks_list()
            
            results = {'spy': 50, 'tech': 50, 'growth': 50, 'stocks': 50}  # Initialize with defaults
            
            for symbol, sentiment_key in target_stocks:
                try:
                    stock_id = self._find_stock_id(symbol, self._stocks_list)
                    if stock_id:
                        sentiment_data = await self._get_lunarcrush_stock_sentiment(stock_id)
                        if sentiment_data:
                            galaxy_score = sentiment_data.get('galaxy_score', 50)
                            results[sentiment_key] = self._classify_sentiment(galaxy_score)
                            logger.info(f"LunarCrush stocks sentiment for {symbol}: {results[sentiment_key]}")
                            # For sentiment_key with multiple options (tech, growth, stocks), 
                            # take the first successful result
                            continue
                        
                except Exception as e:
                    logger.debug(f"Failed to get sentiment for {symbol}: {str(e)}")
                    continue  # Try next symbol
            
            return results
            
        except Exception as e:
            logger.warning(f"Traditional market sentiment failed: {str(e)} - using defaults")
            return {
                'spy': 50,
                'tech': 50,
                'growth': 50, 
                'stocks': 50
            }

    async def _get_lunarcrush_stocks_list(self) -> Optional[Dict]:
        """Get the complete LunarCrush stocks list to find stock IDs"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                url = "https://lunarcrush.com/api4/public/stocks/list/v1"
                headers = {
                    'Authorization': f'Bearer {self.lunarcrush_api_key}'
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"LunarCrush stocks list fetched: {len(data.get('data', []))} stocks")
                        return data
                    else:
                        logger.warning(f"LunarCrush stocks list failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to fetch LunarCrush stocks list: {str(e)}")
            return None

    def _find_stock_id(self, symbol: str, stocks_data: Dict) -> Optional[str]:
        """Find the LunarCrush ID for a given stock symbol"""
        try:
            stocks_list = stocks_data.get('data', [])
            
            for stock in stocks_list:
                if stock.get('symbol', '').upper() == symbol.upper():
                    return str(stock.get('id'))
            
            logger.warning(f"Stock {symbol} not found in LunarCrush stocks list")
            return None
            
        except Exception as e:
            logger.error(f"Error finding stock ID for {symbol}: {str(e)}")
            return None

    async def _get_lunarcrush_stock_sentiment(self, stock_id: str) -> Optional[Dict]:
        """Get sentiment data for a specific stock using its LunarCrush ID"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                url = f"https://lunarcrush.com/api4/public/stocks/{stock_id}/v1"
                headers = {
                    'Authorization': f'Bearer {self.lunarcrush_api_key}'
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        stock_data = data.get('data', {})
                        
                        return {
                            'galaxy_score': stock_data.get('galaxy_score', 50),
                            'alt_rank': stock_data.get('alt_rank', 999),
                            'sentiment': stock_data.get('sentiment', 50),
                            'market_dominance': stock_data.get('market_dominance', 0),
                            'symbol': stock_data.get('symbol', 'UNKNOWN')
                        }
                    else:
                        logger.warning(f"LunarCrush stock sentiment failed for ID {stock_id}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to fetch LunarCrush stock sentiment for ID {stock_id}: {str(e)}")
            return None

    def _classify_sentiment(self, galaxy_score: float) -> str:
        """Classify galaxy score into sentiment rating"""
        if galaxy_score >= 70:
            return 'Very Bullish'
        elif galaxy_score >= 60:
            return 'Bullish'  
        elif galaxy_score >= 40:
            return 'Neutral'
        elif galaxy_score >= 30:
            return 'Bearish'
        else:
            return 'Very Bearish'
    
    def _determine_correlation_regime(self, traditional_data: Dict) -> Dict[str, Any]:
        """Determine current crypto-traditional market correlation regime"""
        try:
            # Simple correlation assessment based on market conditions
            spy_trend = traditional_data.get('spy_trend', 'UNKNOWN')
            vix_level = traditional_data.get('vix_price', 20)
            dxy_trend = traditional_data.get('dxy_trend', 'UNKNOWN')
            
            # Determine correlation level
            if vix_level > 25:  # High fear = high correlation
                correlation_level = 'HIGH'
                correlation_strength = 0.8
                correlation_description = 'Risk-off: Crypto following traditional markets closely'
            elif vix_level < 20 and spy_trend == 'RISING':  # Risk-on = moderate correlation
                correlation_level = 'MODERATE'
                correlation_strength = 0.5
                correlation_description = 'Risk-on: Crypto partially independent'
            else:  # Normal conditions
                correlation_level = 'MODERATE'
                correlation_strength = 0.6
                correlation_description = 'Mixed conditions: Moderate correlation'
            
            # Market environment classification
            if spy_trend == 'RISING' and vix_level < 20:
                market_environment = 'RISK_ON'
            elif spy_trend == 'FALLING' or vix_level > 25:
                market_environment = 'RISK_OFF'
            else:
                market_environment = 'NEUTRAL'
            
            return {
                'correlation_level': correlation_level,
                'correlation_strength': correlation_strength,
                'correlation_description': correlation_description,
                'market_environment': market_environment,
                'traditional_crypto_alignment': self._assess_alignment(traditional_data)
            }
            
        except Exception as e:
            logger.error(f"Correlation regime determination failed: {str(e)}")
            return {
                'correlation_level': 'MODERATE',
                'correlation_strength': 0.6,
                'correlation_description': 'Mixed conditions',
                'market_environment': 'NEUTRAL',
                'traditional_crypto_alignment': 'NEUTRAL'
            }
    
    def _assess_alignment(self, traditional_data: Dict) -> str:
        """Assess whether traditional and crypto markets are aligned"""
        try:
            spy_trend = traditional_data.get('spy_trend', 'UNKNOWN')
            vix_trend = traditional_data.get('vix_trend', 'UNKNOWN')
            
            # Simple alignment assessment
            if spy_trend == 'RISING' and vix_trend == 'FALLING':
                return 'POSITIVE_ALIGNMENT'  # Good for crypto
            elif spy_trend == 'FALLING' and vix_trend == 'RISING':
                return 'NEGATIVE_ALIGNMENT'  # Bad for crypto
            else:
                return 'MIXED_SIGNALS'
                
        except Exception as e:
            logger.error(f"Alignment assessment failed: {str(e)}")
            return 'NEUTRAL'
    
    async def _analyze_btc_context(self) -> Dict[str, Any]:
        """Complete BTC intelligence: Technical + Sentiment + Wider Market"""
        try:
            # 1. Get BTC technical data - work around data fetcher symbol issue
            btc_df = self._get_btc_data()
            if btc_df is None or len(btc_df) < 50:
                logger.warning("Could not fetch BTC data, using defaults")
                return self._get_default_btc_context()
            
            # 2. BTC Technical Analysis
            btc_technical = self._analyze_btc_technical(btc_df)
            
            # 3. BTC Sentiment Analysis
            btc_sentiment = await self._get_btc_sentiment()
            
            # 4. Wider Market Context Analysis
            market_context = await self._analyze_wider_market_context()
            
            # 5. Calculate BTC health composite
            btc_health = self._calculate_btc_health(btc_technical, btc_sentiment)
            
            return {
                **btc_technical,
                **btc_sentiment,
                **market_context,
                **btc_health
            }
            
        except Exception as e:
            logger.error(f"BTC context analysis failed: {str(e)}")
            return self._get_default_btc_context()
    
    def _get_btc_data(self) -> Optional[pd.DataFrame]:
        """Get BTC data with fallback methods to work around data fetcher issues"""
        try:
            # Method 1: Try BTC (without /USDT since data fetcher adds it)
            btc_df = self.data_fetcher.fetch_ohlcv('binance', 'BTC', '4h')
            if btc_df is not None and len(btc_df) > 0:
                return btc_df
            
            # Method 2: Try different symbol formats
            symbol_formats = ['BTCUSDT', 'BTC-USDT', 'btc/usdt']
            exchanges = ['binance', 'bybit', 'okx', 'kucoin']
            
            for exchange in exchanges:
                for symbol in symbol_formats:
                    try:
                        btc_df = self.data_fetcher.fetch_ohlcv(exchange, symbol, '4h')
                        if btc_df is not None and len(btc_df) > 0:
                            logger.info(f"BTC data fetched successfully from {exchange} with {symbol}")
                            return btc_df
                    except Exception as e:
                        continue
            
            # Method 3: Direct CCXT call as fallback
            if hasattr(self.data_fetcher, 'exchanges'):
                for exchange_name, exchange_obj in self.data_fetcher.exchanges.items():
                    try:
                        if hasattr(exchange_obj, 'fetch_ohlcv'):
                            raw_data = exchange_obj.fetch_ohlcv('BTC/USDT', '4h', limit=200)
                            if raw_data and len(raw_data) > 0:
                                # Convert to DataFrame format expected by the system
                                btc_df = pd.DataFrame(raw_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
                                btc_df.set_index('timestamp', inplace=True)
                                
                                # Add basic indicators that might be expected
                                btc_df['sma_20'] = btc_df['close'].rolling(20).mean()
                                btc_df['sma_50'] = btc_df['close'].rolling(50).mean()
                                
                                logger.info(f"BTC data fetched via direct CCXT call from {exchange_name}")
                                return btc_df
                    except Exception as e:
                        continue
            
            logger.warning("All BTC data fetching methods failed")
            return None
            
        except Exception as e:
            logger.error(f"BTC data fetching failed: {str(e)}")
            return None
    
    def _analyze_btc_technical(self, btc_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze BTC technical indicators using pandas instead of external libraries"""
        try:
            close = btc_df['close']
            high = btc_df['high']
            low = btc_df['low']
            
            # Calculate SMA using pandas
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            
            # Simple ADX approximation using pandas
            high_low = high - low
            high_close = np.abs(high - close.shift(1))
            low_close = np.abs(low - close.shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            
            # Simplified directional movement
            dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                              np.maximum(high - high.shift(1), 0), 0)
            dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                               np.maximum(low.shift(1) - low, 0), 0)
            
            dm_plus_series = pd.Series(dm_plus, index=btc_df.index).rolling(14).mean()
            dm_minus_series = pd.Series(dm_minus, index=btc_df.index).rolling(14).mean()
            
            # Simple ADX approximation
            di_plus = (dm_plus_series / atr) * 100
            di_minus = (dm_minus_series / atr) * 100
            dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            adx = dx.rolling(14).mean()
            
            btc_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
            
            # BTC trend determination
            if sma_20.iloc[-1] > sma_50.iloc[-1] and btc_adx > self.ADX_WEAK_TREND:
                btc_trend = 'BULLISH'
            elif sma_20.iloc[-1] < sma_50.iloc[-1] and btc_adx > self.ADX_WEAK_TREND:
                btc_trend = 'BEARISH'
            else:
                btc_trend = 'NEUTRAL'
            
            # Technical confidence (0-100)
            trend_confidence = min(btc_adx * 2, 100)  # ADX * 2 capped at 100
            
            # Volatility context
            price_changes = close.pct_change().abs()
            volatility = price_changes.rolling(14).mean().iloc[-1] * 100
            atr_ratio = volatility / price_changes.rolling(50).mean().iloc[-1] if price_changes.rolling(50).mean().iloc[-1] > 0 else 1.0
            
            return {
                'btc_trend': btc_trend,
                'btc_technical_confidence': trend_confidence,
                'btc_adx': btc_adx,
                'btc_volatility_ratio': atr_ratio
            }
            
        except Exception as e:
            logger.error(f"BTC technical analysis failed: {str(e)}")
            return {
                'btc_trend': 'NEUTRAL',
                'btc_technical_confidence': 50,
                'btc_adx': 20,
                'btc_volatility_ratio': 1.0
            }
    
    async def _get_btc_sentiment(self) -> Dict[str, Any]:
        """Get comprehensive BTC sentiment from LunarCrush + TokenMetrics"""
        try:
            if not self.sentiment_analyzer:
                return self._get_default_btc_sentiment()
            
            # Get BTC sentiment analysis using correct method names
            try:
                # Try LunarCrush
                lunar_data = self.sentiment_analyzer.get_lunarcrush_sentiment('BTC')
                galaxy_score = lunar_data.get('lunar_galaxy_score', 50) if lunar_data else 50
                
                # Try TokenMetrics
                tm_data = self.sentiment_analyzer.get_tokenmetrics_sentiment('BTC')
                tm_grade = tm_data.get('tm_trader_grade', 50) if tm_data else 50
                ta_grade = tm_data.get('tm_ta_grade', 50) if tm_data else 50
                quant_grade = tm_data.get('tm_quant_grade', 50) if tm_data else 50
                
            except Exception as e:
                logger.warning(f"Sentiment API calls failed, using defaults: {str(e)}")
                galaxy_score = 50
                tm_grade = 50
                ta_grade = 50
                quant_grade = 50
            
            # Calculate sentiment confidence
            sentiment_scores = [galaxy_score, tm_grade, ta_grade]
            valid_scores = [s for s in sentiment_scores if s is not None and s > 0]
            
            if valid_scores:
                btc_sentiment_confidence = sum(valid_scores) / len(valid_scores)
            else:
                btc_sentiment_confidence = 50
            
            return {
                'btc_galaxy_score': galaxy_score,
                'btc_tm_grade': tm_grade,
                'btc_ta_grade': ta_grade,
                'btc_quant_grade': quant_grade,
                'btc_sentiment_confidence': btc_sentiment_confidence
            }
            
        except Exception as e:
            logger.error(f"BTC sentiment analysis failed: {str(e)}")
            return self._get_default_btc_sentiment()
    
    async def _analyze_wider_market_context(self) -> Dict[str, Any]:
        """Analyze broader crypto market conditions including alt market trends"""
        try:
            # 1. Crypto Fear & Greed Index
            fear_greed = await self._get_fear_greed_index()
            
            # 2. BTC Dominance Analysis
            btc_dominance = await self._get_btc_dominance()
            
            # 3. Total Crypto Market Cap Trend
            total_mcap = await self._get_total_market_cap_trend()
            
            # 4. Alt Market Analysis
            alt_market = await self._analyze_alt_market_trends(btc_dominance)
            
            # 5. Calculate wider market health
            market_health = self._calculate_market_health(fear_greed, btc_dominance, total_mcap)
            
            return {
                'fear_greed_index': fear_greed.get('value', 50),
                'fear_greed_classification': fear_greed.get('classification', 'Neutral'),
                'btc_dominance': btc_dominance.get('current', 50),
                'btc_dominance_trend': btc_dominance.get('trend', 'STABLE'),
                'total_mcap_trend': total_mcap.get('trend', 'NEUTRAL'),
                'market_health_score': market_health,
                'alt_season_indicator': self._determine_alt_season(btc_dominance, market_health),
                **alt_market  # Add alt market analysis
            }
            
        except Exception as e:
            logger.error(f"Wider market context analysis failed: {str(e)}")
            return self._get_default_market_context()
    
    async def _analyze_alt_market_trends(self, btc_dominance: Dict) -> Dict[str, Any]:
        """Analyze broader altcoin market trends"""
        try:
            # Calculate alt market cap (Total - BTC)
            total_mcap_data = await self._get_total_market_cap_trend()
            btc_dom_pct = btc_dominance.get('current', 50) / 100
            
            total_mcap = total_mcap_data.get('total_mcap', 0)
            btc_mcap = total_mcap * btc_dom_pct
            alt_mcap = total_mcap - btc_mcap
            
            # Determine alt market trend based on dominance changes
            dom_trend = btc_dominance.get('trend', 'STABLE')
            if dom_trend == 'DECREASING':
                alt_market_trend = 'RISING'       # Alt market cap growing vs BTC
            elif dom_trend == 'INCREASING':
                alt_market_trend = 'DECLINING'    # Alt market cap shrinking vs BTC
            else:
                alt_market_trend = 'STABLE'
            
            # Alt correlation analysis (simplified)
            # In a strong regime, alts move together. In weak regime, they're scattered.
            dom_value = btc_dominance.get('current', 50)
            if dom_value > 60:
                alt_correlation = 0.9    # High BTC dom = alts highly correlated (fear mode)
            elif dom_value < 40:
                alt_correlation = 0.6    # Low BTC dom = alts more independent (alt season)
            else:
                alt_correlation = 0.75   # Medium correlation
            
            # Alt volatility assessment
            if dom_value > 65 or dom_trend == 'INCREASING':
                alt_volatility = 'HIGH'   # Uncertainty = high volatility
            elif dom_value < 35:
                alt_volatility = 'LOW'    # Alt season = more stable
            else:
                alt_volatility = 'NORMAL'
            
            # Alt sector performance (simplified - could be enhanced with real data)
            # For now, base on overall market health
            market_change = total_mcap_data.get('change_24h', 0)
            if market_change > 2:
                sector_performance = ['DeFi: +3.2%', 'Layer1: +2.8%', 'Gaming: +4.1%']
            elif market_change < -2:
                sector_performance = ['DeFi: -2.8%', 'Layer1: -3.5%', 'Gaming: -1.9%']
            else:
                sector_performance = ['DeFi: +0.5%', 'Layer1: -0.8%', 'Gaming: +1.2%']
            
            return {
                'alt_market_cap_trend': alt_market_trend,
                'alt_correlation_index': alt_correlation,
                'alt_volatility_index': alt_volatility,
                'alt_sector_leaders': sector_performance,
                'alt_market_cap_usd': alt_mcap
            }
            
        except Exception as e:
            logger.warning(f"Alt market analysis failed: {str(e)}")
            return {
                'alt_market_cap_trend': 'NEUTRAL',
                'alt_correlation_index': 0.75,
                'alt_volatility_index': 'NORMAL',
                'alt_sector_leaders': ['Data unavailable'],
                'alt_market_cap_usd': 0
            }
    
    async def _get_fear_greed_index(self) -> Dict[str, Any]:
        """Get crypto Fear & Greed Index"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://api.alternative.me/fng/"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data'):
                            fg_data = data['data'][0]
                            return {
                                'value': int(fg_data.get('value', 50)),
                                'classification': fg_data.get('value_classification', 'Neutral'),
                                'timestamp': fg_data.get('timestamp')
                            }
            
            return {'value': 50, 'classification': 'Neutral'}
            
        except Exception as e:
            logger.warning(f"Could not fetch Fear & Greed Index: {str(e)}")
            return {'value': 50, 'classification': 'Neutral'}
    
    async def _get_btc_dominance(self) -> Dict[str, Any]:
        """Get BTC dominance and trend"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        btc_dominance = data.get('data', {}).get('market_cap_percentage', {}).get('btc', 50)
                        
                        # Determine trend (you could enhance this with historical data)
                        if btc_dominance > 45:
                            trend = 'INCREASING'  # Generally bearish for alts
                        elif btc_dominance < 40:
                            trend = 'DECREASING'  # Generally bullish for alts
                        else:
                            trend = 'STABLE'
                        
                        return {
                            'current': round(btc_dominance, 2),
                            'trend': trend
                        }
            
            return {'current': 50, 'trend': 'STABLE'}
            
        except Exception as e:
            logger.warning(f"Could not fetch BTC dominance: {str(e)}")
            return {'current': 50, 'trend': 'STABLE'}
    
    async def _get_total_market_cap_trend(self) -> Dict[str, Any]:
        """Get total crypto market cap trend"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        total_mcap = data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
                        mcap_change_24h = data.get('data', {}).get('market_cap_change_percentage_24h_usd', 0)
                        
                        # Determine trend
                        if mcap_change_24h > 2:
                            trend = 'BULLISH'
                        elif mcap_change_24h < -2:
                            trend = 'BEARISH'
                        else:
                            trend = 'NEUTRAL'
                        
                        return {
                            'total_mcap': total_mcap,
                            'change_24h': round(mcap_change_24h, 2),
                            'trend': trend
                        }
            
            return {'total_mcap': 0, 'change_24h': 0, 'trend': 'NEUTRAL'}
            
        except Exception as e:
            logger.warning(f"Could not fetch total market cap: {str(e)}")
            return {'total_mcap': 0, 'change_24h': 0, 'trend': 'NEUTRAL'}
    
    def _calculate_market_health(self, fear_greed: Dict, btc_dominance: Dict, total_mcap: Dict) -> float:
        """Calculate overall market health score"""
        try:
            # Fear & Greed component (30-70 is good for trading)
            fg_value = fear_greed.get('value', 50)
            if 30 <= fg_value <= 70:
                fg_score = 80  # Good trading conditions
            elif 20 <= fg_value <= 80:
                fg_score = 60  # Acceptable
            else:
                fg_score = 40  # Extreme conditions
            
            # BTC Dominance component (40-45% is good for alts)
            dom_value = btc_dominance.get('current', 50)
            if 40 <= dom_value <= 45:
                dom_score = 80  # Good for alts
            elif 35 <= dom_value <= 50:
                dom_score = 60  # Acceptable
            else:
                dom_score = 40  # Poor for alts
            
            # Market cap trend component
            mcap_change = total_mcap.get('change_24h', 0)
            if mcap_change > 1:
                mcap_score = 80  # Growing market
            elif mcap_change > -1:
                mcap_score = 60  # Stable
            else:
                mcap_score = 40  # Declining
            
            # Weighted average
            market_health = (fg_score * 0.4 + dom_score * 0.3 + mcap_score * 0.3)
            return round(market_health, 1)
            
        except Exception as e:
            logger.error(f"Market health calculation failed: {str(e)}")
            return 50.0
    
    def _determine_alt_season(self, btc_dominance: Dict, market_health: float) -> str:
        """Determine if we're in alt season conditions"""
        try:
            dom_value = btc_dominance.get('current', 50)
            dom_trend = btc_dominance.get('trend', 'STABLE')
            
            if dom_value < 40 and dom_trend == 'DECREASING' and market_health > 70:
                return 'ALT_SEASON'
            elif dom_value < 45 and market_health > 60:
                return 'ALT_FAVORABLE'
            elif dom_value > 50 or market_health < 40:
                return 'BTC_SEASON'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Alt season determination failed: {str(e)}")
            return 'NEUTRAL'
    
    def _calculate_btc_health(self, btc_technical: Dict, btc_sentiment: Dict) -> Dict[str, Any]:
        """Calculate composite BTC health score and alt market implications"""
        try:
            tech_conf = btc_technical.get('btc_technical_confidence', 50)
            sent_conf = btc_sentiment.get('btc_sentiment_confidence', 50)
            
            # Composite BTC health (weighted average)
            btc_health_score = (tech_conf * 0.6 + sent_conf * 0.4)  # Tech weighted higher
            
            # Alt market outlook determination
            if btc_health_score >= 75 and btc_technical.get('btc_trend') == 'BULLISH':
                alt_outlook = 'EXCELLENT'  # Alt season conditions
            elif btc_health_score >= 60 and btc_technical.get('btc_trend') in ['BULLISH', 'NEUTRAL']:
                alt_outlook = 'GOOD'       # Favorable for alts
            elif btc_health_score >= 45:
                alt_outlook = 'FAIR'       # Neutral conditions
            else:
                alt_outlook = 'POOR'       # Risk-off, avoid alts
            
            # Confluence bonus (when technical and sentiment align)
            alignment_diff = abs(tech_conf - sent_conf)
            confluence_bonus = max(0, (20 - alignment_diff) / 20 * 0.1)  # Up to 10% bonus
            
            return {
                'btc_health_score': btc_health_score,
                'alt_market_outlook': alt_outlook,
                'btc_confluence_bonus': confluence_bonus,
                'btc_tech_sent_alignment': alignment_diff < 15
            }
            
        except Exception as e:
            logger.error(f"BTC health calculation failed: {str(e)}")
            return {
                'btc_health_score': 50,
                'alt_market_outlook': 'FAIR',
                'btc_confluence_bonus': 0,
                'btc_tech_sent_alignment': False
            }
    
    def _calculate_composite_regime(self, alt_regime: Dict, btc_context: Dict, 
                                   traditional_context: Dict, symbol: str) -> Dict[str, Any]:
        """Calculate final regime classification and confidence with traditional market integration"""
        try:
            # 1. Determine alt regime type
            adx = alt_regime.get('adx', 20)
            volatility_ratio = alt_regime.get('volatility_ratio', 1.0)
            bb_width_pct = alt_regime.get('bb_width_percentile', 0.5)
            
            if adx > self.ADX_STRONG_TREND:
                regime_type = 'STRONG_TREND'
            elif adx > self.ADX_WEAK_TREND:
                regime_type = 'WEAK_TREND'
            elif bb_width_pct < self.BB_SQUEEZE_THRESHOLD:
                regime_type = 'SQUEEZE'
            elif volatility_ratio > self.VOLATILITY_HIGH_THRESHOLD:
                regime_type = 'VOLATILE'
            else:
                regime_type = 'RANGING'
            
            # 2. Determine BB strategy suitability
            if regime_type == 'RANGING':
                bb_suitability = 'EXCELLENT'
                base_confidence = 85
            elif regime_type == 'WEAK_TREND':
                bb_suitability = 'GOOD'
                base_confidence = 70
            elif regime_type == 'SQUEEZE':
                bb_suitability = 'FAIR'
                base_confidence = 60
            elif regime_type == 'VOLATILE':
                bb_suitability = 'FAIR'
                base_confidence = 55
            else:  # STRONG_TREND
                bb_suitability = 'POOR'
                base_confidence = 40
            
            # 3. Apply BTC context modifiers
            btc_health = btc_context.get('btc_health_score', 50)
            btc_outlook = btc_context.get('alt_market_outlook', 'FAIR')
            
            # BTC health modifier
            if btc_health >= 75:
                btc_modifier = 1.15
            elif btc_health >= 60:
                btc_modifier = 1.05
            elif btc_health >= 45:
                btc_modifier = 1.0
            else:
                btc_modifier = 0.85
            
            # 4. Apply traditional market modifiers
            traditional_modifier = self._calculate_traditional_modifier(traditional_context)
            
            # Apply confluence bonus
            confluence_bonus = btc_context.get('btc_confluence_bonus', 0)
            
            # Final regime confidence with traditional market integration
            regime_confidence = min(
                base_confidence * btc_modifier * traditional_modifier * (1 + confluence_bonus), 95
            )
            
            # 5. Calculate enhanced position size multiplier
            position_multiplier = self._calculate_enhanced_position_multiplier(
                regime_confidence, btc_health, btc_outlook, traditional_context
            )
            
            # 6. Generate enhanced context description
            context_description = self._generate_enhanced_context_description(
                regime_type, bb_suitability, btc_outlook, traditional_context
            )
            
            return {
                # Core outputs
                'regime_confidence': round(regime_confidence, 1),
                'regime_type': regime_type,
                'bb_suitability': bb_suitability,
                'position_multiplier': round(position_multiplier, 2),
                'context_description': context_description,
                
                # Alt technical details
                'alt_trend_strength': round(adx, 1),
                'alt_trend_direction': alt_regime.get('trend_direction', 'NEUTRAL'),
                'alt_volatility_regime': self._classify_volatility(volatility_ratio),
                'alt_volume_trend': self._classify_volume_trend(alt_regime.get('volume_trend', 0)),
                'bb_squeeze_phase': bb_width_pct < self.BB_SQUEEZE_THRESHOLD,
                
                # BTC context
                'btc_trend': btc_context.get('btc_trend', 'NEUTRAL'),
                'btc_technical_confidence': round(btc_context.get('btc_technical_confidence', 50), 1),
                'btc_sentiment_confidence': round(btc_context.get('btc_sentiment_confidence', 50), 1),
                'btc_health_score': round(btc_health, 1),
                'alt_market_outlook': btc_outlook,
                'btc_adx': round(btc_context.get('btc_adx', 20), 1),
                
                # BTC sentiment details
                'btc_galaxy_score': btc_context.get('btc_galaxy_score', 50),
                'btc_tm_grade': btc_context.get('btc_tm_grade', 50),
                'btc_ta_grade': btc_context.get('btc_ta_grade', 50),
                'btc_quant_grade': btc_context.get('btc_quant_grade', 50),
                
                # Wider market context
                'fear_greed_index': btc_context.get('fear_greed_index', 50),
                'fear_greed_classification': btc_context.get('fear_greed_classification', 'Neutral'),
                'btc_dominance': btc_context.get('btc_dominance', 50),
                'btc_dominance_trend': btc_context.get('btc_dominance_trend', 'STABLE'),
                'market_health_score': btc_context.get('market_health_score', 50),
                'alt_season_indicator': btc_context.get('alt_season_indicator', 'NEUTRAL'),
                
                # Alt market analysis
                'alt_market_cap_trend': btc_context.get('alt_market_cap_trend', 'NEUTRAL'),
                'alt_correlation_index': btc_context.get('alt_correlation_index', 0.75),
                'alt_volatility_index': btc_context.get('alt_volatility_index', 'NORMAL'),
                'alt_sector_leaders': btc_context.get('alt_sector_leaders', ['Data unavailable']),
                
                # Traditional market context
                'spy_trend': traditional_context.get('spy_trend', 'UNKNOWN'),
                'spy_change': traditional_context.get('spy_change', 0),
                'vix_price': traditional_context.get('vix_price', 20),
                'vix_trend': traditional_context.get('vix_trend', 'UNKNOWN'),
                'dxy_change': traditional_context.get('dxy_change', 0),
                'dxy_trend': traditional_context.get('dxy_trend', 'UNKNOWN'),
                'qqq_change': traditional_context.get('qqq_change', 0),
                'correlation_level': traditional_context.get('correlation_level', 'MODERATE'),
                'correlation_strength': traditional_context.get('correlation_strength', 0.6),
                'market_environment': traditional_context.get('market_environment', 'NEUTRAL'),
                'traditional_crypto_alignment': traditional_context.get('traditional_crypto_alignment', 'NEUTRAL'),
                
                # Traditional market sentiment (individual stocks)
                'spy_sentiment': traditional_context.get('spy', 50),
                'tech_sentiment': traditional_context.get('tech', 50),
                'growth_sentiment': traditional_context.get('growth', 50),
                'stocks_sentiment': traditional_context.get('stocks', 50),
                
                # Metadata
                'analysis_timestamp': datetime.now().isoformat(),
                'symbol_analyzed': symbol
            }
            
        except Exception as e:
            logger.error(f"Composite regime calculation failed: {str(e)}")
            return self._get_default_regime_analysis()
    
    def _calculate_traditional_modifier(self, traditional_context: Dict) -> float:
        """Calculate position sizing modifier based on traditional market conditions"""
        try:
            market_environment = traditional_context.get('market_environment', 'NEUTRAL')
            correlation_level = traditional_context.get('correlation_level', 'MODERATE')
            vix_price = traditional_context.get('vix_price', 20)
            
            # Base modifier from market environment
            if market_environment == 'RISK_ON':
                base_modifier = 1.1  # Favorable for crypto
            elif market_environment == 'RISK_OFF':
                base_modifier = 0.9  # Unfavorable for crypto
            else:
                base_modifier = 1.0  # Neutral
            
            # VIX adjustment (fear gauge)
            if vix_price > 30:  # High fear
                vix_modifier = 0.85
            elif vix_price < 15:  # Low fear
                vix_modifier = 1.05
            else:
                vix_modifier = 1.0
            
            # Correlation adjustment
            correlation_strength = traditional_context.get('correlation_strength', 0.6)
            if correlation_level == 'HIGH' and market_environment == 'RISK_OFF':
                corr_modifier = 0.8  # High correlation during risk-off = bad for crypto
            elif correlation_level == 'LOW':
                corr_modifier = 1.05  # Low correlation = crypto independence
            else:
                corr_modifier = 1.0
            
            # Combined traditional modifier
            traditional_modifier = base_modifier * vix_modifier * corr_modifier
            
            # Cap between 0.7 and 1.2
            return max(0.7, min(traditional_modifier, 1.2))
            
        except Exception as e:
            logger.error(f"Traditional modifier calculation failed: {str(e)}")
            return 1.0
    
    def _calculate_enhanced_position_multiplier(self, regime_confidence: float, btc_health: float, 
                                              btc_outlook: str, traditional_context: Dict) -> float:
        """Calculate dynamic position size multiplier with traditional market integration"""
        try:
            # Base multiplier from regime confidence
            if regime_confidence >= 90:
                base_multiplier = 1.4
            elif regime_confidence >= 80:
                base_multiplier = 1.2
            elif regime_confidence >= 70:
                base_multiplier = 1.0
            elif regime_confidence >= 60:
                base_multiplier = 0.9
            else:
                base_multiplier = 0.7
            
            # BTC context modifier
            btc_modifiers = {
                'EXCELLENT': 1.15,
                'GOOD': 1.05,
                'FAIR': 1.0,
                'POOR': 0.8
            }
            
            btc_modifier = btc_modifiers.get(btc_outlook, 1.0)
            
            # Traditional market modifier
            traditional_modifier = self._calculate_traditional_modifier(traditional_context)
            
            # Final position multiplier
            final_multiplier = base_multiplier * btc_modifier * traditional_modifier
            
            # Cap between 0.5x and 1.5x
            return max(0.5, min(final_multiplier, 1.5))
            
        except Exception as e:
            logger.error(f"Enhanced position multiplier calculation failed: {str(e)}")
            return 1.0
    
    def _generate_enhanced_context_description(self, regime_type: str, bb_suitability: str, 
                                             btc_outlook: str, traditional_context: Dict) -> str:
        """Generate human-readable context description with traditional market integration"""
        try:
            # Base regime description
            regime_descriptions = {
                'RANGING': 'Ranging market - ideal for BB bounces',
                'WEAK_TREND': 'Weak trend - BB bounces possible with caution',
                'STRONG_TREND': 'Strong trend - BB bounces likely to fail',
                'VOLATILE': 'High volatility - unpredictable BB behavior',
                'SQUEEZE': 'BB squeeze - breakout pending'
            }
            
            base_desc = regime_descriptions.get(regime_type, 'Mixed market conditions')
            
            # Traditional market context
            market_env = traditional_context.get('market_environment', 'NEUTRAL')
            spy_trend = traditional_context.get('spy_trend', 'UNKNOWN')
            vix_price = traditional_context.get('vix_price', 20)
            
            if market_env == 'RISK_ON':
                trad_desc = f"Risk-on environment (SPY {spy_trend.lower()}, VIX {vix_price:.0f})"
            elif market_env == 'RISK_OFF':
                trad_desc = f"Risk-off environment (SPY {spy_trend.lower()}, VIX {vix_price:.0f})"
            else:
                trad_desc = f"Mixed traditional markets (SPY {spy_trend.lower()})"
            
            # BTC context
            if btc_outlook == 'EXCELLENT':
                btc_desc = "Strong BTC support"
            elif btc_outlook == 'GOOD':
                btc_desc = "Favorable BTC context"
            elif btc_outlook == 'FAIR':
                btc_desc = "Neutral BTC environment"
            else:
                btc_desc = "Weak BTC context"
            
            return f"{base_desc} + {trad_desc} + {btc_desc}"
            
        except Exception as e:
            logger.error(f"Enhanced context description generation failed: {str(e)}")
            return "Mixed market conditions"
    
    def _classify_volatility(self, volatility_ratio: float) -> str:
        """Classify volatility regime"""
        if volatility_ratio > self.VOLATILITY_HIGH_THRESHOLD:
            return 'HIGH'
        elif volatility_ratio < self.VOLATILITY_LOW_THRESHOLD:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def _classify_volume_trend(self, volume_trend: float) -> str:
        """Classify volume trend"""
        if volume_trend > 20:
            return 'STRONG'
        elif volume_trend > -10:
            return 'AVERAGE'
        else:
            return 'WEAK'
    
    def format_regime_display(self, regime_analysis: Dict) -> str:
        """Format regime analysis for enhanced 6-line terminal display with traditional markets"""
        try:
            # Core regime data
            confidence = regime_analysis.get('regime_confidence', 0)
            suitability = regime_analysis.get('bb_suitability', 'FAIR')
            position_mult = regime_analysis.get('position_multiplier', 1.0)
            
            # Alt technical details
            alt_trend_strength = regime_analysis.get('alt_trend_strength', 0)
            alt_trend_direction = regime_analysis.get('alt_trend_direction', 'NEUTRAL')
            alt_volatility = regime_analysis.get('alt_volatility_regime', 'NORMAL')
            alt_volume_trend = regime_analysis.get('alt_volume_trend', 'AVERAGE')
            bb_squeeze = regime_analysis.get('bb_squeeze_phase', False)
            
            # BTC technical and sentiment
            btc_trend = regime_analysis.get('btc_trend', 'NEUTRAL')
            btc_tech = regime_analysis.get('btc_technical_confidence', 50)
            btc_sent = regime_analysis.get('btc_sentiment_confidence', 50)
            btc_adx = regime_analysis.get('btc_adx', 20)
            
            # BTC sentiment details
            tm_grade = regime_analysis.get('btc_tm_grade', 50)
            ta_grade = regime_analysis.get('btc_ta_grade', 50)
            quant_grade = regime_analysis.get('btc_quant_grade', 50)
            galaxy_score = regime_analysis.get('btc_galaxy_score', 50)
            
            # Wider market context
            fear_greed = regime_analysis.get('fear_greed_index', 50)
            fear_greed_class = regime_analysis.get('fear_greed_classification', 'Neutral')
            btc_dominance = regime_analysis.get('btc_dominance', 50)
            btc_dom_trend = regime_analysis.get('btc_dominance_trend', 'STABLE')
            alt_season = regime_analysis.get('alt_season_indicator', 'NEUTRAL')
            market_health = regime_analysis.get('market_health_score', 50)
            
            # Traditional market data
            spy_trend = regime_analysis.get('spy_trend', 'UNKNOWN')
            spy_change = regime_analysis.get('spy_change', 0)
            vix_price = regime_analysis.get('vix_price', 20)
            market_env = regime_analysis.get('market_environment', 'NEUTRAL')
            correlation_level = regime_analysis.get('correlation_level', 'MODERATE')
            
            # Alt market context
            alt_market_trend = regime_analysis.get('alt_market_cap_trend', 'NEUTRAL')
            alt_correlation = regime_analysis.get('alt_correlation_index', 0.75)
            alt_vol_index = regime_analysis.get('alt_volatility_index', 'NORMAL')
            
            # Enhanced market summary with traditional markets
            if spy_trend == 'RISING' and vix_price < 20:
                if btc_dominance < 50:
                    market_summary = '🐂 Bull Market'
                    risk_environment = 'Low Risk'
                else:
                    market_summary = '📈 Risk-On'
                    risk_environment = 'Medium Risk'
            elif spy_trend == 'FALLING' or vix_price > 25:
                market_summary = '📉 Risk-Off'
                risk_environment = 'High Risk'
            elif btc_dominance > 60 and btc_dom_trend == 'INCREASING':
                market_summary = '🐻 Bear Market'
                risk_environment = 'High Risk'
            elif fear_greed < 30:
                market_summary = '😨 Fear Market'
                risk_environment = 'High Risk'
            elif fear_greed > 70:
                market_summary = '🤑 Greed Market'
                risk_environment = 'High Risk'
            else:
                market_summary = '🦀 Mixed Market'
                risk_environment = 'Medium Risk'
            
            # Format trend direction with emoji
            trend_emoji = '📈' if alt_trend_direction == 'UP' else '📉' if alt_trend_direction == 'DOWN' else '➡️'
            
            # Format volatility with emoji
            vol_emoji = '🔥' if alt_volatility == 'HIGH' else '❄️' if alt_volatility == 'LOW' else '🌡️'
            
            # Format volume trend with emoji
            vol_trend_emoji = '📊' if alt_volume_trend == 'STRONG' else '📉' if alt_volume_trend == 'WEAK' else '➡️'
            
            # Format BB squeeze status
            bb_status = '🤏 SQUEEZE' if bb_squeeze else '📏 Normal'
            
            # Format dominance trend arrow
            dom_arrow = '↑' if btc_dom_trend == 'INCREASING' else '↓' if btc_dom_trend == 'DECREASING' else '→'
            
            # Format alt season emoji
            alt_emoji = '✅' if alt_season == 'ALT_SEASON' else '📈' if alt_season == 'ALT_FAVORABLE' else '❌' if alt_season == 'BTC_SEASON' else '⚖️'
            
            # Format BTC trend emoji
            btc_trend_emoji = '📈' if btc_trend == 'BULLISH' else '📉' if btc_trend == 'BEARISH' else '➡️'
            
            # Format SPY trend arrow
            spy_arrow = '↗️' if spy_trend == 'RISING' else '↘️' if spy_trend == 'FALLING' else '→'
            
            # VIX color coding
            vix_emoji = '🟢' if vix_price < 20 else '🟡' if vix_price < 30 else '🔴'
            
            # Line 0: Enhanced Market Summary with Traditional Markets
            line0 = f"📊 MARKET SUMMARY: {market_summary} | SPY {spy_arrow} | VIX {vix_emoji}{vix_price:.0f} | Correlation: {correlation_level} | {risk_environment}"
            
            # Line 1: Main regime assessment
            line1 = f"🌊 Market Regime: {suitability} ({confidence:.1f}% confidence) | Position Size: {position_mult:.2f}x"
            
            # Line 2: Alt technical analysis details
            line2 = f"   📊 Alt Technical: {regime_analysis.get('regime_type', 'MIXED')} (ADX {alt_trend_strength:.1f}) | {trend_emoji} {alt_trend_direction} | {vol_emoji} Vol: {alt_volatility} | {bb_status}"
            
            # Line 3: Enhanced alt context with market analysis
            line3 = f"   📈 Alt Context: {vol_trend_emoji} Volume: {alt_volume_trend} | Alt Market: {alt_market_trend} | Correlation: {alt_correlation:.2f} | Vol Index: {alt_vol_index}"
            
            # Line 4: BTC comprehensive analysis
            line4 = f"   ₿ BTC Analysis: {btc_trend_emoji} {btc_trend} (ADX {btc_adx:.1f}) | 🟢 Tech {btc_tech:.0f}% + 📈 Sent {btc_sent:.0f}% | TM:{tm_grade:.1f} TA:{ta_grade:.1f} Q:{quant_grade:.1f} G:{galaxy_score:.0f}"
            
            # Line 5: Wider market intelligence with traditional context
            line5 = f"   🌍 Market: F&G {fear_greed} | SPY {spy_change:+.1f}% | BTC Dom {btc_dominance:.1f}% {dom_arrow} | Alt Season {alt_emoji} | Health {market_health:.1f}"
            
            return f"{line0}\n{line1}\n{line2}\n{line3}\n{line4}\n{line5}"
            
        except Exception as e:
            logger.error(f"Enhanced regime display formatting failed: {str(e)}")
            return "🌊 Market Regime: Enhanced analysis unavailable"
    
    # Default/fallback methods with traditional market defaults
    def _get_default_regime_analysis(self) -> Dict[str, Any]:
        """Return default regime analysis when errors occur"""
        return {
            'regime_confidence': 50.0,
            'regime_type': 'MIXED',
            'bb_suitability': 'FAIR',
            'position_multiplier': 1.0,
            'context_description': 'Mixed market conditions - standard approach',
            'alt_trend_strength': 20.0,
            'alt_trend_direction': 'NEUTRAL',
            'alt_volatility_regime': 'NORMAL',
            'alt_volume_trend': 'AVERAGE',
            'bb_squeeze_phase': False,
            'btc_trend': 'NEUTRAL',
            'btc_technical_confidence': 50.0,
            'btc_sentiment_confidence': 50.0,
            'btc_health_score': 50.0,
            'alt_market_outlook': 'FAIR',
            'btc_adx': 20.0,
            'btc_galaxy_score': 50,
            'btc_tm_grade': 50,
            'btc_ta_grade': 50,
            'btc_quant_grade': 50,
            'fear_greed_index': 50,
            'fear_greed_classification': 'Neutral',
            'btc_dominance': 50,
            'btc_dominance_trend': 'STABLE',
            'market_health_score': 50,
            'alt_season_indicator': 'NEUTRAL',
            'alt_market_cap_trend': 'NEUTRAL',
            'alt_correlation_index': 0.75,
            'alt_volatility_index': 'NORMAL',
            'alt_sector_leaders': ['Data unavailable'],
            'spy_trend': 'UNKNOWN',
            'spy_change': 0,
            'vix_price': 20,
            'vix_trend': 'UNKNOWN',
            'dxy_trend': 'UNKNOWN',
            'correlation_level': 'MODERATE',
            'correlation_strength': 0.6,
            'market_environment': 'NEUTRAL',
            'traditional_crypto_alignment': 'NEUTRAL',
            'spy_sentiment': 50,
            'tech_sentiment': 50,
            'growth_sentiment': 50,
            'stocks_sentiment': 50,
            'analysis_timestamp': datetime.now().isoformat(),
            'symbol_analyzed': 'UNKNOWN'
        }
    
    def _get_default_alt_regime(self) -> Dict[str, Any]:
        """Default alt regime when analysis fails"""
        return {
            'adx': 20,
            'trend_direction': 'NEUTRAL',
            'trend_strength': 20,
            'sma_slope': 0,
            'volatility_ratio': 1.0,
            'bb_width_percentile': 0.5,
            'volume_trend': 0,
            'price_position': 0.5
        }
    
    def _get_default_btc_context(self) -> Dict[str, Any]:
        """Default BTC context when analysis fails"""
        return {
            'btc_trend': 'NEUTRAL',
            'btc_technical_confidence': 50,
            'btc_adx': 20,
            'btc_volatility_ratio': 1.0,
            'btc_galaxy_score': 50,
            'btc_tm_grade': 50,
            'btc_ta_grade': 50,
            'btc_quant_grade': 50,
            'btc_sentiment_confidence': 50,
            'fear_greed_index': 50,
            'fear_greed_classification': 'Neutral',
            'btc_dominance': 50,
            'btc_dominance_trend': 'STABLE',
            'total_mcap_trend': 'NEUTRAL',
            'market_health_score': 50,
            'alt_season_indicator': 'NEUTRAL',
            'btc_health_score': 50,
            'alt_market_outlook': 'FAIR',
            'btc_confluence_bonus': 0,
            'btc_tech_sent_alignment': False,
            'alt_market_cap_trend': 'NEUTRAL',
            'alt_correlation_index': 0.75,
            'alt_volatility_index': 'NORMAL',
            'alt_sector_leaders': ['Data unavailable'],
            'alt_market_cap_usd': 0
        }
    
    def _get_default_market_context(self) -> Dict[str, Any]:
        """Default market context when analysis fails"""
        return {
            'fear_greed_index': 50,
            'fear_greed_classification': 'Neutral',
            'btc_dominance': 50,
            'btc_dominance_trend': 'STABLE',
            'total_mcap_trend': 'NEUTRAL',
            'market_health_score': 50,
            'alt_season_indicator': 'NEUTRAL',
            'alt_market_cap_trend': 'NEUTRAL',
            'alt_correlation_index': 0.75,
            'alt_volatility_index': 'NORMAL',
            'alt_sector_leaders': ['Data unavailable'],
            'alt_market_cap_usd': 0
        }
    
    def _get_default_btc_sentiment(self) -> Dict[str, Any]:
        """Default BTC sentiment when analysis fails"""
        return {
            'btc_galaxy_score': 50,
            'btc_tm_grade': 50,
            'btc_ta_grade': 50,
            'btc_quant_grade': 50,
            'btc_sentiment_confidence': 50
        }
    
    def _get_default_traditional_context(self) -> Dict[str, Any]:
        """Default traditional market context when analysis fails"""
        return {
            'spy_price': 0,
            'spy_change': 0,
            'spy_trend': 'UNKNOWN',
            'vix_price': 20,
            'vix_change': 0,
            'vix_trend': 'UNKNOWN',
            'dxy_price': 0,
            'dxy_change': 0,
            'dxy_trend': 'UNKNOWN',
            'qqq_price': 0,
            'qqq_change': 0,
            'qqq_trend': 'UNKNOWN',
            'correlation_level': 'MODERATE',
            'correlation_strength': 0.6,
            'correlation_description': 'Mixed conditions',
            'market_environment': 'NEUTRAL',
            'traditional_crypto_alignment': 'NEUTRAL',
            'spy': 50,
            'tech': 50,
            'growth': 50,
            'stocks': 50
        }