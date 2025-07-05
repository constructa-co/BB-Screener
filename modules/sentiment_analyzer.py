# sentiment_analyzer.py - Sentiment Analysis Module
import requests
import pandas as pd
import pandas_ta as ta
import time
import logging
from datetime import datetime
from typing import Dict, Any, List
from config import *

logger = logging.getLogger(__name__)
sentiment_logger = logging.getLogger('sentiment')

class SentimentAnalyzer:
    """Sentiment analysis for top trades only - NO impact on BB logic"""
    
    def __init__(self):
        self.lunar_api_key = LUNAR_API_KEY
        self.tokenmetrics_api_key = TOKENMETRICS_API_KEY
        self.sentiment_cache = {}
        self.rate_limit_delay = SCANNER_CONFIG["rate_limit_delay"]
        self.symbol_id_cache = {}  # Cache for symbol to ID mapping
        
    def get_lunarcrush_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get LunarCrush sentiment data with corrected endpoint"""
        try:
            # Check cache first
            cache_key = f"lunar_{symbol}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # APPROACH 1: Try using symbol directly as query parameter
            url = "https://lunarcrush.com/api4/public/coins"
            headers = {
                "Authorization": f"Bearer {self.lunar_api_key}",
                "Content-Type": "application/json"
            }
            params = {
                "symbol": symbol
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            sentiment_logger.info(f"LunarCrush API call for {symbol}: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                sentiment_logger.info(f"LunarCrush response structure for {symbol}: {list(data.keys()) if data else 'Empty response'}")
                
                if data and isinstance(data, dict):
                    # Handle the response structure from the working API
                    coin_data = data.get('data', data)
                    
                    # If data is a list (multiple coins), find our symbol
                    if isinstance(coin_data, list):
                        for coin in coin_data:
                            if coin.get('symbol') == symbol:
                                coin_data = coin
                                break
                        else:
                            sentiment_logger.warning(f"Symbol {symbol} not found in LunarCrush response")
                            sentiment_data = self._get_empty_lunar_data()
                            self.sentiment_cache[cache_key] = sentiment_data
                            return sentiment_data
                    
                    # Extract sentiment metrics using the exact field names from API docs
                    sentiment_data = {
                        'lunar_sentiment_score': coin_data.get('galaxy_score', 0),  # Using galaxy_score as main sentiment
                        'lunar_social_score': coin_data.get('social_score', 0),
                        'lunar_galaxy_score': coin_data.get('galaxy_score', 0),
                        'lunar_alt_rank': coin_data.get('alt_rank', 999),
                        'lunar_sentiment_rating': self._get_sentiment_rating(coin_data.get('galaxy_score', 0)),
                        'lunar_data_available': True
                    }
                    
                    sentiment_logger.info(f"LunarCrush extracted for {symbol}: Galaxy={sentiment_data['lunar_galaxy_score']}, AltRank={sentiment_data['lunar_alt_rank']}, Social={sentiment_data['lunar_social_score']}")
                else:
                    sentiment_logger.warning(f"LunarCrush: Unexpected response format for {symbol}")
                    sentiment_data = self._get_empty_lunar_data()
                    
            elif response.status_code == 404:
                # APPROACH 2: If symbol query fails, try getting ID from coins list first
                sentiment_logger.info(f"Symbol query failed for {symbol}, trying ID lookup approach...")
                sentiment_data = self._get_lunarcrush_by_id_lookup(symbol)
            else:
                sentiment_logger.warning(f"LunarCrush API error for {symbol}: {response.status_code} - {response.text[:200]}")
                sentiment_data = self._get_empty_lunar_data()
                
            # Cache the result
            self.sentiment_cache[cache_key] = sentiment_data
            return sentiment_data
            
        except Exception as e:
            sentiment_logger.error(f"Error fetching LunarCrush data for {symbol}: {e}")
            return self._get_empty_lunar_data()
    
    def _get_lunarcrush_by_id_lookup(self, symbol: str) -> Dict[str, Any]:
        """Fallback method: Get LunarCrush data by looking up symbol ID first"""
        try:
            # First, get the coins list to find the ID for our symbol
            if not hasattr(self, '_coins_list_cache'):
                url = "https://lunarcrush.com/api4/public/coins/list/v1"
                headers = {
                    "Authorization": f"Bearer {self.lunar_api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    list_data = response.json()
                    if list_data.get('data'):
                        self._coins_list_cache = list_data['data']
                        sentiment_logger.info(f"Cached {len(self._coins_list_cache)} coins from LunarCrush list")
                    else:
                        return self._get_empty_lunar_data()
                else:
                    sentiment_logger.warning(f"Failed to get coins list: {response.status_code}")
                    return self._get_empty_lunar_data()
            
            # Find our symbol in the cached list
            coin_id = None
            for coin in self._coins_list_cache:
                if coin.get('symbol') == symbol:
                    coin_id = coin.get('id')
                    break
            
            if not coin_id:
                sentiment_logger.warning(f"Symbol {symbol} not found in LunarCrush coins list")
                return self._get_empty_lunar_data()
            
            # Now get individual coin data using the ID
            url = f"https://lunarcrush.com/api4/public/coins/{coin_id}/v1"
            headers = {
                "Authorization": f"Bearer {self.lunar_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                coin_data = data.get('data', data)
                
                sentiment_data = {
                    'lunar_sentiment_score': coin_data.get('galaxy_score', 0),
                    'lunar_social_score': coin_data.get('social_score', 0),
                    'lunar_galaxy_score': coin_data.get('galaxy_score', 0),
                    'lunar_alt_rank': coin_data.get('alt_rank', 999),
                    'lunar_sentiment_rating': self._get_sentiment_rating(coin_data.get('galaxy_score', 0)),
                    'lunar_data_available': True
                }
                
                sentiment_logger.info(f"LunarCrush ID lookup successful for {symbol} (ID: {coin_id})")
                return sentiment_data
            else:
                sentiment_logger.warning(f"Failed to get coin data for {symbol} (ID: {coin_id}): {response.status_code}")
                return self._get_empty_lunar_data()
                
        except Exception as e:
            sentiment_logger.error(f"Error in ID lookup for {symbol}: {e}")
            return self._get_empty_lunar_data()

    def get_tokenmetrics_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get TokenMetrics trader grade for specific symbol - FIXED"""
        try:
            # Check cache first
            cache_key = f"tm_{symbol}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Query specific symbol directly
            url = "https://api.tokenmetrics.com/v2/trader-grades"
            headers = {
                "accept": "application/json",
                "x-api-key": self.tokenmetrics_api_key
            }
            params = {
                "symbol": symbol,  # Query this specific symbol
                "limit": 10,       # Limit results for this symbol
                "page": 1
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=60)
            sentiment_logger.info(f"TokenMetrics API call for {symbol}: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                sentiment_logger.info(f"TokenMetrics response for {symbol}: success={data.get('success')}, length={data.get('length', 0)}")
                
                if data.get('success') and data.get('data') and len(data['data']) > 0:
                    # Get the most recent entry (should be first if sorted by date)
                    latest_data = data['data'][0]
                    
                    # Verify this is actually our symbol
                    if latest_data.get('TOKEN_SYMBOL') == symbol:
                        sentiment_data = {
                            'tm_trader_grade': round(latest_data.get('TM_TRADER_GRADE', 0), 2),
                            'tm_ta_grade': round(latest_data.get('TA_GRADE', 0), 2),
                            'tm_quant_grade': round(latest_data.get('QUANT_GRADE', 0), 2),
                            'tm_grade_change_24h': round(latest_data.get('TM_TRADER_GRADE_24H_PCT_CHANGE', 0), 2),
                            'tm_data_available': True
                        }
                        
                        sentiment_logger.info(f"TokenMetrics extracted for {symbol}: TM_Grade={sentiment_data['tm_trader_grade']}, TA_Grade={sentiment_data['tm_ta_grade']}, Quant_Grade={sentiment_data['tm_quant_grade']}")
                        
                        # Cache and return
                        self.sentiment_cache[cache_key] = sentiment_data
                        return sentiment_data
                    else:
                        sentiment_logger.warning(f"TokenMetrics: Response contains different symbol than requested ({latest_data.get('TOKEN_SYMBOL')} vs {symbol})")
                else:
                    sentiment_logger.info(f"TokenMetrics: No data available for symbol {symbol}")
            else:
                sentiment_logger.error(f"TokenMetrics API error for {symbol}: {response.status_code} - {response.text[:200]}")
            
            # Return empty data if not found
            sentiment_data = self._get_empty_tm_data()
            self.sentiment_cache[cache_key] = sentiment_data
            return sentiment_data
            
        except Exception as e:
            sentiment_logger.error(f"Error fetching TokenMetrics data for {symbol}: {e}")
            return self._get_empty_tm_data()

    def _get_sentiment_rating(self, sentiment_score: float) -> str:
        """Convert numerical sentiment to rating"""
        if sentiment_score >= 70:
            return "Very Bullish"
        elif sentiment_score >= 60:
            return "Bullish"
        elif sentiment_score >= 40:
            return "Neutral"
        elif sentiment_score >= 30:
            return "Bearish"
        else:
            return "Very Bearish"

    def _get_empty_lunar_data(self) -> Dict[str, Any]:
        """Return empty LunarCrush data structure"""
        return {
            'lunar_sentiment_score': 0,
            'lunar_social_score': 0,
            'lunar_galaxy_score': 0,
            'lunar_alt_rank': 999,
            'lunar_sentiment_rating': 'No Data',
            'lunar_data_available': False
        }

    def _get_empty_tm_data(self) -> Dict[str, Any]:
        """Return empty TokenMetrics data structure"""
        return {
            'tm_trader_grade': 0,
            'tm_ta_grade': 0,
            'tm_quant_grade': 0,
            'tm_grade_change_24h': 0,
            'tm_data_available': False
        }

    def analyze_sentiment_alignment(self, setup_type: str, lunar_data: Dict, tm_data: Dict) -> Dict[str, Any]:
        """Analyze if sentiment aligns with trade direction"""
        try:
            alignment_score = 0
            alignment_factors = []
            
            # LunarCrush alignment
            if lunar_data['lunar_data_available']:
                galaxy_score = lunar_data['lunar_galaxy_score']
                
                if setup_type == 'LONG':
                    if galaxy_score >= 70:
                        alignment_score += 2
                        alignment_factors.append("Strong Galaxy Score")
                    elif galaxy_score >= 50:
                        alignment_score += 1
                        alignment_factors.append("Good Galaxy Score")
                    elif galaxy_score < 40:
                        alignment_score -= 1
                        alignment_factors.append("Weak Galaxy Score")
                        
                elif setup_type == 'SHORT':
                    if galaxy_score <= 30:
                        alignment_score += 2
                        alignment_factors.append("Low Galaxy Score Supports Short")
                    elif galaxy_score <= 50:
                        alignment_score += 1
                        alignment_factors.append("Neutral Galaxy Score")
                    else:
                        alignment_score -= 1
                        alignment_factors.append("High Galaxy Score Against Short")
            
            # TokenMetrics alignment (unchanged)
            if tm_data['tm_data_available']:
                trader_grade = tm_data['tm_trader_grade']
                
                if trader_grade >= 80:
                    if setup_type == 'LONG':
                        alignment_score += 2
                        alignment_factors.append("High Trader Grade")
                    elif setup_type == 'SHORT':
                        alignment_score -= 1
                        alignment_factors.append("High Grade Against Short")
                elif trader_grade >= 60:
                    alignment_score += 1
                    alignment_factors.append("Good Trader Grade")
                elif trader_grade < 50:
                    if setup_type == 'SHORT':
                        alignment_score += 1
                        alignment_factors.append("Low Grade Supports Short")
                    else:
                        alignment_score -= 1
                        alignment_factors.append("Low Trader Grade")
            
            # Overall alignment assessment
            if alignment_score >= 3:
                overall_alignment = "Strong Positive"
            elif alignment_score >= 1:
                overall_alignment = "Positive"
            elif alignment_score == 0:
                overall_alignment = "Neutral"
            elif alignment_score >= -1:
                overall_alignment = "Negative"
            else:
                overall_alignment = "Strong Negative"
            
            return {
                'alignment_score': alignment_score,
                'overall_alignment': overall_alignment,
                'alignment_factors': alignment_factors
            }
            
        except Exception as e:
            sentiment_logger.error(f"Error analyzing sentiment alignment: {e}")
            return {
                'alignment_score': 0,
                'overall_alignment': 'No Data',
                'alignment_factors': []
            }


class MarketSentimentAnalyzer:
    """Analyze broader crypto market sentiment"""
    
    def __init__(self):
        self.cmc_api_key = CMC_API_KEY
        
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index from Alternative.me"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    fng_data = data['data'][0]
                    value = int(fng_data['value'])
                    classification = fng_data['value_classification']
                    
                    return {
                        'value': value,
                        'classification': classification,
                        'signal': self._interpret_fear_greed(value),
                        'available': True
                    }
            
            return {'value': 50, 'classification': 'Neutral', 'signal': 'Neutral', 'available': False}
            
        except Exception as e:
            sentiment_logger.error(f"Error fetching Fear & Greed Index: {e}")
            return {'value': 50, 'classification': 'Neutral', 'signal': 'Neutral', 'available': False}

    def get_btc_dominance(self) -> Dict[str, Any]:
        """Get BTC dominance from CoinMarketCap"""
        try:
            url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": self.cmc_api_key}
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                btc_dominance = data['data']['btc_dominance']
                btc_dominance_24h = data['data']['btc_dominance_24h_percentage_change']
                
                return {
                    'current': round(btc_dominance, 2),
                    'change_24h': round(btc_dominance_24h, 2),
                    'signal': self._interpret_btc_dominance(btc_dominance_24h),
                    'available': True
                }
            
            return {'current': 50, 'change_24h': 0, 'signal': 'Neutral', 'available': False}
            
        except Exception as e:
            sentiment_logger.error(f"Error fetching BTC dominance: {e}")
            return {'current': 50, 'change_24h': 0, 'signal': 'Neutral', 'available': False}

    def _interpret_fear_greed(self, value: int) -> str:
        """Interpret Fear & Greed Index"""
        if value <= 20:
            return "Bullish Signal"
        elif value <= 30:
            return "Positive"
        elif value <= 70:
            return "Neutral"
        elif value <= 80:
            return "Caution"
        else:
            return "High Risk"

    def _interpret_btc_dominance(self, change_24h: float) -> str:
        """Interpret BTC dominance change"""
        if change_24h > 2:
            return "High Risk"
        elif change_24h > 0.5:
            return "Caution"
        elif change_24h > -0.5:
            return "Neutral"
        elif change_24h > -2:
            return "Positive"
        else:
            return "Bullish Signal"

    def get_complete_market_sentiment(self) -> Dict[str, Any]:
        """Get complete market sentiment analysis"""
        try:
            fng = self.get_fear_greed_index()
            time.sleep(1)
            
            btc_dominance = self.get_btc_dominance()
            
            return {
                'fear_greed': fng,
                'btc_dominance': btc_dominance,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            sentiment_logger.error(f"Error getting complete market sentiment: {e}")
            return {
                'fear_greed': {'value': 50, 'classification': 'Neutral', 'signal': 'Neutral', 'available': False},
                'btc_dominance': {'current': 50, 'change_24h': 0, 'signal': 'Neutral', 'available': False},
                'timestamp': datetime.now()
            }