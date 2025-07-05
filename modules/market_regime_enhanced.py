"""
Enhanced Market Regime Analyzer with Futures Intelligence
Standalone module that adds 3 new futures-based intelligence layers
"""

import ccxt
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp

class EnhancedMarketRegimeAnalyzer:
    """
    Enhanced Market Regime Analysis with 9-layer intelligence including futures data
    
    INTELLIGENCE LAYERS:
    1-6: Existing (BTC technical, alt context, traditional markets, sentiment, correlation, volatility)
    7: BTC Funding Sentiment (contrarian signals)
    8: Liquidation Pressure (support/resistance zones)  
    9: Synthetic Supply Ratio (squeeze potential)
    """
    
    def __init__(self, original_regime_analyzer):
        """Initialize with reference to original analyzer to preserve existing functionality"""
        self.original_analyzer = original_regime_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Futures data sources
        self.exchanges = self._initialize_futures_exchanges()
        self.funding_thresholds = {
            'extreme_bullish': 0.01,    # 1% funding (extreme greed)
            'moderate_bullish': 0.005,  # 0.5% funding 
            'neutral_high': 0.002,      # 0.2% funding
            'neutral_low': -0.002,      # -0.2% funding
            'moderate_bearish': -0.005, # -0.5% funding
            'extreme_bearish': -0.01    # -1% funding (extreme fear)
        }
        
        # Liquidation analysis parameters
        self.liquidation_levels = {
            'high_pressure': 1000000000,  # $1B+ liquidations
            'moderate_pressure': 500000000, # $500M liquidations
            'low_pressure': 100000000     # $100M liquidations
        }
        
    def _initialize_futures_exchanges(self) -> Dict:
        """Initialize exchange connections for futures data"""
        try:
            exchanges = {
                'binance': ccxt.binance({
                    'apiKey': '',  # Read-only for public data
                    'secret': '',
                    'sandbox': False,
                    'enableRateLimit': True,
                }),
                'bybit': ccxt.bybit({
                    'apiKey': '',
                    'secret': '', 
                    'sandbox': False,
                    'enableRateLimit': True,
                }),
                'okx': ccxt.okx({
                    'apiKey': '',
                    'secret': '',
                    'sandbox': False,
                    'enableRateLimit': True,
                })
            }
            
            self.logger.info("Futures exchanges initialized for enhanced regime analysis")
            return exchanges
            
        except Exception as e:
            self.logger.error(f"Failed to initialize futures exchanges: {e}")
            return {}
    
    async def analyze_enhanced_regime(self, btc_data: pd.DataFrame) -> Dict:
        """
        Main entry point for enhanced 9-layer market regime analysis
        
        Args:
            btc_data: Bitcoin OHLCV data for analysis
            
        Returns:
            Dict containing all 9 layers of market intelligence
        """
        try:
            # Get original 6-layer analysis
            original_analysis = await self._get_original_analysis(btc_data)
            
            # Add 3 new futures intelligence layers
            futures_analysis = await self._analyze_futures_intelligence()
            
            # Combine and calculate enhanced regime score
            enhanced_regime = self._calculate_enhanced_regime_score(
                original_analysis, 
                futures_analysis
            )
            
            return enhanced_regime
            
        except Exception as e:
            self.logger.error(f"Enhanced regime analysis failed: {e}")
            # Fallback to original analysis
            return await self._get_original_analysis(btc_data)
    
    async def _get_original_analysis(self, btc_data: pd.DataFrame) -> Dict:
        """Get analysis from original market regime analyzer"""
        try:
            # Call existing market regime analyzer
            return await self.original_analyzer.analyze_market_regime('ETH', btc_data)
        except Exception as e:
            self.logger.error(f"Original regime analysis failed: {e}")
            return self._get_fallback_analysis()
    
    async def _analyze_futures_intelligence(self) -> Dict:
        """
        Analyze the 3 new futures intelligence layers
        
        Returns:
            Dict with funding sentiment, liquidation pressure, synthetic supply analysis
        """
        try:
            # Layer 7: BTC Funding Sentiment Analysis
            funding_sentiment = await self._analyze_funding_sentiment()
            
            # Layer 8: Liquidation Pressure Analysis  
            liquidation_pressure = await self._analyze_liquidation_pressure()
            
            # Layer 9: Synthetic Supply Ratio Analysis
            synthetic_supply = await self._analyze_synthetic_supply_ratio()
            
            return {
                'funding_sentiment': funding_sentiment,
                'liquidation_pressure': liquidation_pressure, 
                'synthetic_supply_analysis': synthetic_supply,
                'futures_data_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Futures intelligence analysis failed: {e}")
            return self._get_fallback_futures_analysis()
    
    async def _analyze_funding_sentiment(self) -> Dict:
        """
        Layer 7: Analyze BTC funding rates across exchanges for contrarian sentiment
        
        Funding Rate Interpretation:
        - Extreme positive (>1%): Contrarian bearish (market top signal)
        - Extreme negative (<-1%): Contrarian bullish (market bottom signal)  
        - Moderate levels: Neutral to trending sentiment
        """
        funding_data = {}
        
        try:
            # DIAGNOSTIC: Check available methods
            for exchange_name, exchange in self.exchanges.items():
                try:
                    print(f"DEBUG: Available methods for {exchange_name}:")
                    methods = [method for method in dir(exchange) if 'funding' in method.lower()]
                    print(f"DEBUG: Funding-related methods: {methods}")
                    
                    # Also check for general public methods
                    public_methods = [method for method in dir(exchange) if method.startswith('public')]
                    print(f"DEBUG: First 10 public methods: {public_methods[:10]}")
                    
                except Exception as e:
                    print(f"DEBUG: Error checking methods for {exchange_name}: {e}")
            
            # Get funding rates from multiple exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # Get BTC perpetual funding rate
                    if exchange_name == 'binance':
                        funding_info = exchange.fapiPublicGetFundingRate({'symbol': 'BTCUSDT'})
                        current_rate = float(funding_info[0]['fundingRate'])
                    
                    elif exchange_name == 'bybit':
                        funding_info = exchange.publicGetV5MarketFundingHistory({'symbol': 'BTCUSDT', 'category': 'linear'})
                        current_rate = float(funding_info['result']['list'][0]['fundingRate'])
                    
                    elif exchange_name == 'okx':
                        funding_info = exchange.publicGetPublicFundingRate({'instId': 'BTC-USDT-SWAP'})
                        current_rate = float(funding_info['data'][0]['fundingRate'])

                    elif exchange_name == 'kucoin':
                        funding_info = exchange.publicGetApiV1ContractsFundingRates({'symbol': 'XBTUSDM'})
                        current_rate = float(funding_info['data']['value'])
                    
                    funding_data[exchange_name] = current_rate
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get funding rate from {exchange_name}: {e}")
                    continue
            
            # Calculate aggregate funding sentiment
            if funding_data:
                avg_funding = np.mean(list(funding_data.values()))
                sentiment = self._interpret_funding_sentiment(avg_funding)
                
                return {
                    'average_funding_rate': avg_funding,
                    'exchange_rates': funding_data,
                    'sentiment_signal': sentiment['signal'],
                    'contrarian_strength': sentiment['strength'],
                    'market_timing_signal': sentiment['timing'],
                    'confidence': sentiment['confidence']
                }
            else:
                return self._get_fallback_funding_analysis()
                
        except Exception as e:
            self.logger.error(f"Funding sentiment analysis failed: {e}")
            return self._get_fallback_funding_analysis()
    
    def _interpret_funding_sentiment(self, funding_rate: float) -> Dict:
        """Interpret funding rate for contrarian sentiment signals"""
        
        if funding_rate >= self.funding_thresholds['extreme_bullish']:
            return {
                'signal': 'CONTRARIAN_BEARISH',
                'strength': 'EXTREME', 
                'timing': 'POTENTIAL_TOP',
                'confidence': 90,
                'reasoning': 'Extreme positive funding suggests over-leveraged longs, potential top'
            }
        elif funding_rate >= self.funding_thresholds['moderate_bullish']:
            return {
                'signal': 'BEARISH_LEAN',
                'strength': 'MODERATE',
                'timing': 'CAUTION',
                'confidence': 70,
                'reasoning': 'High positive funding suggests bullish excess, exercise caution'
            }
        elif funding_rate <= self.funding_thresholds['extreme_bearish']:
            return {
                'signal': 'CONTRARIAN_BULLISH', 
                'strength': 'EXTREME',
                'timing': 'POTENTIAL_BOTTOM',
                'confidence': 90,
                'reasoning': 'Extreme negative funding suggests over-leveraged shorts, potential bottom'
            }
        elif funding_rate <= self.funding_thresholds['moderate_bearish']:
            return {
                'signal': 'BULLISH_LEAN',
                'strength': 'MODERATE', 
                'timing': 'OPPORTUNITY',
                'confidence': 70,
                'reasoning': 'Negative funding suggests bearish excess, potential opportunity'
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'strength': 'LOW',
                'timing': 'NO_SIGNAL', 
                'confidence': 50,
                'reasoning': 'Funding rate in neutral range, no strong contrarian signal'
            }
    
    async def _analyze_liquidation_pressure(self) -> Dict:
        """
        Layer 8: Analyze liquidation zones for support/resistance levels
        
        Liquidation clusters often act as:
        - Support levels (where shorts get liquidated)
        - Resistance levels (where longs get liquidated)
        """
        try:
            # Get liquidation data from multiple sources
            liquidation_data = await self._fetch_liquidation_data()
            
            # Analyze liquidation clusters
            liquidation_zones = self._identify_liquidation_zones(liquidation_data)
            
            # Calculate pressure metrics
            pressure_analysis = self._calculate_liquidation_pressure(liquidation_zones)
            
            return {
                'total_liquidations_24h': liquidation_data.get('total_24h', 0),
                'long_liquidations': liquidation_data.get('longs_24h', 0),
                'short_liquidations': liquidation_data.get('shorts_24h', 0),
                'liquidation_dominance': liquidation_data.get('dominance', 'BALANCED'),
                'support_zones': liquidation_zones.get('support_levels', []),
                'resistance_zones': liquidation_zones.get('resistance_levels', []),
                'pressure_level': pressure_analysis['level'],
                'market_impact': pressure_analysis['impact'],
                'confidence': pressure_analysis['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Liquidation pressure analysis failed: {e}")
            return self._get_fallback_liquidation_analysis()
    
    async def _fetch_liquidation_data(self) -> Dict:
        """Fetch recent liquidation data from available sources"""
        try:
            # This would integrate with liquidation data APIs
            # For now, return sample structure
            return {
                'total_24h': 150000000,  # $150M total liquidations
                'longs_24h': 90000000,   # $90M long liquidations  
                'shorts_24h': 60000000,  # $60M short liquidations
                'dominance': 'LONG_HEAVY',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch liquidation data: {e}")
            return {}
    
    def _identify_liquidation_zones(self, liquidation_data: Dict) -> Dict:
        """Identify key liquidation zones that may act as support/resistance"""
        try:
            # Analyze liquidation patterns to identify zones
            # This would use actual liquidation heatmap data
            return {
                'support_levels': [60000, 58500, 57000],  # Price levels with short liquidation clusters
                'resistance_levels': [70000, 72000, 75000],  # Price levels with long liquidation clusters
                'zone_strength': 'MODERATE'
            }
        except Exception as e:
            self.logger.error(f"Failed to identify liquidation zones: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _calculate_liquidation_pressure(self, zones: Dict) -> Dict:
        """Calculate overall liquidation pressure and market impact"""
        try:
            support_count = len(zones.get('support_levels', []))
            resistance_count = len(zones.get('resistance_levels', []))
            
            if support_count > resistance_count:
                return {
                    'level': 'UPWARD_PRESSURE',
                    'impact': 'BULLISH_BIAS', 
                    'confidence': 75
                }
            elif resistance_count > support_count:
                return {
                    'level': 'DOWNWARD_PRESSURE',
                    'impact': 'BEARISH_BIAS',
                    'confidence': 75  
                }
            else:
                return {
                    'level': 'BALANCED_PRESSURE',
                    'impact': 'NEUTRAL',
                    'confidence': 60
                }
        except Exception as e:
            return {'level': 'UNKNOWN', 'impact': 'NEUTRAL', 'confidence': 30}
    
    async def _analyze_synthetic_supply_ratio(self) -> Dict:
        """
        Layer 9: Analyze synthetic vs real Bitcoin supply for squeeze potential
        
        Key metrics:
        - Real BTC supply rate: 3.125 BTC per 10 minutes (post-halving)
        - Synthetic BTC creation: Estimated from futures open interest growth
        - Squeeze risk: When synthetic >> real supply
        """
        try:
            # Calculate real Bitcoin supply metrics
            real_supply_metrics = self._calculate_real_supply_metrics()
            
            # Estimate synthetic supply from futures data
            synthetic_supply_metrics = await self._estimate_synthetic_supply()
            
            # Calculate squeeze risk assessment
            squeeze_analysis = self._assess_squeeze_potential(
                real_supply_metrics, 
                synthetic_supply_metrics
            )
            
            return {
                'real_btc_supply_rate': real_supply_metrics['daily_new_btc'],
                'estimated_synthetic_creation': synthetic_supply_metrics['daily_synthetic'],
                'supply_ratio': synthetic_supply_metrics['ratio'],
                'squeeze_risk_level': squeeze_analysis['risk_level'],
                'squeeze_potential': squeeze_analysis['potential'],
                'market_implication': squeeze_analysis['implication'],
                'confidence': squeeze_analysis['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Synthetic supply analysis failed: {e}")
            return self._get_fallback_synthetic_analysis()
    
    def _calculate_real_supply_metrics(self) -> Dict:
        """Calculate real Bitcoin supply creation metrics"""
        # Post-halving: 3.125 BTC per 10 minutes
        btc_per_block = 3.125
        blocks_per_day = 144  # 6 blocks per hour * 24 hours
        daily_new_btc = btc_per_block * blocks_per_day
        
        return {
            'btc_per_block': btc_per_block,
            'blocks_per_day': blocks_per_day,
            'daily_new_btc': daily_new_btc,
            'annual_new_btc': daily_new_btc * 365
        }
    
    async def _estimate_synthetic_supply(self) -> Dict:
        """Estimate synthetic Bitcoin creation from futures market growth"""
        try:
            # This would analyze open interest growth rates
            # For now, use article's estimate: 1.15 synthetic BTC per 10 minutes
            synthetic_per_10min = 1.15
            daily_synthetic = synthetic_per_10min * 144
            
            real_daily = 450  # 3.125 * 144
            ratio = daily_synthetic / real_daily
            
            return {
                'synthetic_per_10min': synthetic_per_10min,
                'daily_synthetic': daily_synthetic,
                'ratio': ratio,
                'interpretation': 'MODERATE_SYNTHETIC_PRESSURE' if ratio < 1 else 'HIGH_SYNTHETIC_PRESSURE'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate synthetic supply: {e}")
            return {'daily_synthetic': 0, 'ratio': 0}
    
    def _assess_squeeze_potential(self, real_metrics: Dict, synthetic_metrics: Dict) -> Dict:
        """Assess potential for supply squeeze based on real vs synthetic ratio"""
        try:
            ratio = synthetic_metrics.get('ratio', 0)
            
            if ratio > 2.0:  # Synthetic > 2x real supply
                return {
                    'risk_level': 'HIGH',
                    'potential': 'SIGNIFICANT_SQUEEZE_RISK',
                    'implication': 'If spot demand surges, potential for violent upward moves',
                    'confidence': 80
                }
            elif ratio > 1.0:  # Synthetic > real supply  
                return {
                    'risk_level': 'MODERATE', 
                    'potential': 'MODERATE_SQUEEZE_RISK',
                    'implication': 'Synthetic supply dominance may amplify volatility',
                    'confidence': 70
                }
            else:  # Real supply >= synthetic
                return {
                    'risk_level': 'LOW',
                    'potential': 'LOW_SQUEEZE_RISK', 
                    'implication': 'Real supply keeps pace with synthetic creation',
                    'confidence': 60
                }
                
        except Exception as e:
            return {
                'risk_level': 'UNKNOWN',
                'potential': 'ANALYSIS_FAILED',
                'implication': 'Unable to assess squeeze potential',
                'confidence': 20
            }
    
    def _calculate_enhanced_regime_score(self, original_analysis: Dict, futures_analysis: Dict) -> Dict:
        """
        Calculate enhanced 9-layer regime score
        
        Weighting:
        - Original 6 layers: 70% weight
        - Futures 3 layers: 30% weight
        """
        try:
            # Get original regime confidence
            original_confidence = original_analysis.get('regime_confidence', 50)
            
            # Calculate futures intelligence score
            futures_score = self._calculate_futures_score(futures_analysis)
            
            # Weighted combination
            enhanced_confidence = (original_confidence * 0.7) + (futures_score * 0.3)
            
            # Determine enhanced regime classification
            enhanced_regime = self._classify_enhanced_regime(enhanced_confidence, futures_analysis)
            
            return {
                # Original analysis (preserved)
                **original_analysis,
                
                # Enhanced futures intelligence
                'futures_intelligence': futures_analysis,
                
                # Enhanced regime metrics
                'enhanced_regime_confidence': enhanced_confidence,
                'enhanced_regime_classification': enhanced_regime,
                'intelligence_layers': 9,
                'futures_weight': 30,
                
                # Trading implications
                'position_sizing_multiplier': self._calculate_position_multiplier(enhanced_confidence),
                'risk_level': self._assess_enhanced_risk_level(enhanced_confidence, futures_analysis),
                'market_timing_signal': self._generate_timing_signal(futures_analysis),
                
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced regime score calculation failed: {e}")
            return original_analysis
    
    def _calculate_futures_score(self, futures_analysis: Dict) -> float:
        """Calculate aggregate score from futures intelligence layers"""
        try:
            scores = []
            
            # Funding sentiment score (0-100)
            funding = futures_analysis.get('funding_sentiment', {})
            funding_score = funding.get('confidence', 50)
            if funding.get('signal') == 'CONTRARIAN_BULLISH':
                funding_score += 20  # Boost for contrarian bullish
            elif funding.get('signal') == 'CONTRARIAN_BEARISH':
                funding_score -= 10  # Reduce for contrarian bearish
            scores.append(max(0, min(100, funding_score)))
            
            # Liquidation pressure score (0-100)
            liquidation = futures_analysis.get('liquidation_pressure', {})
            liq_confidence = liquidation.get('confidence', 50)
            if liquidation.get('market_impact') == 'BULLISH_BIAS':
                liq_confidence += 15
            elif liquidation.get('market_impact') == 'BEARISH_BIAS':
                liq_confidence -= 10
            scores.append(max(0, min(100, liq_confidence)))
            
            # Synthetic supply score (0-100)
            synthetic = futures_analysis.get('synthetic_supply_analysis', {})
            synth_confidence = synthetic.get('confidence', 50)
            if synthetic.get('squeeze_risk_level') == 'HIGH':
                synth_confidence += 10  # Slight boost for squeeze potential
            scores.append(max(0, min(100, synth_confidence)))
            
            return np.mean(scores) if scores else 50
            
        except Exception as e:
            self.logger.error(f"Futures score calculation failed: {e}")
            return 50
    
    def _classify_enhanced_regime(self, confidence: float, futures_analysis: Dict) -> str:
        """Classify enhanced market regime based on 9-layer analysis"""
        try:
            # Base classification from confidence
            if confidence >= 80:
                base_regime = "EXCELLENT"
            elif confidence >= 70:
                base_regime = "GOOD" 
            elif confidence >= 60:
                base_regime = "FAIR"
            elif confidence >= 50:
                base_regime = "POOR"
            else:
                base_regime = "VERY_POOR"
            
            # Modify based on futures intelligence
            funding_signal = futures_analysis.get('funding_sentiment', {}).get('signal', 'NEUTRAL')
            
            if funding_signal == 'CONTRARIAN_BULLISH':
                return f"{base_regime}_CONTRARIAN_BULLISH"
            elif funding_signal == 'CONTRARIAN_BEARISH':
                return f"{base_regime}_CONTRARIAN_BEARISH"
            else:
                return f"{base_regime}_NEUTRAL_FUTURES"
                
        except Exception as e:
            return "UNKNOWN"
    
    def _calculate_position_multiplier(self, confidence: float) -> float:
        """Calculate position sizing multiplier based on enhanced confidence"""
        # More conservative than original due to futures volatility
        if confidence >= 85:
            return 1.2  # Reduced from potential 1.5x
        elif confidence >= 75:
            return 1.0
        elif confidence >= 65:
            return 0.8
        elif confidence >= 55:
            return 0.6
        elif confidence >= 45:
            return 0.4
        else:
            return 0.25  # Very conservative for low confidence
    
    def _assess_enhanced_risk_level(self, confidence: float, futures_analysis: Dict) -> str:
        """Assess overall risk level incorporating futures intelligence"""
        try:
            # Base risk from confidence
            if confidence >= 75:
                base_risk = "LOW"
            elif confidence >= 60:
                base_risk = "MEDIUM"
            else:
                base_risk = "HIGH"
            
            # Adjust for futures factors
            squeeze_risk = futures_analysis.get('synthetic_supply_analysis', {}).get('squeeze_risk_level', 'UNKNOWN')
            funding_strength = futures_analysis.get('funding_sentiment', {}).get('strength', 'LOW')
            
            # High squeeze risk or extreme funding = elevated risk
            if squeeze_risk == 'HIGH' or funding_strength == 'EXTREME':
                if base_risk == "LOW":
                    return "MEDIUM"
                elif base_risk == "MEDIUM":
                    return "HIGH"
                else:
                    return "VERY_HIGH"
            
            return base_risk
            
        except Exception as e:
            return "UNKNOWN"
    
    def _generate_timing_signal(self, futures_analysis: Dict) -> str:
        """Generate market timing signal from futures intelligence"""
        try:
            funding_timing = futures_analysis.get('funding_sentiment', {}).get('timing', 'NO_SIGNAL')
            liquidation_impact = futures_analysis.get('liquidation_pressure', {}).get('market_impact', 'NEUTRAL')
            
            if funding_timing == 'POTENTIAL_BOTTOM' and liquidation_impact == 'BULLISH_BIAS':
                return "STRONG_BUY_SIGNAL"
            elif funding_timing == 'POTENTIAL_BOTTOM':
                return "BUY_SIGNAL"
            elif funding_timing == 'POTENTIAL_TOP' and liquidation_impact == 'BEARISH_BIAS':
                return "STRONG_SELL_SIGNAL"
            elif funding_timing == 'POTENTIAL_TOP':
                return "SELL_SIGNAL"
            elif liquidation_impact == 'BULLISH_BIAS':
                return "BULLISH_LEAN"
            elif liquidation_impact == 'BEARISH_BIAS':
                return "BEARISH_LEAN"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            return "UNKNOWN"
    
    # Fallback methods for error handling
    def _get_fallback_analysis(self) -> Dict:
        """Fallback analysis when all systems fail"""
        return {
            'regime_confidence': 50,
            'regime_classification': 'UNKNOWN',
            'position_sizing_multiplier': 0.5,
            'intelligence_layers': 1,
            'status': 'FALLBACK_MODE'
        }
    
    def _get_fallback_futures_analysis(self) -> Dict:
        """Fallback futures analysis"""
        return {
            'funding_sentiment': {'signal': 'NEUTRAL', 'confidence': 50},
            'liquidation_pressure': {'level': 'UNKNOWN', 'confidence': 30},
            'synthetic_supply_analysis': {'risk_level': 'UNKNOWN', 'confidence': 30}
        }
    
    def _get_fallback_funding_analysis(self) -> Dict:
        """Fallback funding analysis"""
        return {
            'average_funding_rate': 0.0,
            'sentiment_signal': 'NEUTRAL',
            'confidence': 30,
            'status': 'DATA_UNAVAILABLE'
        }
    
    def _get_fallback_liquidation_analysis(self) -> Dict:
        """Fallback liquidation analysis"""
        return {
            'pressure_level': 'UNKNOWN',
            'market_impact': 'NEUTRAL', 
            'confidence': 30,
            'status': 'DATA_UNAVAILABLE'
        }
    
    def _get_fallback_synthetic_analysis(self) -> Dict:
        """Fallback synthetic supply analysis"""
        return {
            'squeeze_risk_level': 'UNKNOWN',
            'squeeze_potential': 'ANALYSIS_FAILED',
            'confidence': 30,
            'status': 'DATA_UNAVAILABLE'
        }

# Integration helper functions
async def create_enhanced_regime_analyzer(original_analyzer):
    """Factory function to create enhanced analyzer"""
    return EnhancedMarketRegimeAnalyzer(original_analyzer)

def format_enhanced_regime_output(enhanced_analysis: Dict) -> Dict:
    """Format enhanced analysis for Excel output"""
    try:
        return {
            # Original fields (preserved)
            'regime_confidence': enhanced_analysis.get('regime_confidence', 50),
            'regime_classification': enhanced_analysis.get('regime_classification', 'UNKNOWN'),
            'position_sizing_multiplier': enhanced_analysis.get('position_sizing_multiplier', 0.5),
            
            # Enhanced fields  
            'enhanced_regime_confidence': enhanced_analysis.get('enhanced_regime_confidence', 50),
            'enhanced_regime_classification': enhanced_analysis.get('enhanced_regime_classification', 'UNKNOWN'),
            'intelligence_layers': enhanced_analysis.get('intelligence_layers', 6),
            
            # Futures intelligence
            'funding_sentiment_signal': enhanced_analysis.get('futures_intelligence', {}).get('funding_sentiment', {}).get('signal', 'UNKNOWN'),
            'funding_rate': enhanced_analysis.get('futures_intelligence', {}).get('funding_sentiment', {}).get('average_funding_rate', 0),
            'liquidation_pressure': enhanced_analysis.get('futures_intelligence', {}).get('liquidation_pressure', {}).get('level', 'UNKNOWN'),
            'squeeze_risk_level': enhanced_analysis.get('futures_intelligence', {}).get('synthetic_supply_analysis', {}).get('squeeze_risk_level', 'UNKNOWN'),
            'market_timing_signal': enhanced_analysis.get('market_timing_signal', 'NEUTRAL'),
            'enhanced_risk_level': enhanced_analysis.get('risk_level', 'MEDIUM')
        }
    except Exception as e:
        logging.error(f"Failed to format enhanced regime output: {e}")
        return {'regime_confidence': 50, 'status': 'FORMAT_ERROR'}