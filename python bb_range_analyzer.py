#!/usr/bin/env python3
"""
BB Range Analyzer - ACCURATE VERSION
====================================
Fixed to match Historical Intelligence methodology and provide real success rates.
"""

import pandas as pd
import numpy as np
import ccxt
import pandas_ta as ta
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
import requests
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BBRangeAnalyzerAccurate:
    """Accurate BB Range Analyzer matching Historical Intelligence methodology"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'timeout': 60000,
            'enableRateLimit': True,
        })
        
        self.cmc_api_key = "7eaf27ab-6af5-436c-8df2-469d7dce91e7"
        
        # Match your Historical Intelligence settings
        self.bb_period = 20
        self.bb_std = 2
        self.analysis_days = 90
        self.min_samples = 15  # Minimum trades per analysis
        
    async def fetch_market_cap_data(self, limit: int = 200) -> Dict[str, Dict]:
        """Fetch market cap data from CoinMarketCap"""
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': self.cmc_api_key,
            }
            params = {
                'start': '1',
                'limit': str(limit),
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            data = response.json()
            
            market_cap_data = {}
            excluded_symbols = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD'}
            
            for coin in data['data']:
                symbol = coin['symbol']
                if symbol not in excluded_symbols:
                    market_cap_data[symbol] = {
                        'rank': coin['cmc_rank'],
                        'market_cap': coin['quote']['USD']['market_cap'],
                        'price': coin['quote']['USD']['price'],
                        'volume_24h': coin['quote']['USD']['volume_24h']
                    }
            
            logger.info(f"✅ Fetched market cap data for {len(market_cap_data)} coins")
            return market_cap_data
            
        except Exception as e:
            logger.error(f"Error fetching market cap data: {e}")
            return {}
    
    def categorize_market_cap(self, rank: int) -> str:
        """Categorize coins by market cap ranking"""
        if rank <= 50:
            return "Large Cap (Top 50)"
        elif rank <= 100:
            return "Mid Cap (51-100)"
        elif rank <= 200:
            return "Small Cap (101-200)"
        else:
            return "Micro Cap (200+)"
    
    def fetch_comprehensive_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Fetch comprehensive OHLCV data"""
        try:
            candles_needed = days * 6  # 4H timeframe = 6 candles per day
            current_limit = min(1000, candles_needed)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, '4h', limit=current_limit)
            
            if not ohlcv or len(ohlcv) < 100:
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate comprehensive indicators
            df = self.calculate_comprehensive_indicators(df)
            return df
            
        except Exception as e:
            logger.debug(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators matching your Historical Intelligence"""
        try:
            # Bollinger Bands (exact match to your settings)
            bb = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std)
            df['bb_lower'] = bb[f'BBL_{self.bb_period}_{self.bb_std}.0']
            df['bb_middle'] = bb[f'BBM_{self.bb_period}_{self.bb_std}.0']
            df['bb_upper'] = bb[f'BBU_{self.bb_period}_{self.bb_std}.0']
            
            # BB Percentage (key metric)
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # BB Width and Squeeze/Expansion
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_width_sma'] = df['bb_width'].rolling(20).mean()
            df['bb_expansion'] = df['bb_width'] > df['bb_width_sma'] * 1.1
            df['bb_squeeze'] = df['bb_width'] < df['bb_width_sma'] * 0.9
            
            # BB Trend Analysis
            df['bb_middle_slope'] = df['bb_middle'].pct_change(5) * 100
            df['bb_trend'] = 'Sideways'
            df.loc[df['bb_middle_slope'] > 1.0, 'bb_trend'] = 'Uptrend'
            df.loc[df['bb_middle_slope'] < -1.0, 'bb_trend'] = 'Downtrend'
            
            # Technical Indicators (match your scanner)
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_surge'] = df['volume'] > df['volume_sma'] * 1.5
            
            # Chaikin Money Flow
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
            
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # BB Touch Detection (strict like your scanner)
            touch_tolerance = 0.003  # 0.3% tolerance
            df['bb_touch_lower'] = df['low'] <= (df['bb_lower'] * (1 + touch_tolerance))
            df['bb_touch_upper'] = df['high'] >= (df['bb_upper'] * (1 - touch_tolerance))
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def find_bb_setups_accurate(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Find BB setups using your exact Historical Intelligence methodology"""
        setups = []
        
        if len(df) < 50:
            return setups
        
        try:
            for i in range(20, len(df) - 20):  # Leave room for outcome analysis
                current = df.iloc[i]
                
                # Test different BB percentage ranges
                bb_ranges_to_test = [
                    (0.00, 0.02, "0-2%", "LONG"),
                    (0.02, 0.05, "2-5%", "LONG"),
                    (0.05, 0.10, "5-10%", "LONG"),
                    (0.10, 0.15, "10-15%", "LONG"),
                    (0.15, 0.20, "15-20%", "LONG"),
                    (0.20, 0.25, "20-25%", "LONG"),
                    (0.75, 0.80, "75-80%", "SHORT"),
                    (0.80, 0.85, "80-85%", "SHORT"),
                    (0.85, 0.90, "85-90%", "SHORT"),
                    (0.90, 0.95, "90-95%", "SHORT"),
                    (0.95, 0.98, "95-98%", "SHORT"),
                    (0.98, 1.00, "98-100%", "SHORT")
                ]
                
                for min_bb, max_bb, range_name, setup_type in bb_ranges_to_test:
                    if min_bb <= current['bb_pct'] <= max_bb:
                        # Analyze confluence and create setup
                        setup_data = self.analyze_setup_confluence(df, i, current, setup_type, range_name, symbol)
                        if setup_data:
                            setups.append(setup_data)
                        
        except Exception as e:
            logger.debug(f"Error finding setups for {symbol}: {e}")
            
        return setups
    
    def analyze_setup_confluence(self, df: pd.DataFrame, idx: int, current: pd.Series, 
                                setup_type: str, bb_range: str, symbol: str) -> Dict:
        """Analyze confluence factors for setup quality"""
        try:
            confluence_data = {
                'symbol': symbol,
                'setup_type': setup_type,
                'bb_range': bb_range,
                'bb_pct_entry': current['bb_pct'],
                'entry_price': current['close'],
                'timestamp': df.index[idx],
                
                # Core indicators
                'rsi': current['rsi'],
                'mfi': current['mfi'],
                'stoch_k': current['stoch_k'],
                'stoch_d': current['stoch_d'],
                'cci': current['cci'],
                'cmf': current['cmf'],
                
                # BB context
                'bb_expansion': current['bb_expansion'],
                'bb_squeeze': current['bb_squeeze'],
                'bb_trend': current['bb_trend'],
                'bb_touch_lower': current['bb_touch_lower'],
                'bb_touch_upper': current['bb_touch_upper'],
                
                # Volume
                'volume_surge': current['volume_surge'],
                
                # Initialize indicator flags
                'mfi_oversold': False,
                'mfi_overbought': False,
                'stoch_oversold': False,
                'stoch_overbought': False,
                'rsi_oversold': False,
                'rsi_overbought': False,
                'cci_oversold': False,
                'cci_overbought': False,
                'cmf_negative': False,
                'cmf_positive': False,
            }
            
            # Set indicator flags based on setup type
            if setup_type == 'LONG':
                confluence_data['mfi_oversold'] = current['mfi'] < 20
                confluence_data['mfi_very_oversold'] = current['mfi'] < 10
                confluence_data['stoch_oversold'] = current['stoch_k'] < 20 and current['stoch_d'] < 20
                confluence_data['rsi_oversold'] = current['rsi'] < 30
                confluence_data['cci_oversold'] = current['cci'] < -100
                confluence_data['cmf_negative'] = current['cmf'] < -0.1
            else:  # SHORT
                confluence_data['mfi_overbought'] = current['mfi'] > 80
                confluence_data['mfi_very_overbought'] = current['mfi'] > 90
                confluence_data['stoch_overbought'] = current['stoch_k'] > 80 and current['stoch_d'] > 80
                confluence_data['rsi_overbought'] = current['rsi'] > 70
                confluence_data['cci_overbought'] = current['cci'] > 100
                confluence_data['cmf_positive'] = current['cmf'] > 0.1
            
            # Calculate setup quality score
            confluence_score = self.calculate_confluence_score(confluence_data)
            confluence_data['confluence_score'] = confluence_score
            
            # Only return setups with minimum confluence
            if confluence_score >= 2:  # Require at least 2 confluence factors
                return confluence_data
                
            return None
            
        except Exception as e:
            logger.debug(f"Error analyzing confluence: {e}")
            return None
    
    def calculate_confluence_score(self, setup_data: Dict) -> int:
        """Calculate confluence score based on multiple factors"""
        score = 0
        
        if setup_data['setup_type'] == 'LONG':
            # Oversold conditions
            if setup_data['mfi_oversold']:
                score += 2
            if setup_data['mfi_very_oversold']:
                score += 1  # Bonus
            if setup_data['stoch_oversold']:
                score += 2
            if setup_data['rsi_oversold']:
                score += 1
            if setup_data['cci_oversold']:
                score += 1
            if setup_data['cmf_negative']:
                score += 1
            
            # BB context bonuses
            if setup_data['bb_touch_lower']:
                score += 1
            if setup_data['bb_expansion']:
                score += 1
            if setup_data['bb_trend'] == 'Uptrend':
                score += 1
            
        else:  # SHORT
            # Overbought conditions
            if setup_data['mfi_overbought']:
                score += 2
            if setup_data['mfi_very_overbought']:
                score += 1
            if setup_data['stoch_overbought']:
                score += 2
            if setup_data['rsi_overbought']:
                score += 1
            if setup_data['cci_overbought']:
                score += 1
            if setup_data['cmf_positive']:
                score += 1
            
            # BB context bonuses
            if setup_data['bb_touch_upper']:
                score += 1
            if setup_data['bb_expansion']:
                score += 1
            if setup_data['bb_trend'] == 'Downtrend':
                score += 1
        
        # Volume bonus
        if setup_data['volume_surge']:
            score += 1
            
        return score
    
    def calculate_realistic_outcome(self, df: pd.DataFrame, entry_idx: int, setup_data: Dict) -> Dict:
        """Calculate realistic outcome using proper targets (NOT BB middle)"""
        try:
            entry_row = df.iloc[entry_idx]
            entry_price = setup_data['entry_price']
            setup_type = setup_data['setup_type']
            
            # Look forward up to 20 periods (80 hours on 4H)
            max_forward = min(20, len(df) - entry_idx - 1)
            if max_forward < 5:
                return None
            
            future_data = df.iloc[entry_idx:entry_idx + max_forward + 1]
            
            # Realistic profit targets (NOT BB middle!)
            if setup_type == 'LONG':
                profit_target_3pct = entry_price * 1.03
                profit_target_5pct = entry_price * 1.05
                stop_loss = entry_price * 0.98  # 2% stop
            else:  # SHORT
                profit_target_3pct = entry_price * 0.97
                profit_target_5pct = entry_price * 0.95
                stop_loss = entry_price * 1.02  # 2% stop
            
            # Track outcome
            max_favorable = 0
            max_adverse = 0
            hit_3pct = False
            hit_5pct = False
            hit_stop = False
            outcome_hours = 0
            final_gain_pct = 0
            
            for i, (timestamp, row) in enumerate(future_data.iterrows()):
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                if setup_type == 'LONG':
                    # Track gains/losses for LONG
                    period_high_gain = ((high_price - entry_price) / entry_price) * 100
                    period_low_loss = ((low_price - entry_price) / entry_price) * 100
                    
                    max_favorable = max(max_favorable, period_high_gain)
                    max_adverse = min(max_adverse, period_low_loss)
                    
                    # Check targets
                    if not hit_3pct and high_price >= profit_target_3pct:
                        hit_3pct = True
                        outcome_hours = i * 4
                        final_gain_pct = 3.0
                        break
                    elif not hit_5pct and high_price >= profit_target_5pct:
                        hit_5pct = True
                        outcome_hours = i * 4
                        final_gain_pct = 5.0
                        break
                    elif low_price <= stop_loss:
                        hit_stop = True
                        outcome_hours = i * 4
                        final_gain_pct = -2.0
                        break
                        
                else:  # SHORT
                    # Track gains/losses for SHORT
                    period_low_gain = ((entry_price - low_price) / entry_price) * 100
                    period_high_loss = ((high_price - entry_price) / entry_price) * 100
                    
                    max_favorable = max(max_favorable, period_low_gain)
                    max_adverse = min(max_adverse, -period_high_loss)
                    
                    # Check targets
                    if not hit_3pct and low_price <= profit_target_3pct:
                        hit_3pct = True
                        outcome_hours = i * 4
                        final_gain_pct = 3.0
                        break
                    elif not hit_5pct and low_price <= profit_target_5pct:
                        hit_5pct = True
                        outcome_hours = i * 4
                        final_gain_pct = 5.0
                        break
                    elif high_price >= stop_loss:
                        hit_stop = True
                        outcome_hours = i * 4
                        final_gain_pct = -2.0
                        break
            
            # If no target/stop hit, use final price
            if not (hit_3pct or hit_5pct or hit_stop):
                final_price = future_data['close'].iloc[-1]
                if setup_type == 'LONG':
                    final_gain_pct = ((final_price - entry_price) / entry_price) * 100
                else:
                    final_gain_pct = ((entry_price - final_price) / entry_price) * 100
                outcome_hours = (len(future_data) - 1) * 4
            
            # Determine success (3%+ gain OR 1.5:1 risk/reward)
            success = hit_3pct or hit_5pct or (final_gain_pct > 1.5 and final_gain_pct > abs(max_adverse) * 1.5)
            
            return {
                'success': success,
                'gain_pct': final_gain_pct,
                'duration_hours': outcome_hours,
                'max_favorable': max_favorable,
                'max_adverse': max_adverse,
                'hit_3pct': hit_3pct,
                'hit_5pct': hit_5pct,
                'hit_stop': hit_stop
            }
            
        except Exception as e:
            logger.debug(f"Error calculating outcome: {e}")
            return None
    
    def run_comprehensive_analysis(self, coin_limit: int = 200) -> Dict:
        """Run comprehensive analysis matching Historical Intelligence methodology"""
        logger.info("🚀 Starting Accurate BB Analysis (Historical Intelligence Methodology)")
        
        # Fetch market cap data
        market_cap_data = asyncio.run(self.fetch_market_cap_data(coin_limit))
        
        if not market_cap_data:
            logger.error("❌ No market cap data - cannot proceed")
            return {}
        
        all_setups = []
        analyzed_coins = 0
        successful_coins = 0
        
        for symbol, cap_data in market_cap_data.items():
            try:
                analyzed_coins += 1
                trading_symbol = f"{symbol}/USDT"
                cap_tier = self.categorize_market_cap(cap_data['rank'])
                
                logger.info(f"📊 Analyzing {symbol} ({cap_tier}) - {analyzed_coins}/{len(market_cap_data)}")
                
                # Fetch comprehensive data
                df = self.fetch_comprehensive_data(trading_symbol, self.analysis_days)
                
                if df.empty or len(df) < 100:
                    logger.debug(f"   ⚠️ Insufficient data for {symbol}")
                    continue
                
                # Find all BB setups
                coin_setups = self.find_bb_setups_accurate(df, symbol)
                
                if coin_setups:
                    # Calculate outcomes for each setup
                    for setup in coin_setups:
                        setup_idx = df.index.get_loc(setup['timestamp'])
                        outcome = self.calculate_realistic_outcome(df, setup_idx, setup)
                        if outcome:
                            setup.update(outcome)
                            setup['market_cap_tier'] = cap_tier
                            setup['market_cap_rank'] = cap_data['rank']
                            all_setups.append(setup)
                    
                    successful_coins += 1
                    logger.info(f"   ✅ Found {len(coin_setups)} setups for {symbol}")
                else:
                    logger.debug(f"   ⚠️ No valid setups found for {symbol}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error analyzing {symbol}: {e}")
                continue
        
        logger.info(f"🎉 Analysis complete! Found {len(all_setups)} total setups from {successful_coins}/{analyzed_coins} coins")
        
        # Comprehensive analysis
        if len(all_setups) >= 100:
            results = self.analyze_comprehensive_results(all_setups)
            results['analysis_meta'] = {
                'coins_analyzed': analyzed_coins,
                'successful_coins': successful_coins,
                'total_setups': len(all_setups),
                'analysis_date': datetime.now().isoformat(),
                'analysis_days': self.analysis_days,
                'coin_limit': coin_limit
            }
            return results
        else:
            logger.warning(f"❌ Insufficient setups: only {len(all_setups)} found (need 100+)")
            return {'error': f'Insufficient setups: {len(all_setups)}', 'all_setups': all_setups}
    
    def analyze_comprehensive_results(self, all_setups: List[Dict]) -> Dict:
        """Analyze comprehensive results with proper statistical analysis"""
        results = {
            'bb_range_analysis': {},
            'indicator_analysis': {},
            'market_cap_analysis': {},
            'summary_stats': {}
        }
        
        # BB Range Analysis
        bb_ranges = [
            "0-2%", "2-5%", "5-10%", "10-15%", "15-20%", "20-25%",
            "75-80%", "80-85%", "85-90%", "90-95%", "95-98%", "98-100%"
        ]
        
        for bb_range in bb_ranges:
            range_setups = [s for s in all_setups if s['bb_range'] == bb_range]
            if len(range_setups) >= self.min_samples:
                results['bb_range_analysis'][bb_range] = self.calculate_performance_stats(range_setups)
        
        # Indicator Analysis
        indicator_analyses = {
            'mfi_oversold': lambda s: s.get('mfi_oversold', False) and s['setup_type'] == 'LONG',
            'mfi_very_oversold': lambda s: s.get('mfi_very_oversold', False) and s['setup_type'] == 'LONG',
            'mfi_overbought': lambda s: s.get('mfi_overbought', False) and s['setup_type'] == 'SHORT',
            'stoch_oversold': lambda s: s.get('stoch_oversold', False) and s['setup_type'] == 'LONG',
            'stoch_overbought': lambda s: s.get('stoch_overbought', False) and s['setup_type'] == 'SHORT',
            'volume_surge': lambda s: s.get('volume_surge', False),
            'bb_expansion': lambda s: s.get('bb_expansion', False),
            'bb_squeeze': lambda s: s.get('bb_squeeze', False),
            'bb_uptrend': lambda s: s.get('bb_trend') == 'Uptrend',
            'bb_downtrend': lambda s: s.get('bb_trend') == 'Downtrend',
        }
        
        for indicator_name, filter_func in indicator_analyses.items():
            indicator_setups = [s for s in all_setups if filter_func(s)]
            if len(indicator_setups) >= self.min_samples:
                results['indicator_analysis'][indicator_name] = self.calculate_performance_stats(indicator_setups)
        
        # Market Cap Analysis
        market_caps = ["Large Cap (Top 50)", "Mid Cap (51-100)", "Small Cap (101-200)", "Micro Cap (200+)"]
        
        for cap_tier in market_caps:
            cap_setups = [s for s in all_setups if s['market_cap_tier'] == cap_tier]
            if len(cap_setups) >= self.min_samples:
                results['market_cap_analysis'][cap_tier] = self.calculate_performance_stats(cap_setups)
                
                # Analyze indicators by market cap
                results['market_cap_analysis'][cap_tier]['indicators'] = {}
                for indicator_name, filter_func in indicator_analyses.items():
                    indicator_cap_setups = [s for s in cap_setups if filter_func(s)]
                    if len(indicator_cap_setups) >= 10:  # Lower threshold for sub-analysis
                        results['market_cap_analysis'][cap_tier]['indicators'][indicator_name] = self.calculate_performance_stats(indicator_cap_setups)
        
        # Overall summary
        successful_setups = [s for s in all_setups if s['success']]
        results['summary_stats'] = {
            'total_setups': len(all_setups),
            'successful_setups': len(successful_setups),
            'overall_success_rate': (len(successful_setups) / len(all_setups)) * 100,
            'avg_gain_all': np.mean([s['gain_pct'] for s in all_setups]),
            'avg_gain_winners': np.mean([s['gain_pct'] for s in successful_setups]) if successful_setups else 0,
            'avg_duration_hours': np.mean([s['duration_hours'] for s in all_setups])
        }
        
        return results
    
    def calculate_performance_stats(self, setups: List[Dict]) -> Dict:
        """Calculate comprehensive performance statistics"""
        if not setups:
            return {}
        
        successful = [s for s in setups if s['success']]
        failed = [s for s in setups if not s['success']]
        
        total_profit = sum([s['gain_pct'] for s in setups if s['gain_pct'] > 0])
        total_loss = abs(sum([s['gain_pct'] for s in setups if s['gain_pct'] < 0]))
        
        return {
            'count': len(setups),
            'success_count': len(successful),
            'success_rate': (len(successful) / len(setups)) * 100,
            'avg_gain_all': np.mean([s['gain_pct'] for s in setups]),
            'avg_gain_winners': np.mean([s['gain_pct'] for s in successful]) if successful else 0,
            'avg_loss_losers': np.mean([s['gain_pct'] for s in failed]) if failed else 0,
            'avg_duration': np.mean([s['duration_hours'] for s in setups]),
            'max_gain': max([s['max_favorable'] for s in setups]) if setups else 0,
            'max_loss': min([s['max_adverse'] for s in setups]) if setups else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'hit_3pct_rate': (sum([s['hit_3pct'] for s in setups]) / len(setups)) * 100,
            'hit_5pct_rate': (sum([s['hit_5pct'] for s in setups]) / len(setups)) * 100,
        }
    
    def generate_comprehensive_report(self, results: Dict) -> str:
        """Generate comprehensive analysis report"""
        if 'error' in results:
            return f"❌ Analysis failed: {results['error']}"
        
        report = []
        report.append("=" * 80)
        report.append("🎯 ACCURATE BB RANGE & INDICATOR ANALYSIS REPORT")
        report.append("=" * 80)
        
        meta = results.get('analysis_meta', {})
        summary = results.get('summary_stats', {})
        
        report.append(f"Analysis Date: {meta.get('analysis_date', 'Unknown')}")
        report.append(f"Analysis Period: {meta.get('analysis_days', 90)} days")
        report.append(f"Coins Analyzed: {meta.get('coins_analyzed', 0)}")
        report.append(f"Total Setups Found: {summary.get('total_setups', 0)}")
        report.append(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
        report.append("")
        
        # BB Range Performance
        report.append("📊 BB RANGE PERFORMANCE (ACCURATE TARGETS):")
        report.append("-" * 85)
        report.append("Range     | Count | Success% | Avg Gain | Winners | Duration | 3%Hit | 5%Hit | PF")
        report.append("-" * 85)
        
        bb_analysis = results.get('bb_range_analysis', {})
        for range_name, stats in bb_analysis.items():
            report.append(f"{range_name:9} | {stats['count']:5d} | "
                         f"{stats['success_rate']:7.1f}% | {stats['avg_gain_all']:8.2f}% | "
                         f"{stats['avg_gain_winners']:7.2f}% | {stats['avg_duration']:8.1f}h | "
                         f"{stats['hit_3pct_rate']:5.1f}% | {stats['hit_5pct_rate']:5.1f}% | {stats['profit_factor']:4.1f}")
        
        report.append("")
        
        # Indicator Performance
        report.append("🎯 INDICATOR PERFORMANCE ANALYSIS:")
        report.append("-" * 75)
        report.append("Indicator            | Count | Success% | Avg Gain | Duration | PF")
        report.append("-" * 75)
        
        indicator_analysis = results.get('indicator_analysis', {})
        for indicator_name, stats in indicator_analysis.items():
            display_name = indicator_name.replace('_', ' ').title()
            report.append(f"{display_name:20} | {stats['count']:5d} | "
                         f"{stats['success_rate']:7.1f}% | {stats['avg_gain_all']:8.2f}% | "
                         f"{stats['avg_duration']:8.1f}h | {stats['profit_factor']:4.1f}")
        
        report.append("")
        
        # Market Cap Analysis
        report.append("🏆 MARKET CAP PERFORMANCE:")
        report.append("-" * 50)
        
        market_cap_analysis = results.get('market_cap_analysis', {})
        for cap_tier, stats in market_cap_analysis.items():
            report.append(f"\n{cap_tier}:")
            report.append(f"  Overall Performance: {stats['success_rate']:.1f}% ({stats['count']} setups)")
            report.append(f"  Average Gain: {stats['avg_gain_all']:.2f}%")
            report.append(f"  Profit Factor: {stats['profit_factor']:.1f}")
            
            # Top indicators for this market cap
            if 'indicators' in stats:
                top_indicators = sorted(stats['indicators'].items(), 
                                      key=lambda x: x[1]['success_rate'], reverse=True)[:3]
                if top_indicators:
                    report.append("  Top Indicators:")
                    for ind_name, ind_stats in top_indicators:
                        display_name = ind_name.replace('_', ' ').title()
                        report.append(f"    {display_name}: {ind_stats['success_rate']:.1f}% ({ind_stats['count']} setups)")
        
        # Optimal Recommendations
        report.append("\n" + "=" * 80)
        report.append("🎯 STRATEGIC RECOMMENDATIONS:")
        report.append("=" * 80)
        
        # Find best BB ranges
        if bb_analysis:
            best_long_ranges = {k: v for k, v in bb_analysis.items() 
                              if any(x in k for x in ['0-', '2-', '5-', '10-', '15-', '20-']) and v['count'] >= 20}
            best_short_ranges = {k: v for k, v in bb_analysis.items() 
                               if any(x in k for x in ['75-', '80-', '85-', '90-', '95-', '98-']) and v['count'] >= 20}
            
            if best_long_ranges:
                best_long = max(best_long_ranges.items(), key=lambda x: x[1]['success_rate'])
                report.append(f"🟢 OPTIMAL LONG BB RANGE: {best_long[0]}")
                report.append(f"   Success Rate: {best_long[1]['success_rate']:.1f}%")
                report.append(f"   Sample Size: {best_long[1]['count']} setups")
                report.append(f"   Recommended Threshold: bb_pct <= {float(best_long[0].split('-')[1].rstrip('%'))/100:.2f}")
            
            if best_short_ranges:
                best_short = max(best_short_ranges.items(), key=lambda x: x[1]['success_rate'])
                report.append(f"🔴 OPTIMAL SHORT BB RANGE: {best_short[0]}")
                report.append(f"   Success Rate: {best_short[1]['success_rate']:.1f}%")
                report.append(f"   Sample Size: {best_short[1]['count']} setups")
                report.append(f"   Recommended Threshold: bb_pct >= {float(best_short[0].split('-')[0])/100:.2f}")
        
        # Best indicators
        if indicator_analysis:
            top_indicators = sorted(indicator_analysis.items(), 
                                  key=lambda x: x[1]['success_rate'], reverse=True)[:5]
            report.append(f"\n🏆 TOP PERFORMING INDICATORS:")
            for i, (ind_name, stats) in enumerate(top_indicators, 1):
                display_name = ind_name.replace('_', ' ').title()
                report.append(f"   {i}. {display_name}: {stats['success_rate']:.1f}% ({stats['count']} setups)")
        
        # Market cap recommendation
        if market_cap_analysis:
            best_cap = max(market_cap_analysis.items(), key=lambda x: x[1]['success_rate'])
            report.append(f"\n📊 FOCUS ON: {best_cap[0]}")
            report.append(f"   Best Overall Performance: {best_cap[1]['success_rate']:.1f}%")
            
            if "Small Cap" in best_cap[0]:
                report.append("   💡 RECOMMENDATION: Set CMC limit to 200 for small cap focus")
            elif "Large Cap" in best_cap[0]:
                report.append("   💡 RECOMMENDATION: Focus on top 50 coins for stability")
            elif "Micro Cap" in best_cap[0]:
                report.append("   💡 RECOMMENDATION: Set CMC limit to 300+ for micro cap opportunities")
        
        return "\n".join(report)
    
    def save_comprehensive_results(self, results: Dict, filename: str = None):
        """Save comprehensive results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bb_accurate_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Accurate results saved to {filename}")

def main():
    """Main execution function"""
    analyzer = BBRangeAnalyzerAccurate()
    
    print("🎯 BB Accurate Analysis Tool - Historical Intelligence Methodology")
    print("=" * 75)
    print("📊 Uses REALISTIC profit targets (3-5%) instead of unrealistic BB middle targets")
    print("🔍 Matches your Historical Intelligence confluence and filtering methodology")
    print("📈 Provides accurate indicator performance and optimal threshold recommendations")
    print("")
    
    try:
        coin_limit = int(input("Enter number of coins to analyze (100-300): ") or "200")
        coin_limit = max(100, min(300, coin_limit))
    except:
        coin_limit = 200
    
    print(f"\n🚀 Starting accurate analysis of top {coin_limit} coins...")
    print("⏱️  This will take 20-30 minutes for comprehensive analysis")
    print("📊 Using realistic 3-5% profit targets and proper confluence filtering...")
    
    # Run accurate analysis
    results = analyzer.run_comprehensive_analysis(coin_limit)
    
    if not results or 'error' in results:
        print("❌ Analysis failed or insufficient data")
        if 'error' in results:
            print(f"Error: {results['error']}")
        return
    
    # Generate and display report
    report = analyzer.generate_comprehensive_report(results)
    print("\n" + report)
    
    # Save results
    analyzer.save_comprehensive_results(results)
    
    print(f"\n✅ Accurate analysis complete!")
    print("🎯 Results show REALISTIC BB performance with proper targets and confluence")
    print("📊 Use these findings to optimize your BB detector settings and thresholds")

if __name__ == "__main__":
    main()