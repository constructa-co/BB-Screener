#!/usr/bin/env python3
"""
ICT Enhanced Backtester - Phase 19: Hybrid Scoring System
Combines R17's quality with R18's breadth, but adds intelligent ranking
instead of binary exclusion for actionable daily scanning.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class ICTBacktester:
    def __init__(self, config=None):
        # R19: MAINTAIN R17'S EXCELLENT WIN RATE WITH MORE OPPORTUNITIES
        self.config = config or {
            # R19: KEEP R17's excellent filters (86.8% win rate)
            'min_order_block_size': 1.8,   # R19: R17's excellent filter
            'min_volume_ratio': 2.2,       # R19: R17's excellent filter  
            'min_quality_score': 65,       # R19: R17's excellent filter
            'min_fvg_size': 0.5,           # R19: R17's excellent filter
            'max_distance_pct': 18.0,
            'max_ob_age': 40,
            'min_confluence_factors': 3,
            'min_risk_reward': 0.8,
            
            # R19: ENHANCED SCORING SYSTEM (for ranking multiple trades)
            'use_scoring_system': True,     # R19: NEW - Enable scoring for ranking
            'high_confidence_threshold': 85, # R19: NEW - Score for high confidence trades
            'medium_confidence_threshold': 70, # R19: NEW - Score for medium confidence trades
            'min_trade_score': 60,         # R19: NEW - Maintain quality threshold
            
            # R19: KEEP R17's excellent Fibonacci targets
            'fibonacci_levels': [0.50, 0.75, 1.00], # R19: R17's excellent levels
            
            # R19: EXPANDED TOKEN SELECTION (36 tokens - more opportunities)
            'expanded_tokens': [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
                'ARB/USDT', 'OP/USDT', 'SHIB/USDT', 'PEPE/USDT',
                'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT',
                'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT',
                'XRP/USDT', 'DOGE/USDT', 'TRX/USDT', 'ETC/USDT',
                'FIL/USDT', 'NEAR/USDT', 'ALGO/USDT', 'VET/USDT',
                'ICP/USDT', 'FLOW/USDT', 'THETA/USDT', 'XTZ/USDT',
                'AAVE/USDT', 'SUSHI/USDT', 'COMP/USDT', 'MKR/USDT',
                'SNX/USDT', 'CRV/USDT', 'YFI/USDT', 'BAL/USDT'
            ]
        }
        
        # Use public data access instead of API keys
        self.exchange = ccxt.binance({
            'enableRateLimit': True
        })
        
        self.trades = []
        self.category_stats = {}
        
    def calculate_trade_confidence_score(self, setup_type, quality_score, confluence_factors, 
                                       volume_ratio, gap_size=None, category=None):
        """R19: NEW - Calculate confidence score instead of binary exclusion"""
        base_score = 50
        
        # Setup type scoring
        if setup_type == 'fvg':
            base_score += 25  # FVGs are more reliable
        elif setup_type == 'order_block':
            base_score += 15
        
        # Quality score contribution
        if quality_score >= 90:
            base_score += 20
        elif quality_score >= 80:
            base_score += 15
        elif quality_score >= 70:
            base_score += 10
        elif quality_score >= 60:
            base_score += 5
        
        # Confluence factors
        base_score += len(confluence_factors) * 5
        
        # Volume ratio bonus
        if volume_ratio >= 3.0:
            base_score += 10
        elif volume_ratio >= 2.5:
            base_score += 8
        elif volume_ratio >= 2.0:
            base_score += 5
        
        # FVG gap size bonus
        if gap_size and gap_size > 1.0:
            base_score += 5
        
        # Category weighting
        if category == 'Layer 2/Scaling':
            base_score += 5
        elif category == 'Meme/Community':
            base_score += 3
        elif category == 'Major Cryptos':
            base_score += 2
        
        return min(base_score, 100)  # Cap at 100
    
    def calculate_enhanced_fibonacci_targets(self, df, entry_price, direction):
        """R19: KEEP R17's excellent Fibonacci targets (50%, 75%, 100%)"""
        try:
            # Find recent swing levels
            window = 20
            recent_highs = df['high'].rolling(window=window).max()
            recent_lows = df['low'].rolling(window=window).min()
            
            swing_high = recent_highs.iloc[-1]
            swing_low = recent_lows.iloc[-1]
            
            if direction == 'BULLISH':
                # Calculate Fibonacci extensions from swing low to swing high (R19: R17's excellent levels)
                range_size = swing_high - swing_low
                fib_50 = swing_low + (range_size * 0.50)  # 50% extension (R17's excellent level)
                fib_75 = swing_low + (range_size * 0.75)  # 75% extension (R17's excellent level)
                fib_100 = swing_low + (range_size * 1.00) # 100% extension (R17's excellent level)
                return [fib_50, fib_75, fib_100], ["Fib 50%", "Fib 75%", "Fib 100%"]
            else:
                # Calculate Fibonacci extensions from swing high to swing low (R19: R17's excellent levels)
                range_size = swing_high - swing_low
                fib_50 = swing_high - (range_size * 0.50)  # 50% extension (R17's excellent level)
                fib_75 = swing_high - (range_size * 0.75)  # 75% extension (R17's excellent level)
                fib_100 = swing_high - (range_size * 1.00) # 100% extension (R17's excellent level)
                return [fib_50, fib_75, fib_100], ["Fib 50%", "Fib 75%", "Fib 100%"]
        except:
            # Fallback to percentage targets
            if direction == 'BULLISH':
                targets = [entry_price * 1.008, entry_price * 1.015, entry_price * 1.025]
                reasons = ["T1: +0.8%", "T2: +1.5%", "T3: +2.5%"]
            else:
                targets = [entry_price * 0.992, entry_price * 0.985, entry_price * 0.975]
                reasons = ["T1: -0.8%", "T2: -1.5%", "T3: -2.5%"]
            return targets, reasons
    
    def calculate_ict_targets(self, df, entry_price, direction, setup_type='order_block'):
        """R19: Enhanced target calculation with optimized Fibonacci levels"""
        # Try enhanced Fibonacci targets first
        targets, reasons = self.calculate_enhanced_fibonacci_targets(df, entry_price, direction)
        if targets:
            return targets[:3], reasons[:3]
        
        # Fallback to percentage targets
        if direction == 'BULLISH':
            targets = [entry_price * 1.008, entry_price * 1.015, entry_price * 1.025]
            target_reasons = ["T1: +0.8%", "T2: +1.5%", "T3: +2.5%"]
        else:
            targets = [entry_price * 0.992, entry_price * 0.985, entry_price * 0.975]
            target_reasons = ["T1: -0.8%", "T2: -1.5%", "T3: -2.5%"]
        
        return targets[:3], target_reasons[:3]

    # ... (keep all other methods from R18, but update detect_historical_setups_enhanced)
    
    def detect_historical_setups_enhanced(self, df, future_df, symbol):
        """R19: Enhanced setup detection with scoring system instead of binary exclusion"""
        # FIX: Define ALL variables that might be used in debug messages
        min_quality = self.config.get('min_quality_score', 55)
        min_confluence = self.config.get('min_confluence_factors', 3)
        quality_filtered = 0
        distance_filtered = 0
        score_filtered = 0
        
        # Category detection for weighting
        category = self.get_token_category(symbol)
        
        # Enhanced market regime filter
        if self.config.get('use_enhanced_regime', True):
            regime = self.detect_market_regime(df)
            if self.config.get('require_market_regime', False) and not regime['favorable']:
                return []
        
        trades = []
        
        # Process Order Blocks with R19 scoring
        obs = self.detect_order_blocks(df)
        for ob in obs:
            if ob['age'] > self.config.get('max_ob_age', 40):
                continue
                
            # Calculate entry price and direction
            if ob['type'] == 'bullish':
                entry_price = ob['high'] + (ob['high'] - ob['low']) * 0.25
                direction = 'BULLISH'
            else:
                entry_price = ob['low'] - (ob['high'] - ob['low']) * 0.25
                direction = 'BEARISH'
            
            # Calculate stop loss
            if ob['type'] == 'bullish':
                stop_loss = entry_price * 0.985  # 1.5% stop
            else:
                stop_loss = entry_price * 1.015  # 1.5% stop
            
            # Calculate targets
            targets, target_reasons = self.calculate_ict_targets(df, entry_price, direction, 'order_block')
            
            # R19: NEW - Calculate confidence score instead of binary filtering
            quality_score = self.calculate_setup_quality(df, ob, 'order_block')
            confluence_factors = self.calculate_indicator_confluence(df, ob)
            
            # Calculate trade confidence score
            confidence_score = self.calculate_trade_confidence_score(
                'order_block', quality_score, confluence_factors, 
                ob.get('volume_ratio', 1.0), None, category
            )
            
            # R19: NEW - Use scoring threshold instead of binary exclusion
            if confidence_score >= self.config.get('min_trade_score', 50):
                trade = {
                    'symbol': symbol,
                    'setup_type': 'order_block',
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'target_reasons': target_reasons,
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'confluence_factors': confluence_factors,
                    'category': category,
                    'volume_ratio': ob.get('volume_ratio', 1.0),
                    'ob_age': ob['age'],
                    'ob_size': ob['size']
                }
                trades.append(trade)
            else:
                score_filtered += 1
        
        # Process Fair Value Gaps with R19 scoring
        fvgs = self.detect_fair_value_gaps(df)
        for fvg in fvgs:
            if fvg['age'] > self.config.get('max_fvg_age', 30):
                continue
            
            # Calculate entry price and direction
            if fvg['type'] == 'bullish':
                entry_price = fvg['low'] + (fvg['high'] - fvg['low']) * 0.5
                direction = 'BULLISH'
            else:
                entry_price = fvg['high'] - (fvg['high'] - fvg['low']) * 0.5
                direction = 'BEARISH'
            
            # Calculate stop loss
            if fvg['type'] == 'bullish':
                stop_loss = entry_price * 0.985  # 1.5% stop
            else:
                stop_loss = entry_price * 1.015  # 1.5% stop
            
            # Calculate targets
            targets, target_reasons = self.calculate_ict_targets(df, entry_price, direction, 'fvg')
            
            # R19: NEW - Calculate confidence score for FVGs
            quality_score = self.calculate_setup_quality(df, fvg, 'fvg')
            confluence_factors = self.calculate_indicator_confluence(df, fvg)
            
            # Calculate trade confidence score
            confidence_score = self.calculate_trade_confidence_score(
                'fvg', quality_score, confluence_factors,
                fvg.get('volume_ratio', 1.0), fvg.get('gap_size', 0), category
            )
            
            # R19: NEW - Use scoring threshold instead of binary exclusion
            if confidence_score >= self.config.get('min_trade_score', 50):
                trade = {
                    'symbol': symbol,
                    'setup_type': 'fvg',
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'target_reasons': target_reasons,
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'confluence_factors': confluence_factors,
                    'category': category,
                    'gap_size': fvg.get('gap_size', 0),
                    'fvg_age': fvg['age']
                }
                trades.append(trade)
            else:
                score_filtered += 1
        
        return trades

    def run_enhanced_backtest(self, symbols, days_back=30):
        """R19: Enhanced backtest with confidence scoring"""
        print(f"\n🔧 PHASE 19 HYBRID SCORING SYSTEM:")
        print(f" • Order Block: {self.config.get('min_order_block_size')}% moves")
        print(f" • Volume Ratio: {self.config.get('min_volume_ratio')}x minimum")
        print(f" • Quality Score: {self.config.get('min_quality_score')} minimum")
        print(f" • Confidence Threshold: {self.config.get('min_trade_score')} minimum")
        print(f" • Enhanced Fibonacci: {self.config.get('fibonacci_levels')}")
        
        all_trades = []
        
        for symbol in symbols:
            try:
                print(f"📊 Backtesting {symbol}...", end="")
                
                # Fetch data
                ohlcv = self.exchange.fetch_ohlcv(symbol, '4h', limit=200)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                # Detect setups with R19 scoring
                trades = self.detect_historical_setups_enhanced(df, df, symbol)
                
                if trades:
                    print(f"   Found {len(trades)} historical trades")
                    all_trades.extend(trades)
                else:
                    print(f"   No trades found")
                    
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        # R19: NEW - Sort trades by confidence score
        all_trades.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Simulate trade outcomes
        for trade in all_trades:
            outcome = self.simulate_trade_outcome_enhanced(trade, df)
            trade.update(outcome)
        
        # Analyze results with confidence scoring
        self.analyze_enhanced_performance_with_scoring(all_trades)
        
        return all_trades
    
    def analyze_enhanced_performance_with_scoring(self, trades):
        """R19: NEW - Enhanced analysis with confidence scoring breakdown"""
        if not trades:
            print("❌ No trades found")
            return
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('win', False)]
        losing_trades = [t for t in trades if not t.get('win', False)]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_return = sum(t.get('return_pct', 0) for t in trades)
        avg_return = total_return / total_trades if total_trades > 0 else 0
        
        # R19: NEW - Confidence score analysis
        high_confidence_trades = [t for t in trades if t['confidence_score'] >= 85]
        medium_confidence_trades = [t for t in trades if 70 <= t['confidence_score'] < 85]
        low_confidence_trades = [t for t in trades if t['confidence_score'] < 70]
        
        print(f"\n🎯 PHASE 19 HYBRID SCORING SYSTEM RESULTS:")
        print(f"==================================================")
        print(f"📊 Total Trades: {total_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}%")
        print(f"💰 Total Return: {total_return:.1f}%")
        print(f"📈 Average Return per Trade: {avg_return:.2f}%")
        
        # R19: NEW - Confidence breakdown
        print(f"\n🎯 CONFIDENCE SCORE BREAKDOWN:")
        print(f"   High Confidence (85+): {len(high_confidence_trades)} trades")
        print(f"   Medium Confidence (70-84): {len(medium_confidence_trades)} trades")
        print(f"   Low Confidence (<70): {len(low_confidence_trades)} trades")
        
        if high_confidence_trades:
            high_win_rate = len([t for t in high_confidence_trades if t.get('win', False)]) / len(high_confidence_trades) * 100
            print(f"   High Confidence Win Rate: {high_win_rate:.1f}%")
        
        if medium_confidence_trades:
            medium_win_rate = len([t for t in medium_confidence_trades if t.get('win', False)]) / len(medium_confidence_trades) * 100
            print(f"   Medium Confidence Win Rate: {medium_win_rate:.1f}%")
        
        # Setup type analysis
        fvg_trades = [t for t in trades if t['setup_type'] == 'fvg']
        ob_trades = [t for t in trades if t['setup_type'] == 'order_block']
        
        if fvg_trades:
            fvg_win_rate = len([t for t in fvg_trades if t.get('win', False)]) / len(fvg_trades) * 100
            print(f"\n📊 FVG Performance: {len(fvg_trades)} trades, {fvg_win_rate:.1f}% win rate")
        
        if ob_trades:
            ob_win_rate = len([t for t in ob_trades if t.get('win', False)]) / len(ob_trades) * 100
            print(f"📊 Order Block Performance: {len(ob_trades)} trades, {ob_win_rate:.1f}% win rate")
        
        # R19: NEW - Top trades by confidence
        print(f"\n🏆 TOP 5 TRADES BY CONFIDENCE SCORE:")
        for i, trade in enumerate(trades[:5]):
            win_status = "✅ WIN" if trade.get('win', False) else "❌ LOSS"
            print(f"   {i+1}. {trade['symbol']} ({trade['setup_type']}) - "
                  f"Score: {trade['confidence_score']}, {win_status}, "
                  f"Return: {trade.get('return_pct', 0):.2f}%")

    def add_technical_indicators(self, df):
        """Add ICT-specific technical indicators"""
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_trend'] = df['volume_sma'] / df['volume_sma'].shift(20)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        money_ratio = positive_flow / negative_flow
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # CMF
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # A/D Line
        ad_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        ad_multiplier = ad_multiplier.fillna(0)
        ad_volume = ad_multiplier * df['volume']
        df['accumulation_distribution'] = ad_volume.cumsum()
        
        # OBV
        obv = []
        obv_value = 0
        for i in range(len(df)):
            if i == 0:
                obv_value = df.iloc[i]['volume']
            else:
                if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                    obv_value += df.iloc[i]['volume']
                elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
                    obv_value -= df.iloc[i]['volume']
            obv.append(obv_value)
        df['on_balance_volume'] = obv
        
        # Williams %R
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # VWAP
        typical_price_vwap = (df['high'] + df['low'] + df['close']) / 3
        vwap_numerator = (typical_price_vwap * df['volume']).rolling(window=20).sum()
        vwap_denominator = df['volume'].rolling(window=20).sum()
        df['vwap'] = vwap_numerator / vwap_denominator
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # CCI
        cci_tp = (df['high'] + df['low'] + df['close']) / 3
        cci_sma = cci_tp.rolling(window=20).mean()
        cci_mad = cci_tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (cci_tp - cci_sma) / (0.015 * cci_mad)
        
        return df
    
    def detect_market_regime(self, df):
        """Enhanced market regime detection"""
        # Simple regime detection
        regime = {
            'volatility': 'normal',
            'trend': 'weak',
            'volume': 'normal',
            'structure': 'neutral',
            'favorable': True  # Default to favorable for testing
        }
        return regime
    
    def detect_order_blocks(self, df):
        """Detect ICT Order Blocks"""
        order_blocks = []
        
        for i in range(1, len(df) - 1):
            # Bullish Order Block
            if (df.iloc[i]['close'] > df.iloc[i]['open'] and  # Bullish candle
                df.iloc[i+1]['low'] < df.iloc[i]['low'] and    # Next candle breaks low
                df.iloc[i+1]['close'] > df.iloc[i]['high']):   # But closes above high
                
                size = (df.iloc[i]['high'] - df.iloc[i]['low']) / df.iloc[i]['low'] * 100
                volume_ratio = df.iloc[i]['volume'] / df.iloc[i-20:i]['volume'].mean()
                
                if size >= self.config.get('min_order_block_size', 1.4):
                    order_blocks.append({
                        'type': 'bullish',
                        'high': df.iloc[i]['high'],
                        'low': df.iloc[i]['low'],
                        'size': size,
                        'volume_ratio': volume_ratio,
                        'age': len(df) - i - 1
                    })
            
            # Bearish Order Block
            elif (df.iloc[i]['close'] < df.iloc[i]['open'] and  # Bearish candle
                  df.iloc[i+1]['high'] > df.iloc[i]['high'] and  # Next candle breaks high
                  df.iloc[i+1]['close'] < df.iloc[i]['low']):    # But closes below low
                
                size = (df.iloc[i]['high'] - df.iloc[i]['low']) / df.iloc[i]['low'] * 100
                volume_ratio = df.iloc[i]['volume'] / df.iloc[i-20:i]['volume'].mean()
                
                if size >= self.config.get('min_order_block_size', 1.4):
                    order_blocks.append({
                        'type': 'bearish',
                        'high': df.iloc[i]['high'],
                        'low': df.iloc[i]['low'],
                        'size': size,
                        'volume_ratio': volume_ratio,
                        'age': len(df) - i - 1
                    })
        
        return order_blocks
    
    def detect_fair_value_gaps(self, df):
        """Detect ICT Fair Value Gaps"""
        fvgs = []
        
        for i in range(1, len(df) - 1):
            # Bullish FVG
            if (df.iloc[i]['low'] > df.iloc[i-1]['high']):
                gap_size = (df.iloc[i]['low'] - df.iloc[i-1]['high']) / df.iloc[i-1]['high'] * 100
                if gap_size >= self.config.get('min_fvg_size', 0.3):
                    fvgs.append({
                        'type': 'bullish',
                        'low': df.iloc[i]['low'],
                        'high': df.iloc[i-1]['high'],
                        'gap_size': gap_size,
                        'age': len(df) - i - 1
                    })
            
            # Bearish FVG
            elif (df.iloc[i]['high'] < df.iloc[i-1]['low']):
                gap_size = (df.iloc[i-1]['low'] - df.iloc[i]['high']) / df.iloc[i-1]['low'] * 100
                if gap_size >= self.config.get('min_fvg_size', 0.3):
                    fvgs.append({
                        'type': 'bearish',
                        'low': df.iloc[i-1]['low'],
                        'high': df.iloc[i]['high'],
                        'gap_size': gap_size,
                        'age': len(df) - i - 1
                    })
        
        return fvgs
    
    def calculate_setup_quality(self, df, setup, setup_type):
        """Calculate quality score for setup"""
        base_score = 50
        
        # Volume ratio contribution
        volume_ratio = setup.get('volume_ratio', 1.0)
        if volume_ratio >= 3.0:
            base_score += 20
        elif volume_ratio >= 2.5:
            base_score += 15
        elif volume_ratio >= 2.0:
            base_score += 10
        elif volume_ratio >= 1.5:
            base_score += 5
        
        # Setup size contribution
        if setup_type == 'order_block':
            size = setup.get('size', 0)
            if size >= 2.0:
                base_score += 15
            elif size >= 1.5:
                base_score += 10
            elif size >= 1.0:
                base_score += 5
        
        # Age contribution (fresher is better)
        age = setup.get('age', 0)
        if age <= 10:
            base_score += 15
        elif age <= 20:
            base_score += 10
        elif age <= 30:
            base_score += 5
        
        return min(base_score, 100)
    
    def calculate_indicator_confluence(self, df, setup):
        """Calculate indicator confluence factors"""
        confluence_factors = []
        
        # Get latest values
        rsi = df['rsi'].iloc[-1]
        mfi = df['mfi'].iloc[-1]
        cmf = df['cmf'].iloc[-1]
        
        # RSI confluence
        if setup['type'] == 'bullish' and rsi < 40:
            confluence_factors.append("Oversold RSI")
        elif setup['type'] == 'bearish' and rsi > 60:
            confluence_factors.append("Overbought RSI")
        
        # MFI confluence
        if setup['type'] == 'bullish' and mfi < 30:
            confluence_factors.append("Oversold MFI")
        elif setup['type'] == 'bearish' and mfi > 70:
            confluence_factors.append("Overbought MFI")
        
        # CMF confluence
        if setup['type'] == 'bullish' and cmf > 0.1:
            confluence_factors.append("Positive CMF")
        elif setup['type'] == 'bearish' and cmf < -0.1:
            confluence_factors.append("Negative CMF")
        
        return confluence_factors
    
    def get_token_category(self, symbol):
        """Get token category for weighting"""
        major_cryptos = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT']
        layer2 = ['ARB/USDT', 'OP/USDT', 'MATIC/USDT']
        meme = ['SHIB/USDT', 'PEPE/USDT', 'DOGE/USDT']
        
        if symbol in major_cryptos:
            return 'Major Cryptos'
        elif symbol in layer2:
            return 'Layer 2/Scaling'
        elif symbol in meme:
            return 'Meme/Community'
        else:
            return 'Altcoins'
    
    def simulate_trade_outcome_enhanced(self, trade, future_df):
        """Simulate trade outcome with enhanced analysis"""
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        targets = trade['targets']
        
        # Find exit conditions
        exit_price = entry_price
        exit_reason = "No exit"
        target_hit = False
        stop_hit = False
        
        for i, row in future_df.iterrows():
            current_price = row['close']
            
            # Check stop loss
            if trade['direction'] == 'BULLISH':
                if current_price <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss"
                    stop_hit = True
                    break
            else:
                if current_price >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss"
                    stop_hit = True
                    break
            
            # Check targets
            for j, target in enumerate(targets):
                if trade['direction'] == 'BULLISH':
                    if current_price >= target:
                        exit_price = target
                        exit_reason = f"Target {j+1}"
                        target_hit = True
                        break
                else:
                    if current_price <= target:
                        exit_price = target
                        exit_reason = f"Target {j+1}"
                        target_hit = True
                        break
            
            if target_hit:
                break
        
        # Calculate returns
        if trade['direction'] == 'BULLISH':
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100
        
        win = return_pct > 0
        
        return {
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'return_pct': return_pct,
            'win': win,
            'target_hit': target_hit,
            'stop_hit': stop_hit
        }

def main():
    backtester = ICTBacktester()
    
    print("🚀 ICT PHASE 19 - MAINTAIN R17 WIN RATE WITH MORE OPPORTUNITIES")
    print("======================================================================")
    print("🎯 STRATEGY: Keep 86.8% win rate, expand token selection")
    print("📊 APPROACH: R17's excellent filters + 36 tokens + scoring")
    print("🔧 FEATURES: R17's Fibonacci targets + Confidence ranking")
    print("======================================================================")
    
    print("\nSelect optimization mode:")
    print("1. Test Phase 19 on 36 expanded tokens (quick)")
    print("2. Test Phase 19 on all 89 tokens")
    print("3. Phase 19 with confidence score analysis")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        print("\n🚀 TESTING PHASE 19 ON 36 EXPANDED TOKENS")
        print("=" * 70)
        
        # R19: Expanded token selection (36 tokens - more opportunities)
        test_symbols = backtester.config.get('expanded_tokens', [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'ARB/USDT', 'OP/USDT', 'SHIB/USDT', 'PEPE/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT',
            'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT',
            'XRP/USDT', 'DOGE/USDT', 'TRX/USDT', 'ETC/USDT',
            'FIL/USDT', 'NEAR/USDT', 'ALGO/USDT', 'VET/USDT',
            'ICP/USDT', 'FLOW/USDT', 'THETA/USDT', 'XTZ/USDT',
            'AAVE/USDT', 'SUSHI/USDT', 'COMP/USDT', 'MKR/USDT',
            'SNX/USDT', 'CRV/USDT', 'YFI/USDT', 'BAL/USDT'
        ])
        
        print(f"\n🎯 PHASE 19 MAINTAIN R17 WIN RATE RESULTS:")
        trades = backtester.run_enhanced_backtest(test_symbols)
        
    elif choice == "2":
        print("\n🚀 TESTING PHASE 19 HYBRID SCORING SYSTEM ON ALL 89 TOKENS")
        print("=" * 70)
        
        # All 89 tokens
        all_symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
            # ... (add all 89 tokens)
        ]
        
        print(f"\n🎯 PHASE 19 HYBRID SCORING SYSTEM SUMMARY:")
        trades = backtester.run_enhanced_backtest(all_symbols)
        
    elif choice == "3":
        print("\n🔍 PHASE 19 CONFIDENCE SCORE ANALYSIS")
        print("=" * 50)
        
        # Run with detailed confidence analysis
        test_symbols = backtester.config.get('focused_tokens', [])
        trades = backtester.run_enhanced_backtest(test_symbols)
        
        # Additional confidence analysis
        if trades:
            print(f"\n📊 CONFIDENCE SCORE DISTRIBUTION:")
            scores = [t['confidence_score'] for t in trades]
            print(f"   Average Score: {np.mean(scores):.1f}")
            print(f"   Median Score: {np.median(scores):.1f}")
            print(f"   Min Score: {min(scores)}")
            print(f"   Max Score: {max(scores)}")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 