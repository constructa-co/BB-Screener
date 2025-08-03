#!/usr/bin/env python3
"""
ICT Scanner with Enhanced Backtesting & Category Analysis
Combines ICT Order Block detection with comprehensive historical performance analysis

Run this file to:
1. Test historical ICT performance across 85+ tokens
2. Find best technical indicators for ICT setups
3. Analyze performance by token categories
4. Optimize confluence factors
5. Generate comprehensive performance reports
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class ICTScanner:
    """Original ICT Scanner class - simplified for backtesting integration"""
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'rateLimit': 50,
            'enableRateLimit': True,
        })
        
        # ICT 4H Parameters
        self.lookback_periods = 30
        self.min_order_block_size = 3.0
        self.min_liquidity_sweep = 1.0
        self.min_volume_ratio = 2.0
        self.max_order_block_age = 15
        self.min_quality_score = 85

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(window=period).mean()

    def detect_order_blocks(self, df):
        """Detect ICT Order Blocks"""
        order_blocks = []
        
        for i in range(5, len(df) - 5):
            current_candle = df.iloc[i]
            move_threshold = self.min_order_block_size
            
            # Bullish Order Block
            if (current_candle['close'] < current_candle['open'] and
                current_candle['volume_ratio'] >= self.min_volume_ratio):
                
                future_moves = []
                for j in range(1, 6):
                    if i + j < len(df):
                        future_price = df.iloc[i + j]['close']
                        move_pct = (future_price - current_candle['close']) / current_candle['close'] * 100
                        future_moves.append(move_pct)
                
                max_future_move = max(future_moves) if future_moves else 0
                
                if max_future_move >= move_threshold:
                    order_blocks.append({
                        'type': 'bullish',
                        'index': i,
                        'timestamp': current_candle['timestamp'],
                        'high': current_candle['high'],
                        'low': current_candle['low'],
                        'open': current_candle['open'],
                        'close': current_candle['close'],
                        'volume_ratio': current_candle['volume_ratio'],
                        'future_move': max_future_move,
                        'age': len(df) - i - 1
                    })
            
            # Bearish Order Block
            elif (current_candle['close'] > current_candle['open'] and
                  current_candle['volume_ratio'] >= self.min_volume_ratio):
                
                future_moves = []
                for j in range(1, 6):
                    if i + j < len(df):
                        future_price = df.iloc[i + j]['close']
                        move_pct = (current_candle['close'] - future_price) / current_candle['close'] * 100
                        future_moves.append(move_pct)
                
                max_future_move = max(future_moves) if future_moves else 0
                
                if max_future_move >= move_threshold:
                    order_blocks.append({
                        'type': 'bearish',
                        'index': i,
                        'timestamp': current_candle['timestamp'],
                        'high': current_candle['high'],
                        'low': current_candle['low'],
                        'open': current_candle['open'],
                        'close': current_candle['close'],
                        'volume_ratio': current_candle['volume_ratio'],
                        'future_move': max_future_move,
                        'age': len(df) - i - 1
                    })
        
        valid_order_blocks = [ob for ob in order_blocks if ob['age'] <= self.max_order_block_age]
        return valid_order_blocks

    def detect_liquidity_grabs(self, df):
        """Detect liquidity grabs"""
        liquidity_grabs = []
        
        for i in range(10, len(df) - 5):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            lookback_data = df.iloc[i-10:i]
            recent_high = lookback_data['high'].max()
            recent_low = lookback_data['low'].min()
            
            # Buy-Side Liquidity Grab
            if current_high > recent_high * 1.001:
                for j in range(1, 6):
                    if i + j < len(df):
                        future_low = df.iloc[i + j]['low']
                        move_down = (current_high - future_low) / current_high * 100
                        
                        if move_down >= self.min_liquidity_sweep:
                            liquidity_grabs.append({
                                'type': 'buy_side_grab',
                                'index': i,
                                'timestamp': df.iloc[i]['timestamp'],
                                'grab_level': current_high,
                                'reversal_move': move_down
                            })
                            break
            
            # Sell-Side Liquidity Grab
            elif current_low < recent_low * 0.999:
                for j in range(1, 6):
                    if i + j < len(df):
                        future_high = df.iloc[i + j]['high']
                        move_up = (future_high - current_low) / current_low * 100
                        
                        if move_up >= self.min_liquidity_sweep:
                            liquidity_grabs.append({
                                'type': 'sell_side_grab',
                                'index': i,
                                'timestamp': df.iloc[i]['timestamp'],
                                'grab_level': current_low,
                                'reversal_move': move_up
                            })
                            break
        
        return liquidity_grabs

    def create_basic_confluences(self, df, order_blocks, liquidity_grabs):
        """Create basic ICT confluences for backtesting"""
        confluences = []
        current_price = df.iloc[-1]['close']
        
        for ob in order_blocks:
            setup_score = 50
            confluence_factors = []
            
            # Basic scoring
            if ob['volume_ratio'] >= 3.0:
                setup_score += 25
                confluence_factors.append("High Volume Order Block")
            elif ob['volume_ratio'] >= 2.0:
                setup_score += 15
                confluence_factors.append("Good Volume Order Block")
            
            if ob['age'] <= 5:
                setup_score += 20
                confluence_factors.append("Fresh Order Block")
            elif ob['age'] <= 10:
                setup_score += 15
                confluence_factors.append("Recent Order Block")
            
            if ob['future_move'] >= 8.0:
                setup_score += 25
                confluence_factors.append("Strong Validation")
            elif ob['future_move'] >= 5.0:
                setup_score += 20
                confluence_factors.append("Good Validation")
            elif ob['future_move'] >= 3.0:
                setup_score += 10
                confluence_factors.append("Moderate Validation")
            
            # Check liquidity confluence
            for lg in liquidity_grabs:
                time_diff = abs((lg['timestamp'] - ob['timestamp']).total_seconds() / 3600)
                if time_diff <= 48:
                    if ((ob['type'] == 'bullish' and lg['type'] == 'sell_side_grab') or
                        (ob['type'] == 'bearish' and lg['type'] == 'buy_side_grab')):
                        setup_score += 15
                        confluence_factors.append(f"Liquidity Confluence ({lg['reversal_move']:.1f}%)")
            
            # Calculate entry/targets
            if ob['type'] == 'bullish':
                entry_price = ob['low']
                stop_loss = ob['low'] - (ob['high'] - ob['low']) * 0.2
                targets = [
                    entry_price * 1.02,
                    entry_price * 1.04,
                    entry_price * 1.06
                ]
            else:
                entry_price = ob['high'] 
                stop_loss = ob['high'] + (ob['high'] - ob['low']) * 0.2
                targets = [
                    entry_price * 0.98,
                    entry_price * 0.96,
                    entry_price * 0.94
                ]
            
            if setup_score >= self.min_quality_score:
                confluences.append({
                    'order_block': ob,
                    'quality_score': min(setup_score, 100),
                    'confluence_factors': confluence_factors,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'distance_pct': abs(current_price - entry_price) / current_price * 100
                })
        
        return confluences

class ICTBacktester:
    def __init__(self):
        self.scanner = ICTScanner()
        self.exchange = self.scanner.exchange
        self.lookback_months = 12  # Extended to 12 months for more comprehensive analysis
        self.commission = 0.001
        self.all_trades = []

    def add_technical_indicators(self, df):
        """Add technical indicators for correlation analysis"""
        
        # Volume indicators first
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
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        
        money_ratio = positive_flow / negative_flow
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # CMF (Chaikin Money Flow)
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        
        return df

    def backtest_symbol(self, symbol, months_back=12):
        """Backtest ICT setups for a single symbol"""
        print(f"📊 Backtesting {symbol.replace('/USDT', '')}... ", end='')
        
        try:
            # Get extended historical data
            since = self.exchange.milliseconds() - (months_back * 30 * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '4h', since=since, limit=months_back * 30 * 6)
            
            if len(ohlcv) < 200:
                print("Insufficient data")
                return []
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Split data for rolling backtests
            trades = []
            window_size = 180  # 30 days of 4H candles
            
            for i in range(window_size, len(df) - 50, 20):  # Step by 20 candles
                window_df = df.iloc[i-window_size:i].copy()
                future_df = df.iloc[i:i+50].copy()
                
                # Detect ICT setups in the window
                ict_setups = self.detect_historical_setups(window_df, future_df, symbol)
                trades.extend(ict_setups)
            
            print(f"Found {len(trades)} historical trades")
            return trades
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def detect_historical_setups(self, df, future_df, symbol):
        """Detect ICT setups and simulate their outcomes"""
        
        # Use ICT scanner detection logic
        order_blocks = self.scanner.detect_order_blocks(df)
        liquidity_grabs = self.scanner.detect_liquidity_grabs(df)
        
        if not order_blocks:
            return []
        
        # Find confluences
        confluences = self.scanner.create_basic_confluences(df, order_blocks, liquidity_grabs)
        
        if not confluences:
            return []
        
        trades = []
        current_time = df.iloc[-1]['timestamp']
        
        for conf in confluences:
            ob = conf['order_block']
            
            # Skip if price is too far from entry
            if conf['distance_pct'] > 5.0:
                continue
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'setup_time': current_time,
                'direction': 'BULLISH' if ob['type'] == 'bullish' else 'BEARISH',
                'entry_price': conf['entry_price'],
                'stop_loss': conf['stop_loss'],
                'targets': conf['targets'],
                'quality_score': conf['quality_score'],
                'confluence_factors': len(conf['confluence_factors']),
                'order_block_age': ob['age'],
                'validation_move': ob['future_move'],
                'volume_ratio': ob['volume_ratio'],
                
                # Technical indicators at setup time
                'rsi': df.iloc[-1]['rsi'],
                'mfi': df.iloc[-1]['mfi'],
                'cmf': df.iloc[-1]['cmf'],
                'macd': df.iloc[-1]['macd'],
                'macd_histogram': df.iloc[-1]['macd_histogram'],
                'momentum_10': df.iloc[-1]['momentum_10'],
                'momentum_5': df.iloc[-1]['momentum_5']
            }
            
            # Simulate trade outcome
            outcome = self.simulate_trade_outcome(trade, future_df)
            trade.update(outcome)
            
            trades.append(trade)
        
        return trades

    def simulate_trade_outcome(self, trade, future_df):
        """Simulate the outcome of an ICT trade"""
        
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        targets = trade['targets']
        direction = trade['direction']
        
        hit_target = None
        hit_stop = False
        exit_price = None
        exit_time = None
        bars_held = 0
        max_favorable = 0
        max_adverse = 0
        
        for i, (_, candle) in enumerate(future_df.iterrows()):
            bars_held = i + 1
            
            if direction == 'BULLISH':
                # Check stop loss
                if candle['low'] <= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                    exit_time = candle['timestamp']
                    break
                
                # Check targets
                for j, target in enumerate(targets):
                    if candle['high'] >= target and hit_target is None:
                        hit_target = j + 1
                        exit_price = target
                        exit_time = candle['timestamp']
                        break
                
                if hit_target:
                    break
                
                # Track excursions
                favorable_move = (candle['high'] - entry_price) / entry_price * 100
                adverse_move = (entry_price - candle['low']) / entry_price * 100
                
            else:  # BEARISH
                # Check stop loss
                if candle['high'] >= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                    exit_time = candle['timestamp']
                    break
                
                # Check targets
                for j, target in enumerate(targets):
                    if candle['low'] <= target and hit_target is None:
                        hit_target = j + 1
                        exit_price = target
                        exit_time = candle['timestamp']
                        break
                
                if hit_target:
                    break
                
                # Track excursions
                favorable_move = (entry_price - candle['low']) / entry_price * 100
                adverse_move = (candle['high'] - entry_price) / entry_price * 100
            
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)
        
        # If no exit found, use last price
        if exit_price is None:
            exit_price = future_df.iloc[-1]['close']
            exit_time = future_df.iloc[-1]['timestamp']
        
        # Calculate results
        if direction == 'BULLISH':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # Apply commission
        pnl_pct -= (self.commission * 2 * 100)
        
        return {
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl_pct': pnl_pct,
            'hit_target': hit_target,
            'hit_stop': hit_stop,
            'bars_held': bars_held,
            'max_favorable_excursion': max_favorable,
            'max_adverse_excursion': max_adverse,
            'win': pnl_pct > 0
        }

    def get_token_category(self, symbol):
        """Categorize tokens by type for analysis"""
        token = symbol.replace('/USDT', '')
        
        categories = {
            'Major Cryptos': ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'TRX', 'LTC', 'BCH'],
            'Layer 1s': ['DOT', 'AVAX', 'MATIC', 'ALGO', 'VET', 'FIL', 'ICP', 'NEAR', 'HBAR', 'FLOW', 'EGLD', 'XTZ', 'THETA', 'KLAY', 'FTM', 'ONE', 'CELO', 'LUNA'],
            'DeFi': ['UNI', 'LINK', 'CRV', 'SUSHI', 'COMP', 'YFI', 'AAVE', 'MKR', 'SNX', '1INCH', 'BAL', 'REN', 'LDO'],
            'Gaming/NFTs': ['AXS', 'ENJ', 'GALA', 'CHZ', 'ALICE', 'SLP', 'MANA', 'SAND'],
            'Meme/Community': ['SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'GMT', 'GST'],
            'Exchange Tokens': ['CRO', 'LEO', 'OKB', 'HT', 'KCS'],
            'Infrastructure': ['GRT', 'BAND', 'API3', 'LPT', 'STORJ', 'AR', 'OCEAN', 'FET'],
            'Altcoins': ['XLM', 'ETC', 'BAT', 'ZIL', 'HOT', 'IOTA', 'QTUM', 'OMG', 'ZEC', 'DASH', 'XMR', 'WAVES', 'KSM', 'RUNE'],
            'Layer 2/Scaling': ['ARB', 'OP', 'BLUR'],
            'Others': ['APE', 'STEPN', 'ATOM']
        }
        
        for category, tokens in categories.items():
            if token in tokens:
                return category
        
        return 'Others'

    def analyze_category_performance(self):
        """Analyze ICT performance by token categories"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Categorize trades
        category_stats = {}
        
        for trade in trades:
            category = self.get_token_category(trade['symbol'])
            
            if category not in category_stats:
                category_stats[category] = {
                    'trades': [],
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'symbols': set()
                }
            
            category_stats[category]['trades'].append(trade)
            category_stats[category]['symbols'].add(trade['symbol'].replace('/USDT', ''))
            
            if trade['win']:
                category_stats[category]['wins'] += 1
            else:
                category_stats[category]['losses'] += 1
            
            category_stats[category]['total_pnl'] += trade['pnl_pct']
        
        print("\n" + "=" * 70)
        print("🏆 ICT PERFORMANCE BY TOKEN CATEGORIES")
        print("=" * 70)
        
        # Sort categories by win rate
        sorted_categories = sorted(category_stats.items(), 
                                 key=lambda x: x[1]['wins'] / len(x[1]['trades']) if x[1]['trades'] else 0, 
                                 reverse=True)
        
        print(f"{'CATEGORY':<20} | {'TRADES':<7} | {'WIN RATE':<9} | {'AVG R/T':<8} | {'TOTAL R':<8} | {'SYMBOLS':<6}")
        print("-" * 70)
        
        for category, stats in sorted_categories:
            if not stats['trades']:
                continue
                
            total_trades = len(stats['trades'])
            win_rate = stats['wins'] / total_trades * 100
            avg_return = stats['total_pnl'] / total_trades
            total_return = stats['total_pnl']
            symbol_count = len(stats['symbols'])
            
            # Performance indicators
            if win_rate >= 85:
                performance_icon = "🔥"
            elif win_rate >= 75:
                performance_icon = "💪"
            elif win_rate >= 65:
                performance_icon = "✅"
            else:
                performance_icon = "⚠️"
            
            print(f"{performance_icon} {category:<17} | {total_trades:<7} | {win_rate:>6.1f}% | {avg_return:>+6.2f}% | {total_return:>+6.1f}% | {symbol_count:<6}")
        
        # Find best and worst categories
        if sorted_categories:
            best_category = sorted_categories[0]
            worst_category = sorted_categories[-1]
            
            print(f"\n🏆 BEST PERFORMING CATEGORY:")
            best_stats = best_category[1]
            best_wr = best_stats['wins'] / len(best_stats['trades']) * 100
            print(f"   {best_category[0]}: {best_wr:.1f}% win rate ({len(best_stats['trades'])} trades)")
            print(f"   Top tokens: {', '.join(list(best_stats['symbols'])[:5])}")
            
            print(f"\n⚠️  LOWEST PERFORMING CATEGORY:")
            worst_stats = worst_category[1]
            worst_wr = worst_stats['wins'] / len(worst_stats['trades']) * 100 if worst_stats['trades'] else 0
            print(f"   {worst_category[0]}: {worst_wr:.1f}% win rate ({len(worst_stats['trades'])} trades)")
            print(f"   Tokens: {', '.join(list(worst_stats['symbols'])[:5])}")
        
        print(f"\n💡 INSIGHTS:")
        print(f"   • Test across {len(category_stats)} different token categories")
        print(f"   • Categories with 85%+ win rates are ideal for ICT strategies")
        print(f"   • Focus trading on top-performing categories for better results")

    def run_backtest(self, symbols=None):
        """Run complete ICT backtest"""
        if symbols is None:
            symbols = [
                # Major Cryptocurrencies
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT',
                'DOGE/USDT', 'TRX/USDT', 'LTC/USDT', 'BCH/USDT', 'DOT/USDT', 'AVAX/USDT',
                'SHIB/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'XLM/USDT', 'ETC/USDT',
                
                # DeFi & Layer 1s
                'MATIC/USDT', 'ALGO/USDT', 'VET/USDT', 'FIL/USDT', 'ICP/USDT', 'NEAR/USDT',
                'HBAR/USDT', 'FLOW/USDT', 'EGLD/USDT', 'XTZ/USDT', 'THETA/USDT', 'KLAY/USDT',
                'MANA/USDT', 'SAND/USDT', 'CRV/USDT', 'SUSHI/USDT', 'COMP/USDT', 'YFI/USDT',
                'AAVE/USDT', 'MKR/USDT', 'SNX/USDT', '1INCH/USDT', 'BAL/USDT', 'REN/USDT',
                
                # Gaming & NFTs
                'AXS/USDT', 'ENJ/USDT', 'GALA/USDT', 'CHZ/USDT', 'ALICE/USDT', 'SLP/USDT',
                
                # Popular Altcoins
                'FTM/USDT', 'ONE/USDT', 'CELO/USDT', 'BAT/USDT', 'ZIL/USDT', 'HOT/USDT',
                'IOTA/USDT', 'QTUM/USDT', 'OMG/USDT', 'ZEC/USDT', 'DASH/USDT', 'XMR/USDT',
                'WAVES/USDT', 'KSM/USDT', 'RUNE/USDT', 'LUNA/USDT', 'FET/USDT', 'OCEAN/USDT',
                
                # Emerging Tokens
                'APE/USDT', 'GMT/USDT', 'GST/USDT', 'STEPN/USDT', 'LDO/USDT', 'ARB/USDT',
                'OP/USDT', 'BLUR/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
                
                # Exchange Tokens
                'CRO/USDT', 'LEO/USDT', 'OKB/USDT', 'HT/USDT', 'KCS/USDT',
                
                # Oracles & Infrastructure  
                'BAND/USDT', 'API3/USDT', 'GRT/USDT', 'LPT/USDT', 'STORJ/USDT', 'AR/USDT'
            ]
        
        print("🔥 ICT SCANNER BACKTESTING - COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        print(f"📊 Testing {len(symbols)} symbols over {self.lookback_months} months")
        print("🎯 Simulating Order Block + Liquidity Grab setups")
        print("🔍 Extended token coverage: Majors, DeFi, Gaming, Altcoins, & Emerging")
        print()
        
        all_trades = []
        
        for symbol in symbols:
            trades = self.backtest_symbol(symbol, self.lookback_months)
            all_trades.extend(trades)
            time.sleep(0.1)  # Rate limiting
        
        self.all_trades = all_trades
        
        if not all_trades:
            print("❌ No trades found in backtest period")
            return
        
        # Analyze results
        self.analyze_performance()
        self.analyze_category_performance()
        self.analyze_indicator_correlations()
        
        return all_trades

    def analyze_performance(self):
        """Analyze overall performance metrics"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['win']])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades * 100
        
        # PnL metrics
        total_pnl = sum([t['pnl_pct'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losing_trades > 0 else 0
        
        # Risk metrics
        avg_bars_held = np.mean([t['bars_held'] for t in trades])
        max_dd_trade = min([t['pnl_pct'] for t in trades])
        best_trade = max([t['pnl_pct'] for t in trades])
        
        # Target hit analysis
        target_hits = {}
        for i in range(1, 4):
            target_hits[f'T{i}'] = len([t for t in trades if t['hit_target'] == i])
        
        print("\n" + "=" * 60)
        print("🎯 ICT BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
        print(f"Total Return: {total_pnl:.1f}%")
        print(f"Average Return per Trade: {total_pnl/total_trades:.2f}%")
        
        print(f"\n💰 WIN/LOSS ANALYSIS:")
        print(f"Average Win: +{avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
        print(f"Best Trade: +{best_trade:.2f}%")
        print(f"Worst Trade: {max_dd_trade:.2f}%")
        
        print(f"\n🎯 TARGET HIT ANALYSIS:")
        for target, count in target_hits.items():
            pct = count / total_trades * 100
            print(f"{target} Hit Rate: {pct:.1f}% ({count} trades)")
        
        stop_hits = len([t for t in trades if t['hit_stop']])
        print(f"Stop Loss Hit: {stop_hits/total_trades*100:.1f}% ({stop_hits} trades)")
        
        print(f"\n⏰ TIMING METRICS:")
        print(f"Average Hold Time: {avg_bars_held:.1f} bars ({avg_bars_held*4:.0f} hours)")

    def analyze_indicator_correlations(self):
        """Analyze which technical indicators correlate with successful ICT trades"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Create DataFrame for analysis
        df_trades = pd.DataFrame(trades)
        
        # Remove NaN values
        numeric_columns = ['rsi', 'mfi', 'cmf', 'macd', 'macd_histogram', 'momentum_10', 'momentum_5', 'pnl_pct']
        df_clean = df_trades[numeric_columns].dropna()
        
        if len(df_clean) < 10:
            print("❌ Insufficient data for indicator correlation analysis")
            return
        
        print("\n" + "=" * 60)
        print("📈 TECHNICAL INDICATOR CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Correlation with PnL
        correlations = {}
        for indicator in numeric_columns[:-1]:  # Exclude pnl_pct
            corr = df_clean[indicator].corr(df_clean['pnl_pct'])
            correlations[indicator] = corr
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🎯 INDICATOR CORRELATION WITH TRADE SUCCESS:")
        print("(Higher absolute values = stronger predictive power)")
        print()
        
        for indicator, corr in sorted_corr:
            strength = self.get_correlation_strength(abs(corr))
            direction = "📈 Positive" if corr > 0 else "📉 Negative"
            print(f"{indicator.upper():15} | {corr:+.3f} | {strength} {direction}")
        
        # Win rate by indicator ranges
        print(f"\n🏆 WIN RATE BY INDICATOR RANGES:")
        self.analyze_indicator_ranges(df_clean)

    def get_correlation_strength(self, abs_corr):
        """Classify correlation strength"""
        if abs_corr >= 0.7:
            return "🔥 Very Strong"
        elif abs_corr >= 0.5:
            return "💪 Strong    "
        elif abs_corr >= 0.3:
            return "✅ Moderate  "
        elif abs_corr >= 0.1:
            return "⚠️  Weak     "
        else:
            return "❌ None     "

    def analyze_indicator_ranges(self, df):
        """Analyze win rates across different indicator value ranges"""
        
        key_indicators = ['rsi', 'mfi', 'cmf', 'macd_histogram']
        
        for indicator in key_indicators:
            if indicator not in df.columns:
                continue
            
            # Create quantile-based ranges
            q25 = df[indicator].quantile(0.25)
            q75 = df[indicator].quantile(0.75)
            
            low_range = df[df[indicator] <= q25]
            mid_range = df[(df[indicator] > q25) & (df[indicator] < q75)]
            high_range = df[df[indicator] >= q75]
            
            ranges = {
                'Low': low_range,
                'Mid': mid_range, 
                'High': high_range
            }
            
            print(f"\n{indicator.upper()} Performance by Range:")
            for range_name, range_data in ranges.items():
                if len(range_data) > 0:
                    wins = len(range_data[range_data['pnl_pct'] > 0])
                    total = len(range_data)
                    win_rate = wins / total * 100
                    avg_return = range_data['pnl_pct'].mean()
                    
                    range_values = f"({range_data[indicator].min():.2f} to {range_data[indicator].max():.2f})"
                    print(f"  {range_name:4} {range_values:15} | {win_rate:5.1f}% WR | {avg_return:+6.2f}% avg | {total:3d} trades")

def main():
    """Run ICT backtesting with comprehensive analysis"""
    try:
        backtester = ICTBacktester()
        results = backtester.run_backtest()
        
        if results:
            print(f"\n✅ Comprehensive backtesting completed successfully!")
            print(f"📊 Results analyzed for {len(results)} historical trades across 85+ tokens")
            print(f"🏆 Category performance analysis included")
            print(f"📈 Technical indicator correlations identified")
        
    except KeyboardInterrupt:
        print("\n❌ Backtesting interrupted by user")
    except Exception as e:
        print(f"\n❌ Backtesting error: {e}")

if __name__ == "__main__":
    main()