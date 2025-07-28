#!/usr/bin/env python3
"""
1H-4H TREND FOLLOWING PULLBACK SCANNER
Professional trend following scanner for pullback entries in established trends
Strategy: 20-30% retracements in strong trends → 5-15% target moves
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class TrendFollowingScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
    def get_top_coins(self, limit=150):
        """Get top coins by volume for trend following"""
        try:
            url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"
            params = {
                'start': 1,
                'limit': limit,
                'sortBy': 'volume_24h',
                'sortType': 'desc',
                'convert': 'USD',
                'cryptoType': 'all',
                'tagType': 'all'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            coins = []
            for coin in data['data']['cryptoCurrencyList']:
                symbol = coin['symbol']
                volume_24h = coin['quotes'][0]['volume24h']
                market_cap = coin['quotes'][0]['marketCap']
                
                # High volume requirements for trend following
                if (volume_24h > 20_000_000 and  # High volume for trend reliability
                    market_cap > 200_000_000 and   # Large caps trend better
                    market_cap < 500_000_000_000): # Exclude BTC/ETH
                    coins.append(symbol)
            
            return coins[:100]  # Top 100 for trend analysis
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            return ['BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LTC', 'LINK']
    
    def get_binance_klines(self, symbol, interval='4h', limit=100):
        """Get klines from Binance for trend analysis"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            return None
    
    def calculate_trend_indicators(self, df):
        """Calculate trend identification indicators"""
        # EMAs for trend direction
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # RSI for momentum
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD for trend confirmation
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        return df
    
    def identify_trend_direction(self, df):
        """Identify the current trend direction and strength"""
        current = df.iloc[-1]
        recent = df.tail(10)  # Last 10 periods
        
        # EMA alignment for trend
        ema_bullish = (current['ema_20'] > current['ema_50'] > current['ema_100'])
        ema_bearish = (current['ema_20'] < current['ema_50'] < current['ema_100'])
        
        # Price vs EMA position
        price_above_emas = current['close'] > current['ema_20']
        price_below_emas = current['close'] < current['ema_20']
        
        # MACD confirmation
        macd_bullish = current['macd'] > current['macd_signal'] and current['macd'] > 0
        macd_bearish = current['macd'] < current['macd_signal'] and current['macd'] < 0
        
        # Trend strength calculation
        ema_spread_20_50 = abs(current['ema_20'] - current['ema_50']) / current['close'] * 100
        ema_spread_50_100 = abs(current['ema_50'] - current['ema_100']) / current['close'] * 100
        
        # Price movement over trend period
        lookback_20 = min(20, len(df))
        price_change_20 = (current['close'] - df.iloc[-lookback_20]['close']) / df.iloc[-lookback_20]['close'] * 100
        
        trend_strength = min(100, (ema_spread_20_50 + ema_spread_50_100) * 10)
        
        # Determine trend
        if ema_bullish and price_above_emas and price_change_20 > 5:
            trend_direction = "BULLISH"
            trend_score = 70 + min(30, trend_strength)
        elif ema_bearish and price_below_emas and price_change_20 < -5:
            trend_direction = "BEARISH" 
            trend_score = 70 + min(30, trend_strength)
        elif (current['ema_20'] > current['ema_50'] and price_change_20 > 2):
            trend_direction = "WEAK_BULLISH"
            trend_score = 50 + min(20, trend_strength)
        elif (current['ema_20'] < current['ema_50'] and price_change_20 < -2):
            trend_direction = "WEAK_BEARISH"
            trend_score = 50 + min(20, trend_strength)
        else:
            trend_direction = "SIDEWAYS"
            trend_score = 30
        
        return {
            'direction': trend_direction,
            'strength': trend_score,
            'price_change_20': price_change_20,
            'ema_alignment': ema_bullish or ema_bearish,
            'macd_confirmation': macd_bullish or macd_bearish
        }
    
    def find_pullback_levels(self, df, trend_direction):
        """Find pullback entry levels in established trends"""
        if trend_direction not in ['BULLISH', 'WEAK_BULLISH', 'BEARISH', 'WEAK_BEARISH']:
            return None
        
        # Look for recent swing high/low for pullback measurement
        lookback = min(20, len(df))
        recent_data = df.tail(lookback)
        
        current_price = df['close'].iloc[-1]
        
        if trend_direction in ['BULLISH', 'WEAK_BULLISH']:
            # Find recent swing high for bullish pullback
            swing_high = recent_data['high'].max()
            swing_high_idx = recent_data['high'].idxmax()
            
            # Find swing low before the high
            pre_high_data = df.loc[:swing_high_idx].tail(10) if len(df.loc[:swing_high_idx]) > 10 else df.loc[:swing_high_idx]
            if len(pre_high_data) > 0:
                swing_low = pre_high_data['low'].min()
            else:
                swing_low = recent_data['low'].min()
            
            # Calculate pullback levels
            move_size = swing_high - swing_low
            pullback_20 = swing_high - (move_size * 0.20)  # 20% retracement
            pullback_30 = swing_high - (move_size * 0.30)  # 30% retracement
            pullback_38 = swing_high - (move_size * 0.382) # 38.2% Fibonacci
            pullback_50 = swing_high - (move_size * 0.50)  # 50% retracement
            
            # Check current pullback percentage
            current_pullback_pct = (swing_high - current_price) / move_size * 100 if move_size > 0 else 0
            
            return {
                'type': 'BULLISH_PULLBACK',
                'swing_high': swing_high,
                'swing_low': swing_low,
                'move_size_pct': move_size / swing_low * 100 if swing_low > 0 else 0,
                'current_pullback_pct': current_pullback_pct,
                'pullback_20': pullback_20,
                'pullback_30': pullback_30,
                'pullback_38': pullback_38,
                'pullback_50': pullback_50,
                'current_price': current_price
            }
        
        else:  # BEARISH trends
            # Find recent swing low for bearish pullback
            swing_low = recent_data['low'].min()
            swing_low_idx = recent_data['low'].idxmin()
            
            # Find swing high before the low
            pre_low_data = df.loc[:swing_low_idx].tail(10) if len(df.loc[:swing_low_idx]) > 10 else df.loc[:swing_low_idx]
            if len(pre_low_data) > 0:
                swing_high = pre_low_data['high'].max()
            else:
                swing_high = recent_data['high'].max()
            
            # Calculate pullback levels
            move_size = swing_high - swing_low
            pullback_20 = swing_low + (move_size * 0.20)  # 20% retracement
            pullback_30 = swing_low + (move_size * 0.30)  # 30% retracement
            pullback_38 = swing_low + (move_size * 0.382) # 38.2% Fibonacci
            pullback_50 = swing_low + (move_size * 0.50)  # 50% retracement
            
            # Check current pullback percentage
            current_pullback_pct = (current_price - swing_low) / move_size * 100 if move_size > 0 else 0
            
            return {
                'type': 'BEARISH_PULLBACK',
                'swing_high': swing_high,
                'swing_low': swing_low,
                'move_size_pct': move_size / swing_high * 100 if swing_high > 0 else 0,
                'current_pullback_pct': current_pullback_pct,
                'pullback_20': pullback_20,
                'pullback_30': pullback_30,
                'pullback_38': pullback_38,
                'pullback_50': pullback_50,
                'current_price': current_price
            }
    
    def calculate_trade_setup(self, df, trend_info, pullback_info):
        """Calculate entry, targets, and stops for trend following trade"""
        if not pullback_info:
            return None
        
        current = df.iloc[-1]
        atr = current['atr']
        
        trade_type = pullback_info['type']
        current_price = pullback_info['current_price']
        current_pullback = pullback_info['current_pullback_pct']
        
        # Entry conditions based on pullback level
        if trade_type == 'BULLISH_PULLBACK':
            # Entry levels for bullish pullback
            if 15 <= current_pullback <= 35:  # 20-30% pullback zone
                entry_price = current_price
                entry_timing = "immediate"
            elif 35 < current_pullback <= 55:  # 38-50% pullback zone
                entry_price = pullback_info['pullback_38']
                entry_timing = "wait_for_pullback"
            elif current_pullback < 15:
                entry_price = pullback_info['pullback_20']
                entry_timing = "wait_for_deeper_pullback"
            else:
                return None  # Too deep pullback
            
            # Targets for bullish trend
            move_size = pullback_info['swing_high'] - pullback_info['swing_low']
            target_1 = pullback_info['swing_high'] + (move_size * 0.50)  # 50% extension
            target_2 = pullback_info['swing_high'] + (move_size * 1.00)  # 100% extension
            target_3 = pullback_info['swing_high'] + (move_size * 1.618) # 161.8% extension
            
            # Stop loss
            stop_loss = min(pullback_info['pullback_50'], current_price - (atr * 2))
            
            trade_direction = "LONG"
        
        else:  # BEARISH_PULLBACK
            # Entry levels for bearish pullback
            if 15 <= current_pullback <= 35:  # 20-30% pullback zone
                entry_price = current_price
                entry_timing = "immediate"
            elif 35 < current_pullback <= 55:  # 38-50% pullback zone
                entry_price = pullback_info['pullback_38']
                entry_timing = "wait_for_pullback"
            elif current_pullback < 15:
                entry_price = pullback_info['pullback_20']
                entry_timing = "wait_for_deeper_pullback"
            else:
                return None  # Too deep pullback
            
            # Targets for bearish trend
            move_size = pullback_info['swing_high'] - pullback_info['swing_low']
            target_1 = pullback_info['swing_low'] - (move_size * 0.50)  # 50% extension
            target_2 = pullback_info['swing_low'] - (move_size * 1.00)  # 100% extension
            target_3 = pullback_info['swing_low'] - (move_size * 1.618) # 161.8% extension
            
            # Stop loss
            stop_loss = max(pullback_info['pullback_50'], current_price + (atr * 2))
            
            trade_direction = "SHORT"
        
        # Calculate risk/reward
        risk_pct = abs(entry_price - stop_loss) / entry_price * 100
        target_1_pct = abs(target_1 - entry_price) / entry_price * 100
        target_2_pct = abs(target_2 - entry_price) / entry_price * 100
        target_3_pct = abs(target_3 - entry_price) / entry_price * 100
        
        risk_reward_1 = target_1_pct / risk_pct if risk_pct > 0 else 0
        risk_reward_2 = target_2_pct / risk_pct if risk_pct > 0 else 0
        risk_reward_3 = target_3_pct / risk_pct if risk_pct > 0 else 0
        
        return {
            'direction': trade_direction,
            'entry_price': entry_price,
            'entry_timing': entry_timing,
            'stop_loss': stop_loss,
            'targets': {
                'target_1': {'price': target_1, 'gain_pct': target_1_pct, 'rr': risk_reward_1},
                'target_2': {'price': target_2, 'gain_pct': target_2_pct, 'rr': risk_reward_2},
                'target_3': {'price': target_3, 'gain_pct': target_3_pct, 'rr': risk_reward_3}
            },
            'risk_pct': risk_pct,
            'current_pullback': current_pullback
        }
    
    def calculate_setup_quality_score(self, trend_info, pullback_info, trade_setup, df):
        """Score the quality of the trend following setup"""
        score = 20  # Base score
        
        # Trend strength scoring
        trend_strength = trend_info['strength']
        if trend_strength >= 80:
            score += 30  # Very strong trend
        elif trend_strength >= 65:
            score += 25  # Strong trend
        elif trend_strength >= 50:
            score += 15  # Moderate trend
        else:
            score += 5   # Weak trend
        
        # Pullback quality
        pullback_pct = pullback_info['current_pullback_pct']
        if 20 <= pullback_pct <= 30:
            score += 20  # Ideal pullback
        elif 15 <= pullback_pct <= 35:
            score += 15  # Good pullback
        elif 35 <= pullback_pct <= 50:
            score += 10  # Acceptable pullback
        else:
            score += 5   # Less ideal
        
        # Move size quality
        move_size = pullback_info['move_size_pct']
        if move_size >= 15:
            score += 15  # Large preceding move
        elif move_size >= 10:
            score += 10  # Good preceding move
        elif move_size >= 7:
            score += 5   # Moderate move
        
        # Risk/reward scoring
        best_rr = max([t['rr'] for t in trade_setup['targets'].values()])
        if best_rr >= 3.0:
            score += 15  # Excellent R:R
        elif best_rr >= 2.0:
            score += 10  # Good R:R
        elif best_rr >= 1.5:
            score += 5   # Acceptable R:R
        
        # Entry timing bonus
        if trade_setup['entry_timing'] == 'immediate':
            score += 10  # Ready to trade
        elif trade_setup['entry_timing'] == 'wait_for_pullback':
            score += 5   # Close to entry
        
        # Volume confirmation
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume_ma'].iloc[-1]
        if current_volume > avg_volume * 1.2:
            score += 5  # Above average volume
        
        return min(score, 100)  # Cap at 100
    
    def scan_for_trend_setups(self):
        """Main scanning function for trend following opportunities"""
        print("🎯 1H-4H TREND FOLLOWING PULLBACK SCANNER")
        print("=" * 65)
        print("📈 Scanning for pullback entries in established trends...")
        print("📊 Strategy: 20-30% retracements → 5-15% target moves")
        print("⏰ Timeframe: 4H primary analysis, 1H confirmation")
        print()
        
        coins = self.get_top_coins()
        opportunities = []
        
        stats = {'scanned': 0, 'no_data': 0, 'trends_found': 0, 'setups_found': 0}
        
        for symbol in coins:
            try:
                print(f"📡 {symbol}... ", end="")
                stats['scanned'] += 1
                
                # Get 4H data for trend analysis
                df_4h = self.get_binance_klines(symbol, interval='4h', limit=100)
                if df_4h is None or len(df_4h) < 50:
                    print("No data")
                    stats['no_data'] += 1
                    continue
                
                # Calculate indicators
                df_4h = self.calculate_trend_indicators(df_4h)
                
                # Identify trend
                trend_info = self.identify_trend_direction(df_4h)
                
                if trend_info['direction'] == 'SIDEWAYS':
                    print("No trend")
                    continue
                
                stats['trends_found'] += 1
                
                # Find pullback levels
                pullback_info = self.find_pullback_levels(df_4h, trend_info['direction'])
                
                if not pullback_info:
                    print(f"{trend_info['direction']} (no pullback)")
                    continue
                
                # Calculate trade setup
                trade_setup = self.calculate_trade_setup(df_4h, trend_info, pullback_info)
                
                if not trade_setup:
                    print(f"{trend_info['direction']} (poor setup)")
                    continue
                
                # Score the setup
                quality_score = self.calculate_setup_quality_score(
                    trend_info, pullback_info, trade_setup, df_4h
                )
                
                if quality_score >= 60:  # Quality threshold for trend following
                    opportunity = {
                        'symbol': symbol,
                        'trend': trend_info,
                        'pullback': pullback_info,
                        'trade': trade_setup,
                        'quality_score': quality_score,
                        'timeframe': '4H'
                    }
                    opportunities.append(opportunity)
                    
                    # Quick display
                    direction = trade_setup['direction']
                    timing_map = {
                        'immediate': '🎯 NOW',
                        'wait_for_pullback': '⏳ WAIT',
                        'wait_for_deeper_pullback': '📉 DEEP'
                    }
                    status = timing_map.get(trade_setup['entry_timing'], '❓')
                    
                    best_target = max([t['gain_pct'] for t in trade_setup['targets'].values()])
                    best_rr = max([t['rr'] for t in trade_setup['targets'].values()])
                    
                    print(f"{status} {direction} | Target {best_target:.1f}% | R:R {best_rr:.1f}:1 | Score {quality_score}")
                    stats['setups_found'] += 1
                else:
                    print(f"{trend_info['direction']} (score {quality_score})")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error - {e}")
                continue
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {stats['scanned']}")
        print(f"No data: {stats['no_data']}")
        print(f"Trends found: {stats['trends_found']}")
        print(f"Quality setups: {stats['setups_found']}")
        
        # Sort by quality score
        opportunities.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display trend following scanning results"""
        print("\n" + "=" * 85)
        print(f"🎯 FOUND {len(opportunities)} TREND FOLLOWING PULLBACK OPPORTUNITIES")
        print("=" * 85)
        
        if not opportunities:
            print("❌ No high-quality trend following setups found")
            print("💡 Wait for stronger trends or deeper pullbacks")
            return
        
        print("📊 Results (sorted by quality score):")
        print()
        
        for i, setup in enumerate(opportunities[:10], 1):
            trend = setup['trend']
            pullback = setup['pullback']
            trade = setup['trade']
            
            # Direction and emoji
            if trade['direction'] == 'LONG':
                direction_emoji = "📈"
                trend_emoji = "🟢"
            else:
                direction_emoji = "📉"
                trend_emoji = "🔴"
            
            # Entry timing status
            timing_map = {
                'immediate': '🎯 NOW',
                'wait_for_pullback': '⏳ WAIT',
                'wait_for_deeper_pullback': '📉 DEEP'
            }
            status = timing_map[trade['entry_timing']]
            
            print(f"{i:2}. {direction_emoji} {setup['symbol']:8} | Score: {setup['quality_score']:2}/100 | "
                  f"{trend_emoji} {trend['direction']:12} | {status}")
            
            # Current situation
            print(f"     Current: ${pullback['current_price']:.4f} | "
                  f"Pullback: {pullback['current_pullback_pct']:.1f}% | "
                  f"Trend Strength: {trend['strength']:.0f}/100")
            
            # Entry and targets
            targets = trade['targets']
            print(f"     Entry: ${trade['entry_price']:.4f} | "
                  f"Stop: ${trade['stop_loss']:.4f} ({trade['risk_pct']:.1f}%)")
            
            print(f"     T1: ${targets['target_1']['price']:.4f} "
                  f"({targets['target_1']['gain_pct']:.1f}% | R:R {targets['target_1']['rr']:.1f}:1)")
            print(f"     T2: ${targets['target_2']['price']:.4f} "
                  f"({targets['target_2']['gain_pct']:.1f}% | R:R {targets['target_2']['rr']:.1f}:1)")
            print(f"     T3: ${targets['target_3']['price']:.4f} "
                  f"({targets['target_3']['gain_pct']:.1f}% | R:R {targets['target_3']['rr']:.1f}:1)")
            
            # Trend details
            if pullback['type'] == 'BULLISH_PULLBACK':
                print(f"     Trend: ${pullback['swing_low']:.4f} → ${pullback['swing_high']:.4f} "
                      f"({pullback['move_size_pct']:.1f}% move)")
            else:
                print(f"     Trend: ${pullback['swing_high']:.4f} → ${pullback['swing_low']:.4f} "
                      f"({pullback['move_size_pct']:.1f}% move)")
            
            # Entry guidance
            if trade['entry_timing'] == 'immediate':
                print(f"     💡 ENTER NOW - Price in optimal pullback zone")
            elif trade['entry_timing'] == 'wait_for_pullback':
                print(f"     💡 Wait for price to reach ${trade['entry_price']:.4f}")
            else:
                print(f"     💡 Wait for deeper pullback to ${trade['entry_price']:.4f}")
            
            print()
        
        print("🎯 TREND FOLLOWING STRATEGY:")
        print("• Entry: 20-30% pullbacks in established trends")
        print("• Targets: 50%, 100%, 161.8% trend extensions")
        print("• Stops: Below/above 50% pullback or 2x ATR")
        print("• Position Size: Scale based on trend strength")
        print("• Hold Time: Days to weeks (not hours)")
        print("• Best R:R: 2:1 minimum, 3:1+ preferred")
        print("• Success Rate: ~60-70% in strong trending markets")

def main():
    scanner = TrendFollowingScanner()
    opportunities = scanner.scan_for_trend_setups()
    scanner.display_results(opportunities)

if __name__ == "__main__":
    main()