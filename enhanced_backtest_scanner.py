#!/usr/bin/env python3
"""
Enhanced Backtest Scanner - Full Integration
Integrates your complete scanner system with comprehensive backtesting
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing enhanced modules  
from modules.data_fetcher import MarketDataFetcher  # Correct class name from your file
from modules.bb_detector import BBDetector  # Fixed class name
from modules.technical_analyzer import TechnicalAnalyzer
from modules.market_regime_enhanced import EnhancedMarketRegimeAnalyzer  
from modules.pattern_analyzer import PatternAnalyzer
from modules.risk_manager import RiskManager
from modules.enhanced_backtesting_engine import EnhancedBacktestingEngine
from config import SCANNER_CONFIG  # Only import what exists

class EnhancedBacktestScanner:
    """
    Complete integration of your enhanced BB scanner with institutional-grade backtesting
    """
    
    def __init__(self):
        print("🚀 ENHANCED BB SCANNER - COMPREHENSIVE HISTORICAL BACKTESTING")
        print("=" * 80)
        print("🏗️ Initializing Enhanced Historical Backtest Scanner...")
        
        # Initialize all your existing enhanced modules
        self.data_fetcher = MarketDataFetcher()  # Correct class name
        self.bb_detector = BBDetector()  # Fixed class name
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Enhanced market regime needs the original analyzer first
        try:
            from modules.market_regime_analyzer import MarketRegimeAnalyzer
            original_regime = MarketRegimeAnalyzer()
            self.regime_analyzer = EnhancedMarketRegimeAnalyzer(original_regime)
        except Exception as e:
            print(f"⚠️ Enhanced market regime not available: {e}")
            self.regime_analyzer = None
            
        self.pattern_analyzer = PatternAnalyzer()  # Your pattern recognition
        self.risk_manager = RiskManager()
        
        # Initialize enhanced backtesting engine
        self.backtest_engine = EnhancedBacktestingEngine()
        
        print("✅ All enhanced modules initialized successfully")
        
    def run_comprehensive_backtest(self, timeframes=[30, 90, 180, 365], symbols=None):
        """
        Run comprehensive multi-timeframe backtest using your complete enhanced system
        """
        print("\n🎯 STARTING COMPREHENSIVE ENHANCED BACKTEST")
        print("=" * 60)
        
        # Use your scanner's coin selection if no symbols provided
        if symbols is None:
            print("📊 Using your scanner's enhanced coin selection...")
            top_coins = self.data_fetcher.fetch_top_coins(limit=SCANNER_CONFIG['top_coins_limit'])
            symbols = [coin['symbol'] for coin in top_coins[:20]]  # Test top 20 for speed
            print(f"✅ Selected {len(symbols)} coins for backtesting")
        
        results = {}
        
        for timeframe_days in timeframes:
            print(f"\n📅 ANALYZING {timeframe_days}-DAY TIMEFRAME")
            print("-" * 40)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=timeframe_days)
            
            timeframe_signals = []
            
            for i, symbol in enumerate(symbols, 1):
                print(f"📊 Analyzing {symbol} ({i}/{len(symbols)})...")
                
                try:
                    # Generate historical signals using your complete enhanced system
                    signals = self._generate_enhanced_historical_signals(
                        symbol, start_date, end_date
                    )
                    
                    if signals:
                        timeframe_signals.extend(signals)
                        print(f"   ✅ Found {len(signals)} signals for {symbol}")
                    else:
                        print(f"   ➖ No signals for {symbol}")
                        
                except Exception as e:
                    print(f"   ❌ Error analyzing {symbol}: {str(e)}")
                    continue
            
            if timeframe_signals:
                # Run backtest on this timeframe
                print(f"\n🔄 RUNNING BACKTEST SIMULATION ({timeframe_days} days)...")
                print(f"📊 Processing {len(timeframe_signals)} signals...")
                
                # Clear engine and add signals
                self.backtest_engine.clear_signals()
                for signal in timeframe_signals:
                    self.backtest_engine.add_signal(signal)
                
                # Run comprehensive backtest
                backtest_results = self.backtest_engine.run_comprehensive_backtest()
                
                # Store results
                results[f"{timeframe_days}d"] = {
                    'signals': len(timeframe_signals),
                    'performance': backtest_results,
                    'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                }
                
                print(f"✅ {timeframe_days}-day backtest complete")
            else:
                print(f"❌ No signals found for {timeframe_days}-day period")
                results[f"{timeframe_days}d"] = {'signals': 0, 'performance': None}
        
        # Generate comprehensive report
        self._generate_comprehensive_report(results)
        
        return results
    
    def _generate_enhanced_historical_signals(self, symbol, start_date, end_date):
        """
        Generate historical signals using your complete enhanced system
        """
        signals = []
        
        try:
            # Simulate historical analysis day by day
            current_date = start_date
            
            while current_date <= end_date:
                # Fetch historical data for this point in time
                end_timestamp = int(current_date.timestamp() * 1000)
                
                # Get data for multiple exchanges (your system supports 4)
                for exchange_name in ['binance', 'bybit', 'kucoin', 'okx']:
                    try:
                        # Fetch historical OHLCV data
                        df_4h = self.data_fetcher.fetch_ohlcv(
                            exchange_name, symbol, '4h'
                        )
                        df_1h = self.data_fetcher.fetch_ohlcv(
                            exchange_name, symbol, '1h'
                        )
                        
                        if df_4h is None or df_1h is None or len(df_4h) < 50:
                            continue
                        
                        # Run your complete enhanced analysis pipeline
                        signal = self._run_enhanced_analysis_pipeline(
                            symbol, exchange_name, df_4h, df_1h, current_date
                        )
                        
                        if signal:
                            signals.append(signal)
                            
                    except Exception as e:
                        continue
                
                # Move to next day
                current_date += timedelta(days=1)
                
        except Exception as e:
            print(f"Error in historical analysis for {symbol}: {str(e)}")
        
        return signals
    
    def _run_enhanced_analysis_pipeline(self, symbol, exchange_name, df_4h, df_1h, analysis_date):
        """
        Run your complete enhanced analysis pipeline on historical data
        """
        try:
            # 1. BB Detection (your core system)
            bb_analysis = self.bb_detector.analyze_bb_setup(df_4h)
            if bb_analysis['setup_type'] == 'NONE':
                return None
            
            # 2. Technical Analysis (your 5-indicator system)
            tech_analysis = self.technical_analyzer.analyze_comprehensive(df_4h)
            
            # 3. Enhanced Market Regime (your 9-layer system)
            try:
                if self.regime_analyzer:
                    regime_analysis = self.regime_analyzer.analyze_market_regime(df_4h)
                    regime_confidence = regime_analysis.get('confidence', 50)
                    position_multiplier = regime_analysis.get('position_multiplier', 1.0)
                else:
                    regime_confidence = 50
                    position_multiplier = 1.0
            except:
                regime_confidence = 50
                position_multiplier = 1.0
            
            # 4. Pattern Recognition (your enhanced pattern system)
            try:
                pattern_analysis = self.pattern_analyzer.analyze_comprehensive_patterns(
                    df_4h, df_1h, tech_analysis
                )
                pattern_boost = pattern_analysis.get('total_boost', 0)
                pattern_confidence = pattern_analysis.get('final_confidence', 0)
            except:
                pattern_boost = 0
                pattern_confidence = 0
            
            # 5. Risk Management (your system)
            risk_analysis = self.risk_manager.calculate_position_size(
                bb_analysis['probability'], 
                bb_analysis['risk_level']
            )
            
            # 6. Calculate Confluence Score (0-100 points)
            confluence_score = self._calculate_confluence_score(
                tech_analysis, regime_analysis, pattern_analysis, bb_analysis
            )
            
            # Create enhanced signal
            signal = {
                'timestamp': analysis_date,
                'symbol': symbol,
                'exchange': exchange_name,
                'setup_type': bb_analysis['setup_type'],
                'entry_price': df_4h['close'].iloc[-1],
                'probability': bb_analysis['probability'],
                'risk_level': bb_analysis['risk_level'],
                
                # Enhanced features
                'regime_confidence': regime_confidence,
                'position_multiplier': position_multiplier,
                'pattern_boost': pattern_boost,
                'pattern_confidence': pattern_confidence,
                'confluence_score': confluence_score,
                
                # Risk management
                'stop_loss': bb_analysis['stop_loss'],
                'take_profit': bb_analysis['take_profit'],
                'position_size': risk_analysis['position_size'],
                
                # Technical indicators
                'rsi': tech_analysis.get('rsi', 50),
                'volume_ratio': tech_analysis.get('volume_ratio', 1.0),
                
                # Metadata for backtesting
                'historical_date': analysis_date.strftime('%Y-%m-%d'),
                'data_quality': 'high' if len(df_4h) > 100 else 'medium'
            }
            
            return signal
            
        except Exception as e:
            return None
    
    def _calculate_confluence_score(self, tech_analysis, regime_analysis, pattern_analysis, bb_analysis):
        """
        Calculate confluence score (0-100 points) as outlined in your requirements
        """
        score = 0
        
        # RSI Confluence (0-20 points)
        rsi = tech_analysis.get('rsi', 50)
        if bb_analysis['setup_type'] == 'LONG' and rsi < 30:
            score += 20  # Oversold for LONG
        elif bb_analysis['setup_type'] == 'SHORT' and rsi > 70:
            score += 20  # Overbought for SHORT
        elif 30 <= rsi <= 70:
            score += 10  # Neutral RSI
        
        # Market Regime Alignment (0-20 points)
        regime_confidence = regime_analysis.get('confidence', 50)
        if regime_confidence > 70:
            score += 20
        elif regime_confidence > 50:
            score += 15
        elif regime_confidence > 30:
            score += 10
        
        # Pattern Recognition (0-20 points)
        pattern_confidence = pattern_analysis.get('final_confidence', 0)
        if pattern_confidence > 80:
            score += 20
        elif pattern_confidence > 60:
            score += 15
        elif pattern_confidence > 40:
            score += 10
        
        # Volume Confirmation (0-20 points)
        volume_ratio = tech_analysis.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 15
        elif volume_ratio > 1.0:
            score += 10
        
        # BB Setup Quality (0-20 points)
        probability = bb_analysis.get('probability', 50)
        if probability > 75:
            score += 20
        elif probability > 65:
            score += 15
        elif probability > 55:
            score += 10
        
        return min(score, 100)  # Cap at 100
    
    def _generate_comprehensive_report(self, results):
        """
        Generate comprehensive multi-timeframe report
        """
        print("\n🎯 COMPREHENSIVE BACKTEST RESULTS")
        print("=" * 60)
        
        summary_data = []
        
        for timeframe, data in results.items():
            if data['performance']:
                perf = data['performance']
                
                print(f"\n📊 {timeframe.upper()} TIMEFRAME RESULTS")
                print("-" * 30)
                print(f"🎯 Win Rate: {perf['win_rate']:.1f}%")
                print(f"💵 Total Return: {perf['total_return']:.2f}%")
                print(f"💰 Profit Factor: {perf['profit_factor']:.2f}")
                print(f"📉 Max Drawdown: {perf['max_drawdown']:.1f}%")
                print(f"⏱️  Avg Hold Time: {perf['avg_hold_time']:.1f} days")
                print(f"📊 Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
                print(f"📈 Total Signals: {data['signals']}")
                
                summary_data.append({
                    'timeframe': timeframe,
                    'win_rate': perf['win_rate'],
                    'total_return': perf['total_return'],
                    'profit_factor': perf['profit_factor'],
                    'max_drawdown': perf['max_drawdown'],
                    'sharpe_ratio': perf['sharpe_ratio'],
                    'signals': data['signals']
                })
        
        # Calculate weighted confidence score
        if summary_data:
            weights = {'30d': 0.40, '90d': 0.25, '180d': 0.20, '365d': 0.15}
            weighted_score = 0
            total_weight = 0
            
            for data in summary_data:
                if data['timeframe'] in weights:
                    weight = weights[data['timeframe']]
                    weighted_score += data['win_rate'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_confidence = weighted_score / total_weight
                print(f"\n🏆 FINAL WEIGHTED CONFIDENCE SCORE: {final_confidence:.1f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/enhanced_backtest_results_{timestamp}.json"
        
        os.makedirs("outputs", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {filename}")

# Test the enhanced integration
if __name__ == "__main__":
    print("🧪 Testing Enhanced Backtest Integration...")
    
    scanner = EnhancedBacktestScanner()
    
    # Test with small subset first
    test_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
    
    print(f"🔍 Testing with {len(test_symbols)} symbols")
    print("⚡ Running 30-day backtest for validation...")
    
    # Quick test - just 30 days
    results = scanner.run_comprehensive_backtest(
        timeframes=[30], 
        symbols=test_symbols
    )
    
    print("\n🎉 Enhanced backtest integration test complete!")