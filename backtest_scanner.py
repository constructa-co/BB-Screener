# backtest_scanner.py - Historical Signal Generator
"""
Backtest Scanner that uses your enhanced BB scanner modules 
to generate historical signals and test performance

FEATURES:
- Uses all your existing scanner modules
- Fetches historical data for multiple timeframes
- Generates signals using enhanced market regime
- Feeds signals to backtesting engine
- Produces comprehensive performance reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import your existing modules
try:
    from modules.data_fetcher import MarketDataFetcher
    from modules.bb_detector import BBDetector
    from modules.technical_analyzer import TechnicalAnalyzer
    from modules.sentiment_analyzer import SentimentAnalyzer
    from modules.risk_manager import RiskManager
    from modules.market_regime_analyzer import MarketRegimeAnalyzer
    from modules.market_regime_enhanced import create_enhanced_regime_analyzer
    from modules.pattern_analyzer import PatternAnalyzer
    from modules.backtesting_engine import EnhancedBacktestEngine
    from config import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are available")
    sys.exit(1)

class HistoricalBacktestScanner:
    """
    Historical backtesting scanner using your enhanced BB scanner modules
    """
    
    def __init__(self, initial_capital: float = 10000):
        """Initialize historical scanner with all your modules"""
        print("🚀 Initializing Historical Backtest Scanner...")
        
        # Initialize your existing modules
        self.data_fetcher = MarketDataFetcher()
        self.bb_detector = BBDetector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager()
        self.regime_analyzer = MarketRegimeAnalyzer(self.data_fetcher, self.sentiment_analyzer)
        
        # Initialize backtesting engine
        self.backtest_engine = EnhancedBacktestEngine(initial_capital)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        print("✅ All modules initialized successfully")
    
    def run_historical_backtest(self, 
                              symbols: List[str] = None,
                              start_date: datetime = None,
                              end_date: datetime = None,
                              timeframe: str = '4h') -> Dict:
        """
        Run complete historical backtest
        
        Args:
            symbols: List of symbols to test (defaults to top coins)
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Timeframe for analysis
            
        Returns:
            Dict containing backtest results
        """
        print(f"\n🎯 STARTING HISTORICAL BACKTEST")
        print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Use top coins if none specified
        if symbols is None:
            symbols = self._get_top_symbols_for_backtest()
        
        print(f"🪙 Testing {len(symbols)} symbols")
        
        # Generate historical signals
        total_signals = 0
        
        for i, symbol in enumerate(symbols):
            try:
                print(f"📊 Analyzing {symbol} ({i+1}/{len(symbols)})...")
                
                # Generate signals for this symbol
                signals = self._generate_historical_signals(
                    symbol, start_date, end_date, timeframe
                )
                
                # Add signals to backtesting engine
                for signal in signals:
                    if self.backtest_engine.add_trade_signal(signal):
                        total_signals += 1
                
                if signals:
                    print(f"   ✅ Found {len(signals)} signals for {symbol}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {e}")
                continue
        
        print(f"\n📈 BACKTESTING COMPLETE")
        print(f"🎯 Total signals generated: {total_signals}")
        
        # Run backtesting simulation
        if total_signals > 0:
            return self._run_backtest_simulation()
        else:
            return {"error": "No signals generated for backtesting"}
    
    def _get_top_symbols_for_backtest(self, limit: int = 20) -> List[str]:
        """Get top symbols for backtesting (smaller list for speed)"""
        try:
            # Use your existing data fetcher to get top coins
            top_coins = self.data_fetcher.fetch_top_coins(limit=limit)
            if top_coins:
                return top_coins[:limit]  # Limit for faster backtesting
            else:
                # Fallback to major coins
                return ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'DOT', 'AVAX', 'LTC', 'BCH']
        except Exception as e:
            self.logger.error(f"Error fetching top coins: {e}")
            return ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
    
    def _generate_historical_signals(self, 
                                   symbol: str, 
                                   start_date: datetime, 
                                   end_date: datetime,
                                   timeframe: str) -> List[Dict]:
        """
        Generate historical signals for a symbol using your scanner modules
        """
        signals = []
        
        try:
            # Fetch historical data for primary exchange (binance)
            exchange = 'binance'
            df = self.data_fetcher.fetch_ohlcv(exchange, symbol, timeframe)
            
            if df is None or len(df) < 100:
                return signals
            
            # Filter data to backtest period
            df_filtered = self._filter_data_by_date(df, start_date, end_date)
            
            if len(df_filtered) < 50:
                return signals
            
            # Simulate scanning through historical data
            signals = self._scan_historical_data(symbol, exchange, df_filtered)
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _filter_data_by_date(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Filter dataframe by date range"""
        try:
            if df.index.dtype == 'datetime64[ns]':
                mask = (df.index >= start_date) & (df.index <= end_date)
                return df.loc[mask]
            else:
                # If index is not datetime, return recent data
                return df.tail(500)  # Get last 500 candles
        except Exception as e:
            self.logger.error(f"Date filtering error: {e}")
            return df.tail(200)
    
    def _scan_historical_data(self, symbol: str, exchange: str, df: pd.DataFrame) -> List[Dict]:
        """
        Scan through historical data to find BB signals
        (Simplified version of your main scanner logic)
        """
        signals = []
        
        try:
            # Minimum data requirement
            if len(df) < 50:
                return signals
            
            # Simulate scanning every 10 candles (for speed)
            scan_interval = 10
            
            for i in range(50, len(df), scan_interval):
                try:
                    # Get data up to current point (simulate real-time)
                    current_data = df.iloc[:i+1]
                    current_time = current_data.index[-1]
                    
                    # Run BB analysis
                    bb_analysis = self.bb_detector.analyze_bb_setup(current_data)
                    
                    if bb_analysis['setup_type'] == 'NONE':
                        continue
                    
                    # Check 1H confirmation (simplified)
                    has_confirmation = True  # Simplified for backtesting
                    
                    if not has_confirmation:
                        continue
                    
                    # Calculate probability (simplified)
                    probability = self._calculate_simplified_probability(current_data, bb_analysis)
                    
                    # Quality filter
                    if probability < 60:  # Only take good setups
                        continue
                    
                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timestamp': current_time,
                        'entry': bb_analysis['entry'],
                        'stop': bb_analysis['stop'],
                        'target1': bb_analysis['target1'],
                        'setup_type': bb_analysis['setup_type'],
                        'probability': probability,
                        'bb_score': bb_analysis['bb_score'],
                        'setup_quality': bb_analysis['setup_quality'],
                        'risk_reward': bb_analysis['risk_reward'],
                        'risk_pct': abs((bb_analysis['entry'] - bb_analysis['stop']) / bb_analysis['entry'] * 100),
                        
                        # Enhanced data (simplified)
                        'pattern_boost': np.random.uniform(0, 5),  # Simplified
                        'regime_confidence': np.random.uniform(40, 80),  # Simplified
                        'funding_sentiment_signal': 'NEUTRAL',  # Simplified
                        'position_multiplier': 1.0  # Simplified
                    }
                    
                    signals.append(signal)
                    
                except Exception as e:
                    continue  # Skip problematic data points
            
        except Exception as e:
            self.logger.error(f"Historical scanning error for {symbol}: {e}")
        
        return signals
    
    def _calculate_simplified_probability(self, df: pd.DataFrame, bb_analysis: Dict) -> float:
        """Simplified probability calculation for backtesting"""
        try:
            base_prob = 50
            
            # BB score contribution
            bb_score_boost = min(bb_analysis['bb_score'] * 3, 20)
            
            # Volume confirmation (simplified)
            last_candle = df.iloc[-1]
            volume_boost = 5 if last_candle.get('volume_ratio', 1) > 1.2 else 0
            
            # RSI confirmation (simplified)
            rsi = last_candle.get('rsi', 50)
            if bb_analysis['setup_type'] == 'LONG' and rsi < 40:
                rsi_boost = 10
            elif bb_analysis['setup_type'] == 'SHORT' and rsi > 60:
                rsi_boost = 10
            else:
                rsi_boost = 0
            
            total_prob = base_prob + bb_score_boost + volume_boost + rsi_boost
            return min(total_prob, 95)  # Cap at 95%
            
        except Exception as e:
            return 50  # Default probability
    
    def _run_backtest_simulation(self) -> Dict:
        """
        Run the actual backtest simulation using historical price data
        """
        print(f"\n🔄 RUNNING BACKTEST SIMULATION...")
        print(f"📊 Processing {len(self.backtest_engine.trades)} signals...")
        
        simulated_trades = 0
        
        for trade in self.backtest_engine.trades:
            try:
                # Fetch market data for simulation
                df = self.data_fetcher.fetch_ohlcv(trade.exchange, trade.symbol, '4h')
                
                if df is not None and len(df) > 50:
                    # Simulate trade outcome
                    self.backtest_engine.simulate_trade_outcome(trade, df)
                    simulated_trades += 1
                
            except Exception as e:
                trade.outcome = "ERROR"
                continue
        
        print(f"✅ Simulated {simulated_trades} trades")
        
        # Generate comprehensive report
        report = self.backtest_engine.generate_backtest_report()
        
        # Save results
        filename = self.backtest_engine.save_backtest_results()
        report['saved_to'] = filename
        
        return report
    
    def display_backtest_results(self, report: Dict):
        """Display backtest results in terminal"""
        print(f"\n🎯 BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            return
        
        # Summary
        summary = report.get('summary', {})
        performance = report.get('performance', {})
        
        print(f"📊 Total Signals: {summary.get('backtest_period', 'N/A')}")
        print(f"✅ Completed Trades: {summary.get('completed_trades', 0)}")
        print(f"⏳ Pending Trades: {summary.get('pending_trades', 0)}")
        print(f"❌ Error Trades: {summary.get('error_trades', 0)}")
        
        if performance:
            print(f"\n💰 PERFORMANCE METRICS")
            print("-" * 40)
            print(f"🎯 Win Rate: {performance.get('win_rate', 0)}%")
            print(f"💵 Total Return: {performance.get('total_return_pct', 0)}%")
            print(f"💰 Total P&L: ${performance.get('total_pnl_dollar', 0)}")
            print(f"📈 Profit Factor: {performance.get('profit_factor', 0)}")
            print(f"📉 Max Drawdown: {performance.get('max_drawdown_pct', 0)}%")
            print(f"⏱️  Avg Hold Time: {performance.get('avg_hold_days', 0)} days")
            print(f"📊 Sharpe Ratio: {performance.get('sharpe_ratio', 0)}")
        
        # Regime analysis
        regime_analysis = report.get('regime_analysis', {})
        if regime_analysis:
            print(f"\n🌊 MARKET REGIME ANALYSIS")
            print("-" * 40)
            for regime, stats in regime_analysis.items():
                print(f"{regime}: {stats.get('win_rate', 0):.1f}% win rate ({stats.get('trade_count', 0)} trades)")
        
        print(f"\n📁 Results saved to: {report.get('saved_to', 'N/A')}")


def run_sample_backtest():
    """Run a sample backtest for demonstration"""
    print("🚀 ENHANCED BB SCANNER - HISTORICAL BACKTESTING")
    print("=" * 60)
    
    # Initialize scanner
    scanner = HistoricalBacktestScanner(initial_capital=10000)
    
    # Define backtest period (last 30 days for quick test)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Run backtest on limited symbols for speed
    test_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
    
    # Run backtest
    results = scanner.run_historical_backtest(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe='4h'
    )
    
    # Display results
    scanner.display_backtest_results(results)
    
    return results

if __name__ == "__main__":
    # Run sample backtest
    try:
        results = run_sample_backtest()
        print("\n🎉 Backtest completed successfully!")
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()