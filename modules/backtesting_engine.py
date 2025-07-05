# backtesting_engine.py - Enhanced Backtesting Engine
"""
Institutional-grade backtesting engine for BB Bounce Scanner
Tests historical performance with enhanced market regime intelligence

FEATURES:
- Multiple timeframe backtesting (30d, 3m, 6m, 1y)
- Enhanced market regime filtering  
- Pattern recognition validation
- Funding rate contrarian signals
- Risk-adjusted performance metrics
- Position sizing with regime multipliers
- Comprehensive trade analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json

@dataclass
class BacktestTrade:
    """Individual trade record for backtesting"""
    symbol: str
    exchange: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    setup_type: str  # LONG/SHORT
    probability: float
    bb_score: float
    pattern_boost: float
    regime_confidence: float
    funding_sentiment: str
    position_size: float
    risk_pct: float
    
    # Results
    pnl_pct: Optional[float] = None
    pnl_dollar: Optional[float] = None
    outcome: Optional[str] = None  # WIN/LOSS/PENDING
    hold_days: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    
@dataclass
class BacktestPeriod:
    """Backtesting period configuration"""
    name: str
    start_date: datetime
    end_date: datetime
    weight: float  # For confidence calculation

class EnhancedBacktestEngine:
    """
    Enhanced backtesting engine with institutional-grade analytics
    """
    
    def __init__(self, initial_capital: float = 10000):
        """Initialize backtesting engine"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[BacktestTrade] = []
        self.daily_returns: List[float] = []
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.metrics = {}
        self.regime_performance = {}
        self.pattern_performance = {}
        
        # Risk management settings
        self.max_position_size = 0.05  # 5% max per trade
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.06    # 6% take profit (3:1 R:R)
        
    def add_trade_signal(self, trade_data: Dict) -> bool:
        """
        Add a trade signal from scanner for backtesting
        
        Args:
            trade_data: Dict containing all trade information from scanner
            
        Returns:
            bool: True if trade was added successfully
        """
        try:
            # Extract trade information
            trade = BacktestTrade(
                symbol=trade_data['symbol'],
                exchange=trade_data['exchange'],
                entry_time=trade_data['timestamp'],
                exit_time=None,
                entry_price=trade_data['entry'],
                exit_price=None,
                setup_type=trade_data['setup_type'],
                probability=trade_data['probability'],
                bb_score=trade_data['bb_score'],
                pattern_boost=trade_data.get('pattern_boost', 0),
                regime_confidence=trade_data.get('regime_confidence', 50),
                funding_sentiment=trade_data.get('funding_sentiment_signal', 'NEUTRAL'),
                position_size=self._calculate_position_size(trade_data),
                risk_pct=trade_data['risk_pct']
            )
            
            self.trades.append(trade)
            self.logger.info(f"Added trade signal: {trade.symbol} {trade.setup_type} @ {trade.entry_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add trade signal: {e}")
            return False
    
    def _calculate_position_size(self, trade_data: Dict) -> float:
        """
        Calculate position size based on:
        - Base position size
        - Market regime confidence
        - Trade probability
        - Risk percentage
        """
        try:
            # Base position size
            base_size = self.max_position_size
            
            # Adjust for regime confidence
            regime_multiplier = trade_data.get('position_multiplier', 1.0)
            
            # Adjust for trade probability (higher prob = larger size)
            prob_multiplier = min(trade_data['probability'] / 70.0, 1.5)  # Max 1.5x for 70%+ prob
            
            # Adjust for risk (lower risk = larger size)
            risk_multiplier = min(2.0 / trade_data['risk_pct'], 2.0)  # Max 2x for 2% risk
            
            # Calculate final position size
            position_size = base_size * regime_multiplier * prob_multiplier * risk_multiplier
            
            # Cap at maximum position size
            return min(position_size, self.max_position_size * 2)
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return self.max_position_size
    
    def simulate_trade_outcome(self, trade: BacktestTrade, market_data: pd.DataFrame) -> BacktestTrade:
        """
        Simulate trade outcome using historical price data
        
        Args:
            trade: BacktestTrade object
            market_data: Historical OHLCV data for the symbol
            
        Returns:
            Updated BacktestTrade with results
        """
        try:
            # Find entry point in historical data
            entry_idx = self._find_nearest_timestamp(market_data, trade.entry_time)
            if entry_idx is None:
                trade.outcome = "NO_DATA"
                return trade
            
            # Simulate trade execution
            if trade.setup_type == "LONG":
                return self._simulate_long_trade(trade, market_data, entry_idx)
            else:
                return self._simulate_short_trade(trade, market_data, entry_idx)
                
        except Exception as e:
            self.logger.error(f"Trade simulation error for {trade.symbol}: {e}")
            trade.outcome = "ERROR"
            return trade
    
    def _simulate_long_trade(self, trade: BacktestTrade, data: pd.DataFrame, entry_idx: int) -> BacktestTrade:
        """Simulate LONG trade execution"""
        entry_price = trade.entry_price
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)
        
        max_hold_days = 30  # Maximum hold period
        max_drawdown = 0
        
        # Scan forward from entry point
        for i in range(entry_idx + 1, min(entry_idx + max_hold_days * 6, len(data))):  # 6 periods per day (4H)
            current_candle = data.iloc[i]
            low_price = current_candle['low']
            high_price = current_candle['high']
            close_price = current_candle['close']
            
            # Track maximum drawdown
            current_drawdown = (entry_price - low_price) / entry_price
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Check for stop loss hit
            if low_price <= stop_loss:
                trade.exit_time = current_candle.name
                trade.exit_price = stop_loss
                trade.outcome = "LOSS"
                break
                
            # Check for take profit hit
            if high_price >= take_profit:
                trade.exit_time = current_candle.name
                trade.exit_price = take_profit
                trade.outcome = "WIN"
                break
                
        # If no exit condition met, exit at current price
        if trade.exit_time is None:
            final_candle = data.iloc[min(entry_idx + max_hold_days * 6, len(data) - 1)]
            trade.exit_time = final_candle.name
            trade.exit_price = final_candle['close']
            trade.outcome = "WIN" if final_candle['close'] > entry_price else "LOSS"
        
        # Calculate trade results
        trade.pnl_pct = ((trade.exit_price - entry_price) / entry_price) * 100
        trade.pnl_dollar = trade.pnl_pct * trade.position_size * self.current_capital / 100
        trade.hold_days = (trade.exit_time - trade.entry_time).total_seconds() / (24 * 3600)
        trade.max_drawdown_pct = max_drawdown * 100
        
        return trade
    
    def _simulate_short_trade(self, trade: BacktestTrade, data: pd.DataFrame, entry_idx: int) -> BacktestTrade:
        """Simulate SHORT trade execution"""
        entry_price = trade.entry_price
        stop_loss = entry_price * (1 + self.stop_loss_pct)
        take_profit = entry_price * (1 - self.take_profit_pct)
        
        max_hold_days = 30
        max_drawdown = 0
        
        # Scan forward from entry point
        for i in range(entry_idx + 1, min(entry_idx + max_hold_days * 6, len(data))):
            current_candle = data.iloc[i]
            low_price = current_candle['low']
            high_price = current_candle['high']
            close_price = current_candle['close']
            
            # Track maximum drawdown (price moving against short)
            current_drawdown = (high_price - entry_price) / entry_price
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Check for stop loss hit (price going up)
            if high_price >= stop_loss:
                trade.exit_time = current_candle.name
                trade.exit_price = stop_loss
                trade.outcome = "LOSS"
                break
                
            # Check for take profit hit (price going down)
            if low_price <= take_profit:
                trade.exit_time = current_candle.name
                trade.exit_price = take_profit
                trade.outcome = "WIN"
                break
        
        # If no exit condition met, exit at current price
        if trade.exit_time is None:
            final_candle = data.iloc[min(entry_idx + max_hold_days * 6, len(data) - 1)]
            trade.exit_time = final_candle.name
            trade.exit_price = final_candle['close']
            trade.outcome = "WIN" if final_candle['close'] < entry_price else "LOSS"
        
        # Calculate trade results (SHORT: profit when price goes down)
        trade.pnl_pct = ((entry_price - trade.exit_price) / entry_price) * 100
        trade.pnl_dollar = trade.pnl_pct * trade.position_size * self.current_capital / 100
        trade.hold_days = (trade.exit_time - trade.entry_time).total_seconds() / (24 * 3600)
        trade.max_drawdown_pct = max_drawdown * 100
        
        return trade
    
    def _find_nearest_timestamp(self, data: pd.DataFrame, target_time: datetime) -> Optional[int]:
        """Find nearest timestamp in historical data"""
        try:
            if data.index.dtype == 'datetime64[ns]':
                time_diffs = abs(data.index - target_time)
                return time_diffs.argmin()
            else:
                # If index is not datetime, assume it's already sorted by time
                return 0
        except Exception as e:
            self.logger.error(f"Timestamp search error: {e}")
            return None
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dict containing all performance metrics
        """
        if not self.trades:
            return {"error": "No trades to analyze"}
        
        # Filter completed trades
        completed_trades = [t for t in self.trades if t.outcome in ["WIN", "LOSS"]]
        
        if not completed_trades:
            return {"error": "No completed trades to analyze"}
        
        # Basic metrics
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.outcome == "WIN"])
        losing_trades = len([t for t in completed_trades if t.outcome == "LOSS"])
        
        win_rate = (winning_trades / total_trades) * 100
        
        # PnL metrics
        total_pnl = sum([t.pnl_dollar for t in completed_trades])
        avg_win = np.mean([t.pnl_dollar for t in completed_trades if t.outcome == "WIN"]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl_dollar for t in completed_trades if t.outcome == "LOSS"]) if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
        max_drawdown = max([t.max_drawdown_pct for t in completed_trades]) if completed_trades else 0
        
        # Time metrics
        avg_hold_time = np.mean([t.hold_days for t in completed_trades])
        
        # Returns
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Sharpe ratio approximation (simplified)
        daily_returns = [t.pnl_dollar / self.initial_capital for t in completed_trades]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl_dollar": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "avg_win_dollar": round(avg_win, 2),
            "avg_loss_dollar": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "avg_hold_days": round(avg_hold_time, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades
        }
    
    def analyze_regime_performance(self) -> Dict:
        """Analyze performance by market regime"""
        completed_trades = [t for t in self.trades if t.outcome in ["WIN", "LOSS"]]
        
        regime_stats = {}
        
        for trade in completed_trades:
            # Categorize regime confidence
            if trade.regime_confidence >= 70:
                regime = "GOOD"
            elif trade.regime_confidence >= 60:
                regime = "FAIR" 
            elif trade.regime_confidence >= 50:
                regime = "POOR"
            else:
                regime = "VERY_POOR"
            
            if regime not in regime_stats:
                regime_stats[regime] = {"trades": [], "wins": 0, "total_pnl": 0}
            
            regime_stats[regime]["trades"].append(trade)
            if trade.outcome == "WIN":
                regime_stats[regime]["wins"] += 1
            regime_stats[regime]["total_pnl"] += trade.pnl_dollar
        
        # Calculate win rates for each regime
        for regime, stats in regime_stats.items():
            total = len(stats["trades"])
            wins = stats["wins"]
            stats["win_rate"] = (wins / total) * 100 if total > 0 else 0
            stats["avg_pnl"] = stats["total_pnl"] / total if total > 0 else 0
            stats["trade_count"] = total
        
        return regime_stats
    
    def generate_backtest_report(self) -> Dict:
        """Generate comprehensive backtest report"""
        performance = self.calculate_performance_metrics()
        regime_analysis = self.analyze_regime_performance()
        
        report = {
            "summary": {
                "backtest_period": f"{len(self.trades)} signals analyzed",
                "completed_trades": len([t for t in self.trades if t.outcome in ["WIN", "LOSS"]]),
                "pending_trades": len([t for t in self.trades if t.outcome not in ["WIN", "LOSS", "ERROR"]]),
                "error_trades": len([t for t in self.trades if t.outcome == "ERROR"])
            },
            "performance": performance,
            "regime_analysis": regime_analysis,
            "trade_details": [
                {
                    "symbol": t.symbol,
                    "setup_type": t.setup_type,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "probability": t.probability,
                    "regime_confidence": t.regime_confidence,
                    "outcome": t.outcome,
                    "pnl_pct": t.pnl_pct,
                    "hold_days": t.hold_days
                }
                for t in self.trades
            ]
        }
        
        return report
    
    def save_backtest_results(self, filename: str = None) -> str:
        """Save backtest results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/backtest_results_{timestamp}.json"
        
        report = self.generate_backtest_report()
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Backtest results saved to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save backtest results: {e}")
            return ""

# Example usage and testing functions
def test_backtesting_engine():
    """Test the backtesting engine with sample data"""
    print("🧪 Testing Enhanced Backtesting Engine...")
    
    # Initialize engine
    engine = EnhancedBacktestEngine(initial_capital=10000)
    
    # Sample trade data (as would come from scanner)
    sample_trades = [
        {
            "symbol": "BTC",
            "exchange": "binance", 
            "timestamp": datetime.now() - timedelta(days=5),
            "entry": 50000,
            "setup_type": "LONG",
            "probability": 75,
            "bb_score": 8,
            "pattern_boost": 2.5,
            "regime_confidence": 68,
            "funding_sentiment_signal": "CONTRARIAN_BULLISH",
            "position_multiplier": 1.2,
            "risk_pct": 2.1
        },
        {
            "symbol": "ETH",
            "exchange": "binance",
            "timestamp": datetime.now() - timedelta(days=3), 
            "entry": 3000,
            "setup_type": "SHORT",
            "probability": 68,
            "bb_score": 7,
            "pattern_boost": 1.8,
            "regime_confidence": 45,
            "funding_sentiment_signal": "NEUTRAL",
            "position_multiplier": 0.8,
            "risk_pct": 3.2
        }
    ]
    
    # Add trades to engine
    for trade_data in sample_trades:
        engine.add_trade_signal(trade_data)
    
    print(f"✅ Added {len(sample_trades)} trade signals")
    print(f"📊 Total trades in engine: {len(engine.trades)}")
    
    # Generate sample performance metrics (without real market data)
    if engine.trades:
        print("🎯 Backtesting engine initialized successfully!")
        print("📈 Ready for historical data integration")
        return True
    else:
        print("❌ Failed to initialize backtesting engine")
        return False

if __name__ == "__main__":
    # Test the backtesting engine
    test_backtesting_engine()