#!/usr/bin/env python3
"""
Trend Following Logger - Production Implementation
Combines best practices from all research reviews
File: database_and_logging/trend_following_logger.py
"""

import os
import json
import math
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import psycopg2
import psycopg2.extras

DEFAULT_EXPIRE_MINUTES = 120

def _to_decimal(val):
    """Convert to float safely (ChatGPT's approach)"""
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
    except Exception:
        return None

class TrendDataExtractor:
    """Robust field extraction (Perplexity's enhancement)"""
    
    def extract_from_opportunity(self, opportunity: dict, symbol: str) -> dict:
        """Extract data regardless of scanner output format"""
        # Handle your scanner's specific format
        trend = opportunity.get("trend", {}) or {}
        pullback = opportunity.get("pullback", {}) or {}
        trade = opportunity.get("trade", {}) or {}
        
        # Determine signal type
        direction = (trade.get("direction") or "").upper()
        if direction == "LONG":
            signal_type = "BULLISH"
        elif direction == "SHORT":
            signal_type = "BEARISH"
        else:
            signal_type = "NEUTRAL"
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'trend_direction': trend.get("direction", "SIDEWAYS"),
            'trend_strength': _to_decimal(trend.get("strength")),
            'momentum_score': _to_decimal(trend.get("price_change_20")),
            'volume_trend': "ABOVE_AVG" if trend.get("macd_confirmation") else None,
            'entry_price': _to_decimal(trade.get("entry_price")),
            'stop_loss': _to_decimal(trade.get("stop_loss")),
            'targets': trade.get("targets", {}),
            'risk_pct': _to_decimal(trade.get("risk_pct")),
            'entry_timing': trade.get("entry_timing"),
            'quality_score': int(opportunity.get("quality_score", 0))
        }

class TrendFollowingLogger:
    def __init__(self, timeframe='1h', dsn=None):
        self.timeframe = timeframe
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError("DATABASE_URL is not set")
        
        # Circuit breaker (Claude's robustness)
        self._fail_count = 0
        self._max_failures = 3
        self._cooldown_until = None
        
        # Memory management
        self._last_cleanup = time.time()
        
        # Data extractor
        self.extractor = TrendDataExtractor()
        
        # Ensure table exists
        self._ensure_table()
    
    def _conn(self):
        """Get database connection"""
        return psycopg2.connect(self.dsn)
    
    def _ensure_table(self):
        """Create schema/table if not exists"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        sql_path = os.path.join(base_dir, "scripts", "trend_following_integration", 
                                "setup_trend_following_database.sql")
        
        if os.path.exists(sql_path):
            try:
                with self._conn() as conn, conn.cursor() as cur:
                    with open(sql_path, "r", encoding="utf-8") as f:
                        cur.execute(f.read())
                print("[TrendFollowingLogger] Database schema ensured")
            except Exception as e:
                print(f"[TrendFollowingLogger] Schema creation warning: {e}")
    
    def _make_signal_id(self, symbol: str, primary_tf: str, detected_at: datetime) -> str:
        """Deterministic signal ID (ChatGPT's approach)"""
        bucket = detected_at.strftime("%Y%m%d%H%M")
        return f"tf::{symbol}::{primary_tf}::{bucket}"
    
    def _check_circuit_breaker(self) -> bool:
        """Simple circuit breaker (Claude's pattern, simplified)"""
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            return False
        return True
    
    def _check_memory(self):
        """Periodic memory cleanup (Claude's optimization)"""
        if time.time() - self._last_cleanup > 1800:  # 30 minutes
            gc.collect()
            self._last_cleanup = time.time()
    
    def log_opportunity(self, symbol: str, opportunity: dict, primary_tf: str = "4h") -> bool:
        """
        Log a single trend-following opportunity
        Returns True if successful, False otherwise
        """
        # Circuit breaker check
        if not self._check_circuit_breaker():
            print("[TrendFollowingLogger] Circuit breaker open, skipping")
            return False
        
        detected_at = datetime.utcnow()
        signal_id = self._make_signal_id(symbol, primary_tf.lower(), detected_at)
        
        # Extract data robustly
        data = self.extractor.extract_from_opportunity(opportunity, symbol)
        
        # Extract targets safely
        targets = data.get('targets', {})
        target_1 = _to_decimal(targets.get("target_1", {}).get("price")) if targets else None
        target_2 = _to_decimal(targets.get("target_2", {}).get("price")) if targets else None
        target_3 = _to_decimal(targets.get("target_3", {}).get("price")) if targets else None
        rr_1 = _to_decimal(targets.get("target_1", {}).get("rr")) if targets else None
        rr_2 = _to_decimal(targets.get("target_2", {}).get("rr")) if targets else None
        rr_3 = _to_decimal(targets.get("target_3", {}).get("rr")) if targets else None
        
        # Prepare record
        record = {
            "symbol": symbol,
            "timeframe": self.timeframe.lower(),
            "signal_id": signal_id,
            "signal_type": data["signal_type"],
            "trend_direction": data["trend_direction"],
            "trend_strength": data["trend_strength"],
            "momentum_score": data["momentum_score"],
            "volume_trend": data["volume_trend"],
            "ma_20": None,
            "ma_50": None,
            "ma_100": None,
            "price_to_ma50_distance": None,
            "current_price": data.get("entry_price"),  # Use entry as current
            "atr_value": None,
            "entry_price": data["entry_price"],
            "stop_loss": data["stop_loss"],
            "target_1": target_1,
            "target_2": target_2,
            "target_3": target_3,
            "risk_reward_1": rr_1,
            "risk_reward_2": rr_2,
            "risk_reward_3": rr_3,
            "risk_pct": data["risk_pct"],
            "entry_timing": data["entry_timing"],
            "confidence_score": _to_decimal(data.get("trend_strength", 50)),
            "quality_score": data["quality_score"],
            "scanner_version": "1.0.0",
            "algorithm_parameters": json.dumps({
                "trend": opportunity.get("trend", {}),
                "pullback": opportunity.get("pullback", {}),
                "trade": opportunity.get("trade", {}),
                "scanner_timeframe": primary_tf
            }),
            "detected_at": detected_at,
            "expires_at": detected_at + timedelta(minutes=DEFAULT_EXPIRE_MINUTES),
            "post_mortem_status": "PENDING",
            "actual_outcome": None,
            "profit_loss_percentage": None,
            "post_mortem_notes": None
        }
        
        # SQL with upsert
        insert_sql = """
            INSERT INTO other_scanners.trend_following_signals (
                symbol, timeframe, signal_id, signal_type,
                trend_direction, trend_strength, momentum_score, volume_trend,
                ma_20, ma_50, ma_100, price_to_ma50_distance,
                current_price, atr_value,
                entry_price, stop_loss, target_1, target_2, target_3,
                risk_reward_1, risk_reward_2, risk_reward_3, risk_pct, entry_timing,
                confidence_score, quality_score,
                scanner_version, algorithm_parameters, detected_at, expires_at,
                post_mortem_status, actual_outcome, profit_loss_percentage, post_mortem_notes
            ) VALUES (
                %(symbol)s, %(timeframe)s, %(signal_id)s, %(signal_type)s,
                %(trend_direction)s, %(trend_strength)s, %(momentum_score)s, %(volume_trend)s,
                %(ma_20)s, %(ma_50)s, %(ma_100)s, %(price_to_ma50_distance)s,
                %(current_price)s, %(atr_value)s,
                %(entry_price)s, %(stop_loss)s, %(target_1)s, %(target_2)s, %(target_3)s,
                %(risk_reward_1)s, %(risk_reward_2)s, %(risk_reward_3)s, %(risk_pct)s, %(entry_timing)s,
                %(confidence_score)s, %(quality_score)s,
                %(scanner_version)s, %(algorithm_parameters)s, %(detected_at)s, %(expires_at)s,
                %(post_mortem_status)s, %(actual_outcome)s, %(profit_loss_percentage)s, %(post_mortem_notes)s
            )
            ON CONFLICT (signal_id) DO UPDATE SET
                signal_type = EXCLUDED.signal_type,
                trend_strength = EXCLUDED.trend_strength,
                confidence_score = EXCLUDED.confidence_score,
                quality_score = EXCLUDED.quality_score,
                algorithm_parameters = EXCLUDED.algorithm_parameters,
                expires_at = EXCLUDED.expires_at
        """
        
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(insert_sql, record)
                self._fail_count = 0  # Reset on success
                print(f"[TrendFollowingLogger] Logged: {signal_id}")
                return True
                
        except Exception as e:
            self._fail_count += 1
            if self._fail_count >= self._max_failures:
                self._cooldown_until = datetime.utcnow() + timedelta(minutes=5)
                print(f"[TrendFollowingLogger] Circuit breaker opened: {e}")
            else:
                print(f"[TrendFollowingLogger] Insert failed ({self._fail_count}/{self._max_failures}): {e}")
            return False
        
        finally:
            self._check_memory()
    
    def close(self):
        """Cleanup resources"""
        print("[TrendFollowingLogger] Logger closed")
