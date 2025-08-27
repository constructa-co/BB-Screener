#!/usr/bin/env python3
"""
Fair Value Gap Logger - Production Implementation
Zero-regression approach with proven patterns from 16 scanners
"""

import os
import json
import math
import time
import gc
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import psycopg2
import psycopg2.extras

DEFAULT_EXPIRE_HOURS = {
    '1m': 4,
    '5m': 8,
    '15m': 12,
    '1h': 24,
    '4h': 72
}

def _to_decimal(val):
    """Safe decimal conversion"""
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        return float(val)
    except:
        return None

class FairValueGapLogger:
    def __init__(self, timeframe='1m', dsn=None):
        self.timeframe = timeframe
        self.dsn = dsn or os.getenv("DATABASE_URL")
        
        # .env fallback
        if not self.dsn:
            env_file = '/opt/bb-screener/.env'
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('DATABASE_URL='):
                            self.dsn = line.split('=', 1)[1].strip().strip('"\'')
                            break
        
        if not self.dsn:
            raise RuntimeError("DATABASE_URL not found")
        
        # Circuit breaker
        self._fail_count = 0
        self._max_failures = 3
        self._cooldown_until = None
        
        # Memory management
        self._last_cleanup = time.time()
        self._batch_count = 0
        
        print(f"[FVGLogger] Initialized for {timeframe}")
    
    def _conn(self):
        """Get database connection"""
        return psycopg2.connect(self.dsn)
    
    def _make_signal_id(self, symbol: str, gap_type: str, gap_high: float, 
                       gap_low: float, detected_at: datetime) -> str:
        """Create deterministic signal ID"""
        # Include price levels to make unique
        bucket = detected_at.strftime("%Y%m%d%H")
        if self.timeframe in ['1m', '5m']:
            bucket = detected_at.strftime("%Y%m%d%H%M")
        
        id_string = f"fvg::{symbol}::{self.timeframe}::{gap_type}::{gap_high:.2f}::{gap_low:.2f}::{bucket}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:40]
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operation"""
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.utcnow()).total_seconds()
            print(f"[FVGLogger] Circuit breaker OPEN ({remaining:.0f}s remaining)")
            return False
        return True
    
    def _check_memory(self):
        """Periodic memory cleanup"""
        if time.time() - self._last_cleanup > 1800:  # 30 minutes
            gc.collect()
            self._last_cleanup = time.time()
            print(f"[FVGLogger] Memory cleanup performed")
    
    def log_fvg(self, symbol: str, fvg_data: dict) -> bool:
        """Log single FVG signal"""
        if not self._check_circuit_breaker():
            return False
        
        detected_at = datetime.utcnow()
        gap_type = str(fvg_data.get('gap_type', fvg_data.get('direction', 'UNKNOWN'))).upper()
        gap_high = _to_decimal(fvg_data.get('gap_high', fvg_data.get('gap_top')))
        gap_low = _to_decimal(fvg_data.get('gap_low', fvg_data.get('gap_bottom')))
        
        if not all([gap_high, gap_low, gap_type in ['BULLISH', 'BEARISH']]):
            print(f"[FVGLogger] Invalid FVG data for {symbol}")
            return False
        
        signal_id = self._make_signal_id(symbol, gap_type, gap_high, gap_low, detected_at)
        
        # Calculate expiry based on timeframe
        expire_hours = DEFAULT_EXPIRE_HOURS.get(self.timeframe, 24)
        
        # Build record
        record = {
            "signal_id": signal_id,
            "symbol": symbol.upper(),
            "timeframe": self.timeframe,
            "detected_at": detected_at,
            "gap_type": gap_type,
            "gap_high": gap_high,
            "gap_low": gap_low,
            "gap_size": gap_high - gap_low if gap_high and gap_low else None,
            "gap_size_pct": _to_decimal(fvg_data.get('gap_size_pct')),
            "current_price": _to_decimal(fvg_data.get('current_price')),
            "entry_price": _to_decimal(fvg_data.get('entry_price', fvg_data.get('entry'))),
            "stop_loss": _to_decimal(fvg_data.get('stop_loss', fvg_data.get('stop'))),
            "target_1": _to_decimal(fvg_data.get('target_1', fvg_data.get('target'))),
            "target_2": _to_decimal(fvg_data.get('target_2')),
            "target_3": _to_decimal(fvg_data.get('target_3')),
            "risk_reward_1": _to_decimal(fvg_data.get('risk_reward_1', fvg_data.get('rr1'))),
            "risk_reward_2": _to_decimal(fvg_data.get('risk_reward_2', fvg_data.get('rr2'))),
            "risk_reward_3": _to_decimal(fvg_data.get('risk_reward_3', fvg_data.get('rr3'))),
            "fib_level": _to_decimal(fvg_data.get('fib_level', fvg_data.get('fibonacci_level'))),
            "fib_confluence": bool(fvg_data.get('fib_confluence', fvg_data.get('fibonacci_confluence', False))),
            "fib_confluence_score": fvg_data.get('fib_confluence_score'),
            "setup_score": int(fvg_data.get('setup_score', fvg_data.get('score', 0))),
            "volume_at_gap": fvg_data.get('volume_at_gap', fvg_data.get('volume')),
            "volume_confirmation": bool(fvg_data.get('volume_confirmation', False)),
            "momentum_confirmation": bool(fvg_data.get('momentum_confirmation', False)),
            "gap_status": 'OPEN',
            "fill_percentage": 0,
            "gap_age_minutes": fvg_data.get('gap_age'),
            "entry_timing": fvg_data.get('entry_timing'),
            "current_distance_pct": _to_decimal(fvg_data.get('current_distance_pct')),
            "risk_pct": _to_decimal(fvg_data.get('risk_pct')),
            "swing_high": _to_decimal(fvg_data.get('swing_high')),
            "swing_low": _to_decimal(fvg_data.get('swing_low')),
            "fib_levels": fvg_data.get('fib_levels'),
            "target_levels": fvg_data.get('target_levels'),
            "expires_at": detected_at + timedelta(hours=expire_hours),
            "scanner_version": "1.0.0",
            "algorithm_parameters": json.dumps(fvg_data),
            "source": f"fvg_{self.timeframe}_scanner"
        }
        
        # SQL with upsert
        insert_sql = """
            INSERT INTO other_scanners.fvg_signals (
                signal_id, symbol, timeframe, detected_at,
                gap_type, gap_high, gap_low, gap_size, gap_size_pct,
                current_price, entry_price, stop_loss, target_1, target_2, target_3,
                risk_reward_1, risk_reward_2, risk_reward_3, fib_level, fib_confluence,
                fib_confluence_score, setup_score, volume_at_gap, volume_confirmation,
                momentum_confirmation, gap_status, fill_percentage, gap_age_minutes,
                entry_timing, current_distance_pct, risk_pct, swing_high, swing_low,
                fib_levels, target_levels, expires_at, scanner_version, algorithm_parameters, source
            ) VALUES (
                %(signal_id)s, %(symbol)s, %(timeframe)s, %(detected_at)s,
                %(gap_type)s, %(gap_high)s, %(gap_low)s, %(gap_size)s, %(gap_size_pct)s,
                %(current_price)s, %(entry_price)s, %(stop_loss)s, %(target_1)s, %(target_2)s, %(target_3)s,
                %(risk_reward_1)s, %(risk_reward_2)s, %(risk_reward_3)s, %(fib_level)s, %(fib_confluence)s,
                %(fib_confluence_score)s, %(setup_score)s, %(volume_at_gap)s, %(volume_confirmation)s,
                %(momentum_confirmation)s, %(gap_status)s, %(fill_percentage)s, %(gap_age_minutes)s,
                %(entry_timing)s, %(current_distance_pct)s, %(risk_pct)s, %(swing_high)s, %(swing_low)s,
                %(fib_levels)s, %(target_levels)s, %(expires_at)s, %(scanner_version)s, %(algorithm_parameters)s, %(source)s
            )
            ON CONFLICT (signal_id) DO UPDATE SET
                detected_at = EXCLUDED.detected_at,
                expires_at = EXCLUDED.expires_at
        """
        
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, record)
                    conn.commit()
            
            self._fail_count = 0
            print(f"[FVGLogger] ✅ Logged: {symbol} {gap_type} (Score: {record['setup_score']})")
            return True
            
        except Exception as e:
            self._fail_count += 1
            if self._fail_count >= self._max_failures:
                self._cooldown_until = datetime.utcnow() + timedelta(minutes=5)
                print(f"[FVGLogger] 🔴 Circuit breaker triggered: {e}")
            else:
                print(f"[FVGLogger] ⚠️ Failed ({self._fail_count}/{self._max_failures}): {e}")
            return False
        
        finally:
            self._batch_count += 1
            if self._batch_count % 20 == 0:
                self._check_memory()
