#!/usr/bin/env python3
"""
Wyckoff Logger - Production Implementation
Following proven patterns from Trend Following and Supply & Demand
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

DEFAULT_EXPIRE_HOURS = 12  # Wyckoff setups last 3-12 hours

def _to_decimal(val):
    """Convert to float safely"""
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
    except Exception:
        return None

class WyckoffDataExtractor:
    """Extract Wyckoff-specific data from scanner output"""
    
    def extract_from_setup(self, setup: dict, symbol: str) -> dict:
        """Extract data from Wyckoff scanner output format"""
        
        # Expected structure from R0 scanner
        # setup = {
        #     'phase': 'ACCUMULATION',
        #     'pattern': 'SPRING',
        #     'entry': 150.25,
        #     'stop_loss': 148.50,
        #     'target_1': 153.00,
        #     'target_2': 156.00,
        #     'risk_reward_1': 1.5,
        #     'risk_reward_2': 3.0,
        #     'setup_score': 75,
        #     'volume_confirmation': 1.3,
        #     'pattern_duration': 12,
        #     'trade_direction': 'LONG',
        #     'entry_signal': 'WAIT_FOR_TEST',
        #     'wait_condition': 'Test of spring level'
        # }
        
        return {
            'symbol': symbol,
            'phase': setup.get('phase', 'UNKNOWN'),
            'pattern_type': setup.get('pattern', 'UNKNOWN'),
            'pattern_duration': setup.get('pattern_duration'),
            'trade_direction': setup.get('trade_direction', 'NEUTRAL'),
            'entry_price': _to_decimal(setup.get('entry')),
            'stop_loss': _to_decimal(setup.get('stop_loss')),
            'target_1': _to_decimal(setup.get('target_1')),
            'target_2': _to_decimal(setup.get('target_2')),
            'risk_reward_1': _to_decimal(setup.get('risk_reward_1')),
            'risk_reward_2': _to_decimal(setup.get('risk_reward_2')),
            'setup_score': int(setup.get('setup_score', 0)),
            'volume_confirmation': _to_decimal(setup.get('volume_confirmation')),
            'entry_signal': setup.get('entry_signal', 'IMMEDIATE'),
            'wait_condition': setup.get('wait_condition'),
            'current_price': _to_decimal(setup.get('current_price', setup.get('entry'))),
            'support_level': _to_decimal(setup.get('support')),
            'resistance_level': _to_decimal(setup.get('resistance'))
        }

class WyckoffLogger:
    def __init__(self, timeframe='1h', dsn=None):
        self.timeframe = timeframe
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            # Fallback to .env file reading
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('DATABASE_URL='):
                            self.dsn = line.split('=', 1)[1].strip().strip('"')
                            break
            except:
                pass
        
        if not self.dsn:
            raise RuntimeError("DATABASE_URL not found")
        
        # Circuit breaker
        self._fail_count = 0
        self._max_failures = 3
        self._cooldown_until = None
        
        # Memory management
        self._last_cleanup = time.time()
        
        # Data extractor
        self.extractor = WyckoffDataExtractor()
        
        # Ensure table exists
        self._ensure_table()
    
    def _conn(self):
        """Get database connection"""
        return psycopg2.connect(self.dsn)
    
    def _ensure_table(self):
        """Create schema/table if not exists"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        sql_path = os.path.join(base_dir, "scripts", "wyckoff_integration", 
                                "setup_wyckoff_database.sql")
        
        if os.path.exists(sql_path):
            try:
                with self._conn() as conn, conn.cursor() as cur:
                    with open(sql_path, "r", encoding="utf-8") as f:
                        cur.execute(f.read())
                print("[WyckoffLogger] Database schema ensured")
            except Exception as e:
                print(f"[WyckoffLogger] Schema creation warning: {e}")
    
    def _make_signal_id(self, symbol: str, pattern: str, detected_at: datetime) -> str:
        """Deterministic signal ID"""
        bucket = detected_at.strftime("%Y%m%d%H")  # Hour bucket for 1H timeframe
        return f"wyckoff::{symbol}::{pattern}::{bucket}"
    
    def _check_circuit_breaker(self) -> bool:
        """Circuit breaker check"""
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            return False
        return True
    
    def _check_memory(self):
        """Periodic memory cleanup"""
        if time.time() - self._last_cleanup > 1800:  # 30 minutes
            gc.collect()
            self._last_cleanup = time.time()
    
    def log_setup(self, symbol: str, setup: dict) -> bool:
        """
        Log a single Wyckoff setup
        Returns True if successful, False otherwise
        """
        # Circuit breaker check
        if not self._check_circuit_breaker():
            print("[WyckoffLogger] Circuit breaker open, skipping")
            return False
        
        detected_at = datetime.utcnow()
        
        # Extract data
        data = self.extractor.extract_from_setup(setup, symbol)
        signal_id = self._make_signal_id(symbol, data['pattern_type'], detected_at)
        
        # Prepare record
        record = {
            "symbol": symbol,
            "timeframe": self.timeframe,
            "signal_id": signal_id,
            "phase": data["phase"],
            "pattern_type": data["pattern_type"],
            "pattern_duration": data["pattern_duration"],
            "trade_direction": data["trade_direction"],
            "entry_price": data["entry_price"],
            "stop_loss": data["stop_loss"],
            "target_1": data["target_1"],
            "target_2": data["target_2"],
            "risk_reward_1": data["risk_reward_1"],
            "risk_reward_2": data["risk_reward_2"],
            "position_size": 1.0,  # Default, can be calculated
            "setup_score": data["setup_score"],
            "volume_confirmation": data["volume_confirmation"],
            "strength_score": _to_decimal(data.get("setup_score", 0)),  # Use setup_score as strength
            "entry_signal": data["entry_signal"],
            "wait_condition": data["wait_condition"],
            "current_price": data["current_price"],
            "support_level": data["support_level"],
            "resistance_level": data["resistance_level"],
            "scanner_version": "1.0.0",
            "algorithm_parameters": json.dumps({
                "phase": data["phase"],
                "pattern": data["pattern_type"],
                "full_setup": setup
            }),
            "detected_at": detected_at,
            "expires_at": detected_at + timedelta(hours=DEFAULT_EXPIRE_HOURS),
            "post_mortem_status": "PENDING",
            "actual_outcome": None,
            "profit_loss_percentage": None,
            "pattern_held": None,
            "post_mortem_notes": None
        }
        
        # SQL with upsert
        insert_sql = """
            INSERT INTO other_scanners.wyckoff_signals (
                symbol, timeframe, signal_id, phase, pattern_type, pattern_duration,
                trade_direction, entry_price, stop_loss, target_1, target_2,
                risk_reward_1, risk_reward_2, position_size,
                setup_score, volume_confirmation, strength_score,
                entry_signal, wait_condition,
                current_price, support_level, resistance_level,
                scanner_version, algorithm_parameters, detected_at, expires_at,
                post_mortem_status, actual_outcome, profit_loss_percentage, 
                pattern_held, post_mortem_notes
            ) VALUES (
                %(symbol)s, %(timeframe)s, %(signal_id)s, %(phase)s, %(pattern_type)s, %(pattern_duration)s,
                %(trade_direction)s, %(entry_price)s, %(stop_loss)s, %(target_1)s, %(target_2)s,
                %(risk_reward_1)s, %(risk_reward_2)s, %(position_size)s,
                %(setup_score)s, %(volume_confirmation)s, %(strength_score)s,
                %(entry_signal)s, %(wait_condition)s,
                %(current_price)s, %(support_level)s, %(resistance_level)s,
                %(scanner_version)s, %(algorithm_parameters)s, %(detected_at)s, %(expires_at)s,
                %(post_mortem_status)s, %(actual_outcome)s, %(profit_loss_percentage)s,
                %(pattern_held)s, %(post_mortem_notes)s
            )
            ON CONFLICT (signal_id) DO UPDATE SET
                setup_score = GREATEST(EXCLUDED.setup_score, other_scanners.wyckoff_signals.setup_score),
                volume_confirmation = EXCLUDED.volume_confirmation,
                expires_at = EXCLUDED.expires_at,
                algorithm_parameters = EXCLUDED.algorithm_parameters
        """
        
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(insert_sql, record)
                self._fail_count = 0  # Reset on success
                print(f"[WyckoffLogger] Logged: {signal_id} (Score: {data['setup_score']})")
                return True
                
        except Exception as e:
            self._fail_count += 1
            if self._fail_count >= self._max_failures:
                self._cooldown_until = datetime.utcnow() + timedelta(minutes=5)
                print(f"[WyckoffLogger] Circuit breaker opened: {e}")
            else:
                print(f"[WyckoffLogger] Insert failed ({self._fail_count}/{self._max_failures}): {e}")
            return False
        
        finally:
            self._check_memory()
