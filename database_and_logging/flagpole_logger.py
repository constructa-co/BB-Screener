#!/usr/bin/env python3
"""
Flagpole Pattern Logger - Production Implementation
Handles parsed pattern data from FlagpoleOutputParser
"""

import os
import json
import hashlib
import time
import gc
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import Json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('FlagpoleLogger')

class FlagpoleLogger:
    """Production-grade logger with circuit breaker and quality filtering"""
    
    def __init__(self, timeframe='5m', min_score=60):
        self.timeframe = timeframe
        self.min_score = min_score
        self.dsn = self._get_dsn()
        
        # Circuit breaker
        self._failures = 0
        self._max_failures = 3
        self._cooldown_until = None
        
        # Memory management
        self._last_gc = time.time()
        
        logger.info(f"Initialized for {timeframe} with min_score={min_score}")
    
    def _get_dsn(self) -> str:
        """Get database connection string"""
        dsn = os.getenv("DATABASE_URL")
        
        if not dsn:
            # Try loading from .env file
            env_path = '/opt/bb-screener/.env'
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('DATABASE_URL='):
                            dsn = line.split('=', 1)[1].strip().strip('"\'')
                            break
        
        if not dsn:
            raise RuntimeError("DATABASE_URL not found")
        
        return dsn
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operation"""
        if self._cooldown_until:
            if datetime.now(timezone.utc) < self._cooldown_until:
                remaining = (self._cooldown_until - datetime.now(timezone.utc)).total_seconds()
                logger.warning(f"Circuit breaker OPEN - {remaining:.0f}s remaining")
                return False
            else:
                # Reset after cooldown
                self._cooldown_until = None
                self._failures = 0
        return True
    
    def _make_signal_id(self, symbol: str, pattern_type: str, detected_at: datetime) -> str:
        """Create deterministic signal ID"""
        # Use 5-minute bucket to prevent duplicates within same period
        bucket = detected_at.strftime("%Y%m%d%H") + str(detected_at.minute // 5)
        key = f"flagpole::{symbol}::{self.timeframe}::{pattern_type}::{bucket}"
        return hashlib.sha256(key.encode()).hexdigest()[:48]
    
    def _clean_memory(self):
        """Periodic garbage collection"""
        if time.time() - self._last_gc > 1800:  # 30 minutes
            gc.collect()
            self._last_gc = time.time()
    
    def log_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Log a single parsed pattern to database
        Returns True if successful, False otherwise
        """
        
        # Circuit breaker check
        if not self._check_circuit_breaker():
            return False
        
        # Quality filter
        score = pattern.get('score', 0)
        if score < self.min_score:
            logger.info(f"Skipping {pattern.get('symbol')} - score {score} < {self.min_score}")
            return False
        
        detected_at = datetime.now(timezone.utc)
        symbol = pattern.get('symbol', 'UNKNOWN')
        pattern_type = pattern.get('pattern_type', 'UNKNOWN')
        
        signal_id = self._make_signal_id(symbol, pattern_type, detected_at)
        
        # Build record
        record = {
            'signal_id': signal_id,
            'symbol': symbol.upper(),
            'timeframe': self.timeframe,
            'detected_at': detected_at,
            'pattern_type': pattern_type,
            'pattern_details': pattern.get('pattern_details'),
            'direction': pattern.get('direction'),
            'current_price': pattern.get('current_price'),
            'breakout_level': pattern.get('breakout_level'),
            'target_price': pattern.get('target_price'),
            'stop_loss': pattern.get('stop_loss'),
            'potential_pct': pattern.get('potential_pct'),
            'risk_pct': pattern.get('risk_pct'),
            'risk_reward': pattern.get('risk_reward'),
            'pole_pct': pattern.get('pole_pct'),
            'vol_decline_pct': pattern.get('vol_decline_pct'),
            'slope_pct': pattern.get('slope_pct'),
            'age_candles': pattern.get('age_candles'),
            'score': score,
            'quality_indicators': pattern.get('quality_indicators', []),
            'quality_raw': pattern.get('quality_raw'),
            'is_ready': pattern.get('is_ready', False),
            'has_strong_vol': pattern.get('has_strong_vol', False),
            'has_fast_pole': pattern.get('has_fast_pole', False),
            'expires_at': detected_at + timedelta(hours=12),
            'scanner_version': '1.0.0',
            'raw_output': json.dumps(pattern),
            'source': f'flagpole_{self.timeframe}_scanner'
        }
        
        sql = """
            INSERT INTO other_scanners.flagpole_signals (
                signal_id, symbol, timeframe, detected_at,
                pattern_type, pattern_details, direction,
                current_price, breakout_level, target_price, stop_loss,
                potential_pct, risk_pct, risk_reward,
                pole_pct, vol_decline_pct, slope_pct, age_candles,
                score, quality_indicators, quality_raw,
                is_ready, has_strong_vol, has_fast_pole,
                expires_at, scanner_version, raw_output, source
            ) VALUES (
                %(signal_id)s, %(symbol)s, %(timeframe)s, %(detected_at)s,
                %(pattern_type)s, %(pattern_details)s, %(direction)s,
                %(current_price)s, %(breakout_level)s, %(target_price)s, %(stop_loss)s,
                %(potential_pct)s, %(risk_pct)s, %(risk_reward)s,
                %(pole_pct)s, %(vol_decline_pct)s, %(slope_pct)s, %(age_candles)s,
                %(score)s, %(quality_indicators)s, %(quality_raw)s,
                %(is_ready)s, %(has_strong_vol)s, %(has_fast_pole)s,
                %(expires_at)s, %(scanner_version)s, %(raw_output)s, %(source)s
            )
            ON CONFLICT (signal_id) DO UPDATE SET
                detected_at = EXCLUDED.detected_at,
                current_price = EXCLUDED.current_price,
                score = EXCLUDED.score
        """
        
        try:
            with psycopg2.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, record)
                    conn.commit()
            
            self._failures = 0  # Reset on success
            logger.info(f"✅ Logged: {symbol} {pattern_type} (Score: {score})")
            return True
            
        except Exception as e:
            self._failures += 1
            logger.error(f"❌ Failed to log {symbol}: {e}")
            
            # Activate circuit breaker if needed
            if self._failures >= self._max_failures:
                self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=5)
                logger.warning("Circuit breaker activated - 5 minute cooldown")
            
            return False
        
        finally:
            self._clean_memory()
    
    def log_patterns(self, patterns: List[Dict[str, Any]]) -> int:
        """
        Log multiple patterns
        Returns count of successfully logged patterns
        """
        logged = 0
        for pattern in patterns:
            if self.log_pattern(pattern):
                logged += 1
        
        logger.info(f"Logged {logged}/{len(patterns)} patterns")
        return logged
