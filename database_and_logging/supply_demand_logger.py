#!/usr/bin/env python3
"""
Supply & Demand Zone Logger - Production Implementation
Follows Fibonacci pattern with simplifications from ChatGPT review
"""

import os
import json
import time
import logging
import gc
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd

# Configuration
SCANNER_ID = os.getenv('SD_SCANNER_ID', 'supply_demand_01')

class SupplyDemandLogger:
    """Supply & Demand logger with complete isolation"""
    
    def __init__(self, timeframe: str = '1h'):
        self.timeframe = timeframe
        self.scanner_id = f"{SCANNER_ID}_{timeframe}"
        self.connection = None
        self.cursor = None
        
        # Check database URL
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Circuit breaker state
        self.failure_count = 0
        self.circuit_open_time = None
        self.circuit_threshold = 5
        self.recovery_timeout = 300
        
        # Memory management
        self.last_cleanup = time.time()
        self.cleanup_interval = 1800  # 30 minutes
        
        # Setup console logging only (no file handlers)
        self.logger = logging.getLogger(f'sd_{self.scanner_id}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize values for database insertion"""
        if value is None or pd.isna(value):
            return None
        
        # Handle numpy types
        if hasattr(value, 'item'):
            return self._sanitize_value(value.item())
        
        # Handle pandas types
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        
        # Handle float types with 8 decimal precision
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return None
            return round(float(value), 8)
        
        # Handle integers
        if isinstance(value, (int, np.integer)):
            return int(value)
        
        # Handle lists/arrays
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self._sanitize_value(v) for v in value]
        
        # Handle dictionaries
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        
        return value
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operation"""
        if self.failure_count >= self.circuit_threshold:
            if self.circuit_open_time:
                elapsed = time.time() - self.circuit_open_time
                if elapsed < self.recovery_timeout:
                    return False  # Circuit still open
                else:
                    # Try to recover
                    self.logger.info("Circuit breaker attempting recovery")
                    self.circuit_open_time = None
                    self.failure_count = self.circuit_threshold - 1
        return True
    
    def _on_success(self):
        """Reset circuit breaker on successful operation"""
        self.failure_count = 0
        self.circuit_open_time = None
    
    def _on_failure(self):
        """Handle operation failure"""
        self.failure_count += 1
        if self.failure_count >= self.circuit_threshold:
            self.circuit_open_time = time.time()
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _check_memory(self):
        """Periodic memory cleanup"""
        if time.time() - self.last_cleanup > self.cleanup_interval:
            gc.collect()
            self.last_cleanup = time.time()
            self.logger.debug("Memory cleanup performed")
    
    @contextmanager
    def db_session(self):
        """Context manager for database operations"""
        if not self._check_circuit_breaker():
            self.logger.warning("Circuit breaker open - skipping DB operation")
            yield None
            return
        
        try:
            if not self.connection or self.connection.closed:
                self.connection = psycopg2.connect(self.db_url)
                self.connection.autocommit = False
            
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            yield self.cursor
            self.connection.commit()
            self._on_success()
            
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"Database error: {e}")
            self._on_failure()
            yield None
            
        finally:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            self._check_memory()
    
    def log_zone(self, zone_data: Dict[str, Any]) -> Optional[str]:
        """Log a single supply/demand zone"""
        with self.db_session() as cursor:
            if cursor is None:
                return None
            
            try:
                # Sanitize all values
                zone_data = self._sanitize_value(zone_data)
                
                # Generate unique zone ID
                zone_id = f"{zone_data['symbol']}_{self.timeframe}_{zone_data['zone_type']}_{datetime.now().timestamp()}"
                
                # Calculate trading levels
                if zone_data['zone_type'] == 'DEMAND':
                    entry = zone_data['zone_top']
                    stop_loss = zone_data['zone_bottom'] * 0.99
                    target_1 = entry * 1.01
                    target_2 = entry * 1.02
                    target_3 = entry * 1.03
                else:  # SUPPLY
                    entry = zone_data['zone_bottom']
                    stop_loss = zone_data['zone_top'] * 1.01
                    target_1 = entry * 0.99
                    target_2 = entry * 0.98
                    target_3 = entry * 0.97
                
                risk = abs(entry - stop_loss)
                rr_1 = abs(target_1 - entry) / risk if risk > 0 else 0
                rr_2 = abs(target_2 - entry) / risk if risk > 0 else 0
                rr_3 = abs(target_3 - entry) / risk if risk > 0 else 0
                
                sql = """
                    INSERT INTO other_scanners.supply_demand_zones (
                        symbol, timeframe, zone_id, zone_type,
                        zone_top, zone_bottom, zone_strength,
                        touch_count, current_price, distance_to_zone,
                        distance_percentage, price_position,
                        zone_volume, average_volume, volume_ratio,
                        volume_confirmation, entry_price, stop_loss,
                        target_1, target_2, target_3,
                        risk_reward_1, risk_reward_2, risk_reward_3,
                        formation_type, formation_candles,
                        formation_start, formation_end,
                        quality_score, freshness_score, reliability_score,
                        scanner_version, algorithm_parameters, validation_status,
                        expires_at, post_mortem_status
                    ) VALUES (
                        %(symbol)s, %(timeframe)s, %(zone_id)s, %(zone_type)s,
                        %(zone_top)s, %(zone_bottom)s, %(zone_strength)s,
                        %(touch_count)s, %(current_price)s, %(distance_to_zone)s,
                        %(distance_percentage)s, %(price_position)s,
                        %(zone_volume)s, %(average_volume)s, %(volume_ratio)s,
                        %(volume_confirmation)s, %(entry_price)s, %(stop_loss)s,
                        %(target_1)s, %(target_2)s, %(target_3)s,
                        %(rr_1)s, %(rr_2)s, %(rr_3)s,
                        %(formation_type)s, %(formation_candles)s,
                        %(formation_start)s, %(formation_end)s,
                        %(quality_score)s, %(freshness_score)s, %(reliability_score)s,
                        %(scanner_version)s, %(algorithm_parameters)s, %(validation_status)s,
                        %(expires_at)s, %(post_mortem_status)s
                    )
                    ON CONFLICT (zone_id) 
                    DO UPDATE SET
                        touch_count = other_scanners.supply_demand_zones.touch_count + 1,
                        last_tested_at = CURRENT_TIMESTAMP,
                        current_price = EXCLUDED.current_price,
                        distance_to_zone = EXCLUDED.distance_to_zone,
                        distance_percentage = EXCLUDED.distance_percentage,
                        quality_score = GREATEST(
                            other_scanners.supply_demand_zones.quality_score,
                            EXCLUDED.quality_score
                        )
                """
                
                params = {
                    'symbol': zone_data['symbol'],
                    'timeframe': self.timeframe,
                    'zone_id': zone_id,
                    'zone_type': zone_data['zone_type'],
                    'zone_top': zone_data['zone_top'],
                    'zone_bottom': zone_data['zone_bottom'],
                    'zone_strength': zone_data.get('zone_strength', 50),
                    'touch_count': zone_data.get('touch_count', 1),
                    'current_price': zone_data['current_price'],
                    'distance_to_zone': zone_data.get('distance_to_zone'),
                    'distance_percentage': zone_data.get('distance_percentage'),
                    'price_position': zone_data.get('price_position'),
                    'zone_volume': zone_data.get('zone_volume'),
                    'average_volume': zone_data.get('average_volume'),
                    'volume_ratio': zone_data.get('volume_ratio'),
                    'volume_confirmation': zone_data.get('volume_confirmation', False),
                    'entry_price': entry,
                    'stop_loss': stop_loss,
                    'target_1': target_1,
                    'target_2': target_2,
                    'target_3': target_3,
                    'rr_1': rr_1,
                    'rr_2': rr_2,
                    'rr_3': rr_3,
                    'formation_type': zone_data.get('formation_type'),
                    'formation_candles': zone_data.get('formation_candles'),
                    'formation_start': zone_data.get('formation_start'),
                    'formation_end': zone_data.get('formation_end'),
                    'quality_score': zone_data.get('quality_score', 50),
                    'freshness_score': zone_data.get('freshness_score', 50),
                    'reliability_score': zone_data.get('reliability_score', 50),
                    'scanner_version': '1.0.0',
                    'algorithm_parameters': json.dumps(zone_data.get('algorithm_parameters', {})),
                    'validation_status': 'ACTIVE',
                    'expires_at': datetime.now() + timedelta(days=7),
                    'post_mortem_status': 'PENDING'
                }
                
                cursor.execute(sql, params)
                self.logger.info(f"Zone logged: {zone_id}")
                return zone_id
                
            except Exception as e:
                self.logger.error(f"Failed to log zone: {e}")
                print(f"    ❌ Database error: {e}")
                return None
    
    def log_zones_batch(self, zones: List[Dict[str, Any]]) -> int:
        """Log multiple zones in a batch"""
        logged_count = 0
        for i, zone in enumerate(zones):
            print(f"    🔍 Logging zone {i+1}/{len(zones)}: {zone.get('symbol', 'UNKNOWN')}")
            result = self.log_zone(zone)
            if result:
                logged_count += 1
                print(f"    ✅ Zone {i+1} logged successfully: {result}")
            else:
                print(f"    ❌ Zone {i+1} failed to log")
        return logged_count
    
    def close(self):
        """Clean up resources"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("Logger closed")
