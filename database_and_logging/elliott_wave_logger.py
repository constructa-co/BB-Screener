#!/usr/bin/env python3
"""
Elliott Wave Logger - Completely Separate from Universal Logger
Handles the unique hierarchical data structure of Elliott Wave patterns
DOES NOT import or reference universal_scanner_logger.py
ONLY writes to elliott_wave_signals table
"""

import os
import json
import uuid
import math
from datetime import datetime
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, Json, execute_values
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElliottWaveLogger:
    """
    Dedicated logger for Elliott Wave patterns - COMPLETELY SEPARATE from universal logger
    
    This logger:
    - Does NOT import universal_scanner_logger.py
    - Does NOT write to other_scanners_trades table
    - ONLY writes to elliott_wave_signals table
    - Handles hierarchical wave data in JSONB format
    - Implements upsert logic to prevent duplicates
    """
    
    def __init__(self, dsn: Optional[str] = None):
        """
        Initialize Elliott Wave logger with separate database connection
        
        Args:
            dsn: Database connection string (defaults to ELLIOTT_WAVE_DATABASE_URL env var)
        """
        self.db_url = dsn or os.environ.get('ELLIOTT_WAVE_DATABASE_URL')
        if not self.db_url:
            raise ValueError("ELLIOTT_WAVE_DATABASE_URL environment variable not set")
        
        self.connection = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """Establish database connection to other_scanners schema"""
        try:
            self.connection = psycopg2.connect(
                self.db_url,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
            
            # Set search path to other_scanners schema
            self.cursor.execute("SET search_path TO other_scanners")
            
            logger.info("✅ Elliott Wave Logger connected to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def _sanitize_value(self, value: Any) -> Any:
        """
        Clean values for database insertion
        
        Handles:
        - None values
        - NaN and Inf floats
        - Numpy types
        - Pandas Timestamp objects
        - JSON serialization
        """
        if value is None:
            return None
        
        # Handle pandas Timestamp objects
        try:
            import pandas as pd
            if isinstance(value, pd.Timestamp):
                return value.strftime('%Y-%m-%d %H:%M:%S')
        except ImportError:
            pass
        
        # Handle numpy types if present
        try:
            import numpy as np
            if isinstance(value, np.generic):
                value = value.item()
        except ImportError:
            pass
        
        # Handle NaN and Inf
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        
        return value
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary values for JSON serialization
        """
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = self._sanitize_list(value)
            else:
                sanitized[key] = self._sanitize_value(value)
        return sanitized
    
    def _sanitize_list(self, data: list) -> list:
        """
        Recursively sanitize list values for JSON serialization
        """
        if not isinstance(data, list):
            return data
        
        sanitized = []
        for item in data:
            if isinstance(item, dict):
                sanitized.append(self._sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(self._sanitize_list(item))
            else:
                sanitized.append(self._sanitize_value(item))
        return sanitized
    
    def log_elliott_signal(self, signal_data: Dict[str, Any], scan_id: str) -> Optional[str]:
        """
        Log Elliott Wave signal to elliott_wave_signals table
        
        Args:
            signal_data: Complete Elliott Wave analysis from scanner
            scan_id: Unique identifier for this scan run
            
        Returns:
            UUID of inserted record or None if failed
        """
        try:
            signal_id = str(uuid.uuid4())
            
            # Map Elliott Wave data to database columns
            record = {
                'id': signal_id,
                'scanner_name': signal_data.get('scanner_name', 'elliott_wave'),
                'scanner_version': signal_data.get('scanner_version', 'R0'),
                'scan_id': scan_id,
                'symbol': signal_data['symbol'],
                'exchange': signal_data.get('exchange', 'binance'),
                'timeframe': signal_data['timeframe'],
                
                # Trading signals
                'direction': signal_data.get('direction'),
                'entry_price': self._sanitize_value(signal_data.get('entry_price')),
                'stop_loss': self._sanitize_value(signal_data.get('stop_loss')),
                'tp1': self._sanitize_value(signal_data.get('targets', [None, None, None])[0] if isinstance(signal_data.get('targets'), list) else signal_data.get('tp1')),
                'tp2': self._sanitize_value(signal_data.get('targets', [None, None, None])[1] if isinstance(signal_data.get('targets'), list) else signal_data.get('tp2')),
                'tp3': self._sanitize_value(signal_data.get('targets', [None, None, None])[2] if isinstance(signal_data.get('targets'), list) else signal_data.get('tp3')),
                'risk_reward': self._sanitize_value(signal_data.get('risk_reward')),
                
                # Elliott Wave specific fields
                'pattern_type': signal_data.get('pattern_type'),
                'current_wave': str(signal_data.get('current_wave', '')),
                'wave_degree': signal_data.get('wave_degree', 'minor'),
                'pattern_quality': self._sanitize_value(signal_data.get('pattern_quality') or signal_data.get('quality_score')),
                'confidence_score': self._sanitize_value(signal_data.get('confidence_score')),
                'invalidation_level': self._sanitize_value(signal_data.get('invalidation_level')),
                
                # Complete wave analysis in JSONB (hierarchical structure)
                'wave_analysis': Json(self._sanitize_dict({
                    'waves': signal_data.get('waves', []),
                    'wave_relationships': signal_data.get('wave_relationships', {}),
                    'internal_structure': signal_data.get('internal_structure', {}),
                    'alternation': signal_data.get('alternation', {}),
                    'wave_start': signal_data.get('wave_start'),
                    'wave_start_date': signal_data.get('wave_start_date'),
                    'current_price': signal_data.get('current_price'),
                    'duration': signal_data.get('duration')
                })),
                
                # Fibonacci levels and relationships
                'fibonacci_levels': Json(self._sanitize_dict(signal_data.get('fibonacci_levels', {}))),
                
                # Pattern quality metrics and validation
                'pattern_metrics': Json(self._sanitize_dict({
                    'quality_score': signal_data.get('pattern_quality') or signal_data.get('quality_score'),
                    'confidence': signal_data.get('confidence_score'),
                    'strength_indicators': signal_data.get('strength_indicators', {}),
                    'validation_checks': signal_data.get('validation_checks', {}),
                    'wave_validation': signal_data.get('wave_validation', {})
                })),
                
                'analysis_ts': datetime.utcnow()
            }
            
            # SQL with upsert logic to prevent duplicates
            sql = """
                INSERT INTO elliott_wave_signals 
                (id, scanner_name, scanner_version, scan_id, symbol, exchange, timeframe,
                 direction, entry_price, stop_loss, tp1, tp2, tp3, risk_reward,
                 pattern_type, current_wave, wave_degree, pattern_quality, confidence_score,
                 invalidation_level, wave_analysis, fibonacci_levels, pattern_metrics, analysis_ts)
                VALUES 
                (%(id)s, %(scanner_name)s, %(scanner_version)s, %(scan_id)s, %(symbol)s, 
                 %(exchange)s, %(timeframe)s, %(direction)s, %(entry_price)s, %(stop_loss)s,
                 %(tp1)s, %(tp2)s, %(tp3)s, %(risk_reward)s, %(pattern_type)s, %(current_wave)s,
                 %(wave_degree)s, %(pattern_quality)s, %(confidence_score)s, %(invalidation_level)s,
                 %(wave_analysis)s, %(fibonacci_levels)s, %(pattern_metrics)s, %(analysis_ts)s)
                ON CONFLICT (scan_id, symbol, timeframe, current_wave)
                DO UPDATE SET
                    pattern_quality = EXCLUDED.pattern_quality,
                    confidence_score = EXCLUDED.confidence_score,
                    wave_analysis = EXCLUDED.wave_analysis,
                    fibonacci_levels = EXCLUDED.fibonacci_levels,
                    pattern_metrics = EXCLUDED.pattern_metrics,
                    analysis_ts = EXCLUDED.analysis_ts
                RETURNING id;
            """
            
            self.cursor.execute(sql, record)
            result = self.cursor.fetchone()
            
            logger.info(f"✅ Logged Elliott Wave signal for {signal_data['symbol']} "
                       f"({signal_data['timeframe']}) - Wave {signal_data.get('current_wave', 'N/A')}")
            
            return result['id'] if result else signal_id
            
        except Exception as e:
            logger.error(f"❌ Failed to log Elliott Wave signal: {e}")
            logger.error(f"Signal data: {json.dumps(signal_data, default=str)}")
            return None
    
    def log_elliott_signals_batch(self, signals: list, scan_id: str) -> list:
        """
        Log multiple Elliott Wave signals in a batch
        
        Args:
            signals: List of signal data dictionaries
            scan_id: Unique identifier for this scan run
            
        Returns:
            List of successfully logged signal IDs
        """
        successful_ids = []
        
        for signal_data in signals:
            try:
                signal_id = self.log_elliott_signal(signal_data, scan_id)
                if signal_id:
                    successful_ids.append(signal_id)
            except Exception as e:
                logger.error(f"❌ Failed to log signal in batch: {e}")
                continue
        
        logger.info(f"✅ Batch logged {len(successful_ids)}/{len(signals)} Elliott Wave signals")
        return successful_ids
    
    def close(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Elliott Wave Logger connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Example usage and testing
if __name__ == "__main__":
    # Test the Elliott Wave logger
    try:
        logger = ElliottWaveLogger()
        
        # Test signal data (matches Elliott Wave scanner output)
        test_signal = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'direction': 'LONG',
            'pattern_type': 'BULLISH_IMPULSE',
            'current_wave': 'WAVE_3',
            'wave_degree': 'intermediate',
            'pattern_quality': 85.5,
            'confidence_score': 0.78,
            'entry_price': 45000.0,
            'stop_loss': 44000.0,
            'targets': [46000.0, 47000.0, 48000.0],
            'risk_reward': 2.5,
            'invalidation_level': 43500.0,
            
            # Wave analysis (hierarchical structure)
            'waves': [
                {
                    'wave': 1,
                    'start': {'price': 44000, 'date': '2025-08-18 10:00:00'},
                    'end': {'price': 45000, 'date': '2025-08-18 12:00:00'},
                    'size': 2.27
                },
                {
                    'wave': 2,
                    'start': {'price': 45000, 'date': '2025-08-18 12:00:00'},
                    'end': {'price': 44500, 'date': '2025-08-18 14:00:00'},
                    'retrace': 38.2
                },
                {
                    'wave': 3,
                    'start': {'price': 44500, 'date': '2025-08-18 14:00:00'},
                    'end': {'price': 45500, 'date': '2025-08-18 16:00:00'},
                    'size': 2.25,
                    'vs_wave1': 1.0
                }
            ],
            
            # Fibonacci levels
            'fibonacci_levels': {
                'extensions': [1.618, 2.618, 4.236],
                'retracements': [0.382, 0.5, 0.618],
                'current_confluence': 45500
            },
            
            # Pattern metrics
            'strength_indicators': {
                'wave_strength': 'strong',
                'pattern_clarity': 'high',
                'fibonacci_alignment': 'excellent'
            }
        }
        
        # Test logging
        scan_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result = logger.log_elliott_signal(test_signal, scan_id)
        
        if result:
            print(f"✅ Test successful! Logged signal with ID: {result}")
        else:
            print("❌ Test failed!")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        if 'logger' in locals():
            logger.close()
