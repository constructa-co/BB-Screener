"""
Universal Scanner Logger for BB Screener
Provides database logging for all non-BB scanners with minimal integration effort.
Supports post-mortem analysis, ML training, and complete trade lifecycle tracking.
USES SEPARATE SCHEMA FOR COMPLETE ISOLATION FROM MAIN SCANNER
"""

import os
import re
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json
import time

class UniversalScannerLogger:
    """Universal logger for all BB Screener scanners with connection pooling and UUID tracking."""
    
    def __init__(self, scanner_name: str, scanner_version: str = None, config: Dict[str, Any] = None):
        """
        Initialize the universal scanner logger.
        
        Args:
            scanner_name: Name of the scanner (e.g., 'ict_scanner', 'smc_scanner')
            scanner_version: Version of the scanner (e.g., 'r8', 'v1.0.0')
            config: Optional configuration dictionary
        """
        self.scanner_name = scanner_name
        self.scanner_version = scanner_version or self._extract_version()
        self.config = config or {}
        
        # Setup logging FIRST (before connection pool)
        self.logger = logging.getLogger(f"scanner.{scanner_name}")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # UUID for this scanner instance
        self.scanner_uuid = str(uuid.uuid4())
        
        # Initialize connection pool AFTER logger is set up
        self._init_connection_pool()
        
        self.logger.info(f"✅ {scanner_name} logger initialized (version: {self.scanner_version})")
        
    def _extract_version(self) -> str:
        """Extract version from filename pattern: scanner_v1.2.3.py or scanner_r8.py"""
        import sys
        script_name = os.path.basename(sys.argv[0]) if sys.argv else "unknown"
        
        # Try version pattern first (v1.2.3)
        version_match = re.search(r'v(\d+\.\d+\.\d+)', script_name)
        if version_match:
            return version_match.group(1)
        
        # Try revision pattern (r8)
        revision_match = re.search(r'_r(\d+)', script_name)
        if revision_match:
            return f"r{revision_match.group(1)}"
        
        return "1.0.0"  # Default version
    
    def _init_connection_pool(self):
        """Initialize PostgreSQL connection pool with schema isolation."""
        try:
            # Get database configuration from environment - USE SEPARATE ENV VAR FOR ISOLATION
            DATABASE_URL = os.getenv('OTHER_SCANNERS_DATABASE_URL')
            
            if not DATABASE_URL:
                # Fallback to main DATABASE_URL but with schema specification
                main_db_url = os.getenv('DATABASE_URL')
                if main_db_url:
                    # Add schema parameter to main database URL
                    if '?' in main_db_url:
                        DATABASE_URL = main_db_url + '&options=-csearch_path=other_scanners'
                    else:
                        DATABASE_URL = main_db_url + '?options=-csearch_path=other_scanners'
                else:
                    # Fallback configuration for local development
                    db_config = {
                        'host': os.getenv('DB_HOST', 'localhost'),
                        'port': int(os.getenv('DB_PORT', 5432)),
                        'database': os.getenv('DB_NAME', 'bb_scanners'),
                        'user': os.getenv('DB_USER', 'bbscanner'),
                        'password': os.getenv('DB_PASSWORD', ''),
                    }
                    
                    # Build connection string with schema
                    DATABASE_URL = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?options=-csearch_path=other_scanners"
            
            # Create connection pool
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=int(os.getenv('DB_POOL_SIZE', 20)),
                dsn=DATABASE_URL
            )
            
            self.logger.info(f"✅ Database connection pool initialized (max connections: {os.getenv('DB_POOL_SIZE', 20)})")
            self.logger.info(f"✅ Using schema: other_scanners (isolated from main scanner)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with schema isolation."""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            # Set schema for this connection
            with conn.cursor() as cursor:
                cursor.execute("SET search_path TO other_scanners")
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"❌ Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def log_trade(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """
        Log a new trade with UUID generation.
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            trade_id: UUID of the logged trade, or None if logging failed
        """
        trade_id = str(uuid.uuid4())
        
        try:
            # Prepare JSONB fields with defaults
            market_conditions = trade_data.pop('market_conditions', {})
            technical_indicators = trade_data.pop('technical_indicators', {})
            scanner_signals = trade_data.pop('scanner_signals', {})
            feature_vector = trade_data.pop('feature_vector', [])
            execution_metadata = trade_data.pop('execution_metadata', {})
            
            # Add scanner instance UUID to metadata
            execution_metadata['scanner_instance_uuid'] = self.scanner_uuid
            execution_metadata['logged_at'] = datetime.now(timezone.utc).isoformat()
            execution_metadata['scanner_name'] = self.scanner_name
            execution_metadata['scanner_version'] = self.scanner_version
            execution_metadata['schema'] = 'other_scanners'  # Track schema usage
            
            # Set default values for required fields
            trade_data.setdefault('timeframe', '4H')
            trade_data.setdefault('side', 'BUY')
            trade_data.setdefault('quantity', 1.0)
            
            query = """
                INSERT INTO other_scanners_trades (
                    id, scanner_name, scanner_version, timeframe, symbol, side,
                    entry_price, quantity, stop_loss, take_profit,
                    market_conditions, technical_indicators, scanner_signals,
                    feature_vector, execution_metadata, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, 'PENDING'
                )
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (
                        trade_id,
                        self.scanner_name,
                        self.scanner_version,
                        trade_data.get('timeframe'),
                        trade_data.get('symbol'),
                        trade_data.get('side'),
                        trade_data.get('entry_price'),
                        trade_data.get('quantity'),
                        trade_data.get('stop_loss'),
                        trade_data.get('take_profit'),
                        Json(market_conditions),
                        Json(technical_indicators),
                        Json(scanner_signals),
                        Json(feature_vector),
                        Json(execution_metadata)
                    ))
            
            self.logger.info(f"✅ Trade logged: {trade_id} for {trade_data.get('symbol')} (schema: other_scanners)")
            return trade_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log trade {trade_data.get('symbol', 'UNKNOWN')}: {e}")
            return None
    
    def log_trades_batch(self, trades_list: List[Dict[str, Any]]) -> int:
        """
        Log multiple trades at once.
        
        Args:
            trades_list: List of trade dictionaries
            
        Returns:
            success_count: Number of successfully logged trades
        """
        if not trades_list:
            self.logger.info("ℹ️ No trades to log")
            return 0
        
        success_count = 0
        for trade in trades_list:
            if self.log_trade(trade):
                success_count += 1
        
        self.logger.info(f"✅ Logged {success_count}/{len(trades_list)} trades to database (schema: other_scanners)")
        return success_count
    
    def update_trade_status(self, trade_id: str, new_status: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Update trade lifecycle status with timestamp tracking.
        
        Args:
            trade_id: UUID of the trade to update
            new_status: New status ('PENDING', 'ACTIVE', 'CLOSED', 'CANCELLED', 'ERROR')
            metadata: Optional metadata to add to execution_metadata
            
        Returns:
            success: True if update was successful
        """
        try:
            status_column = f"status_{new_status.lower()}_at"
            
            query = f"""
                UPDATE other_scanners_trades 
                SET status = %s, {status_column} = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (new_status, trade_id))
                    
                    if metadata:
                        cursor.execute("""
                            UPDATE other_scanners_trades
                            SET execution_metadata = execution_metadata || %s
                            WHERE id = %s
                        """, (Json(metadata), trade_id))
            
            self.logger.info(f"✅ Trade {trade_id} status updated to {new_status} (schema: other_scanners)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update trade {trade_id} status: {e}")
            return False
    
    def close_trade(self, trade_id: str, exit_price: float, metadata: Dict[str, Any] = None) -> bool:
        """
        Close a trade with exit price and calculate PnL.
        
        Args:
            trade_id: UUID of the trade to close
            exit_price: Exit price for the trade
            metadata: Optional metadata to add
            
        Returns:
            success: True if trade was closed successfully
        """
        try:
            query = """
                UPDATE other_scanners_trades
                SET status = 'CLOSED',
                    exit_price = %s,
                    status_closed_at = CURRENT_TIMESTAMP,
                    duration_minutes = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 60
                WHERE id = %s
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (exit_price, trade_id))
                    
                    if metadata:
                        cursor.execute("""
                            UPDATE other_scanners_trades
                            SET execution_metadata = execution_metadata || %s
                            WHERE id = %s
                        """, (Json(metadata), trade_id))
            
            self.logger.info(f"✅ Trade {trade_id} closed at {exit_price} (schema: other_scanners)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to close trade {trade_id}: {e}")
            return False
    
    def log_analysis(self, trade_id: str, analysis_type: str, analysis_data: Dict[str, Any]) -> bool:
        """
        Log post-mortem analysis data.
        
        Args:
            trade_id: UUID of the trade
            analysis_type: Type of analysis ('market', 'technical', 'signals', 'features')
            analysis_data: Analysis data to log
            
        Returns:
            success: True if analysis was logged successfully
        """
        try:
            column_map = {
                'market': 'market_conditions',
                'technical': 'technical_indicators',
                'signals': 'scanner_signals',
                'features': 'feature_vector'
            }
            
            column = column_map.get(analysis_type)
            if not column:
                self.logger.warning(f"⚠️ Unknown analysis type: {analysis_type}")
                return False
            
            query = f"""
                UPDATE other_scanners_trades
                SET {column} = {column} || %s
                WHERE id = %s
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (Json(analysis_data), trade_id))
            
            self.logger.info(f"✅ Analysis logged for trade {trade_id}: {analysis_type} (schema: other_scanners)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log analysis for trade {trade_id}: {e}")
            return False
    
    def get_active_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve active trades for monitoring.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of active trades
        """
        try:
            query = """
                SELECT * FROM other_scanners_trades
                WHERE status = 'ACTIVE'
                AND scanner_name = %s
            """
            params = [self.scanner_name]
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            query += " ORDER BY created_at DESC"
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to get active trades: {e}")
            return []
    
    def log_error(self, trade_id: Optional[str], error: Exception, context: Dict[str, Any] = None) -> bool:
        """
        Log scanner errors with context.
        
        Args:
            trade_id: Optional trade ID if error is trade-specific
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            success: True if error was logged successfully
        """
        try:
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'context': context or {},
                'scanner_name': self.scanner_name,
                'scanner_version': self.scanner_version,
                'schema': 'other_scanners'
            }
            
            if trade_id:
                return self.update_trade_status(trade_id, 'ERROR', error_data)
            else:
                self.logger.error(f"❌ Scanner error: {error}", exc_info=True)
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log error: {e}")
            return False
    
    def cleanup_old_trades(self, days: int = 30) -> int:
        """
        Archive or delete old trades.
        
        Args:
            days: Number of days to keep trades
            
        Returns:
            Number of deleted trades
        """
        try:
            query = """
                DELETE FROM other_scanners_trades
                WHERE status = 'CLOSED'
                AND created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND scanner_name = %s
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (days, self.scanner_name))
                    deleted = cursor.rowcount
            
            self.logger.info(f"✅ Cleaned up {deleted} old trades (schema: other_scanners)")
            return deleted
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old trades: {e}")
            return 0
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection pool."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            self.logger.info(f"✅ {self.scanner_name} logger closed")

    @classmethod
    def from_env(cls, scanner_name: str = None):
        """
        Create logger instance from environment variables.
        
        Args:
            scanner_name: Optional scanner name (extracted from script name if not provided)
            
        Returns:
            UniversalScannerLogger instance
        """
        if not scanner_name:
            import sys
            scanner_name = os.path.basename(sys.argv[0]).replace('.py', '')
        
        config = {
            'database_pool': True,
            'trade_logging': True,
            'error_tracking': True,
            'schema_isolation': True  # Track schema isolation
        }
        
        return cls(scanner_name=scanner_name, config=config)
