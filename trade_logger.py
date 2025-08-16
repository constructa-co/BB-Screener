import os
from datetime import datetime
import json
import logging

# Try to import dotenv for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not available - using system environment variables")
    def load_dotenv():
        pass

# Try to import psycopg2, but handle gracefully if not available
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    print("⚠️  psycopg2 not available - database logging will be disabled")
    PSYCOPG2_AVAILABLE = False

class TradeLogger:
    """Handles all database operations for crypto scanner results"""
    
    def __init__(self):
        if not PSYCOPG2_AVAILABLE:
            logging.warning("⚠️  TradeLogger initialized in offline mode (psycopg2 not available)")
            self.db_url = None
            self.connection = None
            self.cursor = None
            return
            
        self.db_url = os.getenv('DATABASE_URL')
        self.connection = None
        self.cursor = None
        
        if not self.db_url:
            logging.warning("⚠️ No DATABASE_URL found, logging to files only")
        else:
            self.connect()
    
    def connect(self):
        """Establish database connection"""
        if not PSYCOPG2_AVAILABLE:
            return
            
        try:
            self.connection = psycopg2.connect(self.db_url)
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            logging.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logging.error(f"❌ Database connection failed: {e}")
            self.connection = None
    
    def create_tables(self):
        """Create all necessary tables if they don't exist"""
        if not self.connection:
            return False
            
        try:
            # Main scan results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    id SERIAL PRIMARY KEY,
                    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scan_type VARCHAR(50),
                    total_coins_analyzed INT,
                    premium_trades_found INT,
                    execution_time_seconds FLOAT,
                    server_ip VARCHAR(50),
                    scanner_version VARCHAR(20)
                );
            """)
            
            # Individual trade opportunities
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_opportunities (
                    id SERIAL PRIMARY KEY,
                    scan_id INT REFERENCES scan_results(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol VARCHAR(20),
                    exchange VARCHAR(50),
                    timeframe VARCHAR(10),
                    
                    -- Core metrics
                    bb_score FLOAT,
                    probability FLOAT,
                    risk_reward_ratio FLOAT,
                    
                    -- Price data
                    current_price DECIMAL(20,10),
                    entry_price DECIMAL(20,10),
                    stop_loss DECIMAL(20,10),
                    target_1 DECIMAL(20,10),
                    target_2 DECIMAL(20,10),
                    target_3 DECIMAL(20,10),
                    
                    -- Technical indicators
                    rsi FLOAT,
                    mfi FLOAT,
                    stochastic_k FLOAT,
                    volume_surge FLOAT,
                    macd_signal VARCHAR(20),
                    
                    -- Pattern data
                    pattern_type VARCHAR(100),
                    pattern_quality VARCHAR(20),
                    confluence_score FLOAT,
                    
                    -- Historical performance
                    historical_win_rate FLOAT,
                    category_win_rate FLOAT,
                    similar_setups_count INT,
                    
                    -- Market context
                    market_cap DECIMAL(20,2),
                    volume_24h DECIMAL(20,2),
                    price_change_24h FLOAT,
                    
                    -- Scanner specific data
                    scanner_specific_data JSONB,
                    
                    -- Status tracking
                    trade_taken BOOLEAN DEFAULT FALSE,
                    trade_result VARCHAR(20),
                    actual_exit_price DECIMAL(20,10),
                    actual_exit_time TIMESTAMP,
                    profit_loss_percent FLOAT
                );
            """)
            
            # Backtest results storage
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id SERIAL PRIMARY KEY,
                    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    strategy_name VARCHAR(100),
                    timeframe VARCHAR(10),
                    total_trades INT,
                    winning_trades INT,
                    win_rate FLOAT,
                    avg_profit FLOAT,
                    max_drawdown FLOAT,
                    sharpe_ratio FLOAT,
                    parameters JSONB,
                    detailed_results JSONB
                );
            """)
            
            # Daily performance tracking
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id SERIAL PRIMARY KEY,
                    date DATE DEFAULT CURRENT_DATE,
                    scanner_type VARCHAR(50),
                    total_opportunities INT,
                    trades_taken INT,
                    wins INT,
                    losses INT,
                    total_pnl_percent FLOAT,
                    best_trade VARCHAR(20),
                    worst_trade VARCHAR(20),
                    notes TEXT
                );
            """)
            
            # Create indexes for performance
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade_opportunities(timestamp);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_opportunities(symbol);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_probability ON trade_opportunities(probability DESC);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_results(scan_timestamp);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_type ON scan_results(scan_type);")
            
            self.connection.commit()
            logging.info("✅ Database tables created successfully")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error creating tables: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def log_scan_start(self, scan_type='main', version='1.0'):
        """Log the start of a scan and return scan_id"""
        if not self.connection:
            return None
            
        try:
            self.cursor.execute("""
                INSERT INTO scan_results (scan_type, scan_timestamp, scanner_version, server_ip)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (scan_type, datetime.now(), version, os.getenv('SERVER_IP', '165.232.160.52')))
            
            scan_id = self.cursor.fetchone()['id']
            self.connection.commit()
            logging.info(f"📊 Started scan {scan_id} - Type: {scan_type}")
            return scan_id
            
        except Exception as e:
            logging.error(f"Error logging scan start: {e}")
            return None
    
    def log_trade_opportunity(self, scan_id, trade_data):
        """Enhanced to handle all scanner-specific fields"""
        if not self.connection or not scan_id:
            return False
            
        try:
            # Define core fields that go in regular columns for fast querying
            core_fields = {
                'symbol', 'exchange', 'probability', 'entry_price', 
                'stop_loss', 'target_1', 'target_2', 'target_3',
                'bb_score', 'risk_reward_ratio', 'current_price',
                'rsi', 'mfi', 'stochastic_k', 'volume_surge', 'macd_signal',
                'pattern_type', 'pattern_quality', 'confluence_score',
                'historical_win_rate', 'category_win_rate', 'similar_setups_count',
                'market_cap', 'volume_24h', 'price_change_24h'
            }
            
            # Extract core fields for regular columns, put everything else in JSONB
            column_values = {}
            json_data = {}
            
            for key, value in trade_data.items():
                if key in core_fields and value is not None:
                    column_values[key] = value
                else:
                    # Put everything else in JSONB (safe serialization)
                    try:
                        if hasattr(value, 'tolist'):  # numpy arrays
                            json_data[key] = value.tolist()
                        elif hasattr(value, '__dict__'):  # objects
                            json_data[key] = str(value)
                        elif pd.isna(value):  # pandas NaN values
                            json_data[key] = None
                        elif hasattr(value, 'isoformat'):  # datetime objects
                            json_data[key] = value.isoformat()
                        elif hasattr(value, 'timestamp'):  # pandas timestamps
                            json_data[key] = value.isoformat()
                        elif str(type(value)).startswith('<class \'pandas'):  # other pandas types
                            json_data[key] = str(value)
                        else:
                            json_data[key] = value
                    except:
                        json_data[key] = str(value)
            
            # Build dynamic INSERT query with core columns
            columns = ['scan_id'] + list(column_values.keys())
            values = [scan_id] + list(column_values.values())
            
            # Add JSON data with safe serialization
            columns.append('scanner_specific_data')
            values.append(json.dumps(json_data))
            
            # Execute INSERT
            placeholders = ','.join(['%s'] * len(values))
            query = f"INSERT INTO trade_opportunities ({','.join(columns)}) VALUES ({placeholders})"
            
            self.cursor.execute(query, values)
            self.connection.commit()
            
            # Debug: Log field count for verification
            field_count = len(trade_data)
            jsonb_count = len(json_data)
            logging.info(f"✅ Logged trade {trade_data.get('symbol')} with {field_count} total fields ({len(column_values)} in columns, {jsonb_count} in JSONB)")
            return True
            
        except Exception as e:
            logging.error(f"Error logging trade {trade_data.get('symbol')}: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def complete_scan(self, scan_id, total_coins, premium_trades, execution_time):
        """Update scan results with completion data"""
        if not self.connection or not scan_id:
            return
            
        try:
            self.cursor.execute("""
                UPDATE scan_results 
                SET total_coins_analyzed = %s,
                    premium_trades_found = %s,
                    execution_time_seconds = %s
                WHERE id = %s
            """, (total_coins, premium_trades, execution_time, scan_id))
            
            self.connection.commit()
            logging.info(f"✅ Scan {scan_id} completed: {premium_trades} opportunities from {total_coins} coins in {execution_time:.1f}s")
            
        except Exception as e:
            logging.error(f"Error completing scan: {e}")
    
    def log_backtest_results(self, strategy_name, timeframe, results):
        """Log backtest results"""
        if not self.connection:
            return
            
        try:
            self.cursor.execute("""
                INSERT INTO backtest_results (
                    strategy_name, timeframe, total_trades, winning_trades,
                    win_rate, avg_profit, max_drawdown, sharpe_ratio,
                    parameters, detailed_results
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                strategy_name,
                timeframe,
                results.get('total_trades'),
                results.get('winning_trades'),
                results.get('win_rate'),
                results.get('avg_profit'),
                results.get('max_drawdown'),
                results.get('sharpe_ratio'),
                json.dumps(results.get('parameters', {})),
                json.dumps(results.get('detailed_results', {}))
            ))
            
            self.connection.commit()
            logging.info(f"📈 Logged backtest: {strategy_name} - {timeframe} - Win Rate: {results.get('win_rate')}%")
            
        except Exception as e:
            logging.error(f"Error logging backtest: {e}")
    
    def get_recent_trades(self, limit=50, min_probability=70):
        """Retrieve recent high-probability trades"""
        if not self.connection:
            return []
            
        try:
            self.cursor.execute("""
                SELECT t.*, s.scan_type, s.scan_timestamp
                FROM trade_opportunities t
                JOIN scan_results s ON t.scan_id = s.id
                WHERE t.probability >= %s
                ORDER BY t.timestamp DESC
                LIMIT %s
            """, (min_probability, limit))
            
            return self.cursor.fetchall()
            
        except Exception as e:
            logging.error(f"Error retrieving trades: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.cursor.close()
            self.connection.close()
            logging.info("Database connection closed")


# Standalone function to create tables
def initialize_database():
    """Initialize database with all tables"""
    logger = TradeLogger()
    if logger.connection:
        logger.create_tables()
        logger.close()
        print("✅ Database initialized successfully")
    else:
        print("❌ Could not connect to database")


if __name__ == "__main__":
    # If run directly, initialize the database
    initialize_database()
