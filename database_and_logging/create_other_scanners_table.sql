-- BB Screener: Other Scanners Database Schema
-- Creates the other_scanners_trades table for all non-BB scanners
-- Supports post-mortem analysis, ML training, and complete trade lifecycle tracking
-- USES SEPARATE SCHEMA FOR COMPLETE ISOLATION FROM MAIN SCANNER

-- Create separate schema for other scanners
CREATE SCHEMA IF NOT EXISTS other_scanners;
SET search_path TO other_scanners;

-- Create the main trades table with all required fields
CREATE TABLE IF NOT EXISTS other_scanners_trades (
    -- Primary identification
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scanner_name VARCHAR(100) NOT NULL,
    scanner_version VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL CHECK (timeframe IN ('1m', '5m', '15m', '1h', '4h', '1d', '1w')),
    
    -- Trade lifecycle tracking
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ACTIVE', 'CLOSED', 'CANCELLED', 'ERROR')),
    status_pending_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status_active_at TIMESTAMP,
    status_closed_at TIMESTAMP,
    status_cancelled_at TIMESTAMP,
    status_error_at TIMESTAMP,
    
    -- Trading data
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) CHECK (side IN ('BUY', 'SELL', 'LONG', 'SHORT')),
    entry_price DECIMAL(18, 8),
    exit_price DECIMAL(18, 8),
    quantity DECIMAL(18, 8) CHECK (quantity > 0),
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    
    -- Performance metrics
    pnl DECIMAL(18, 8),
    pnl_percentage DECIMAL(6, 2),
    max_drawdown DECIMAL(18, 8),
    duration_minutes INTEGER,
    
    -- Post-mortem analysis fields
    market_conditions JSONB DEFAULT '{}',
    technical_indicators JSONB DEFAULT '{}',
    scanner_signals JSONB DEFAULT '{}',
    
    -- ML training fields
    feature_vector JSONB DEFAULT '[]',
    label VARCHAR(50),
    model_version VARCHAR(20),
    prediction_confidence DECIMAL(4, 3) CHECK (prediction_confidence BETWEEN 0 AND 1),
    
    -- Metadata
    execution_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance-optimized indexes
CREATE INDEX IF NOT EXISTS idx_scanner_name ON other_scanners_trades(scanner_name);
CREATE INDEX IF NOT EXISTS idx_status ON other_scanners_trades(status);
CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON other_scanners_trades(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_created_at ON other_scanners_trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_status_active ON other_scanners_trades(status) WHERE status = 'ACTIVE';

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_market_conditions ON other_scanners_trades USING GIN (market_conditions);
CREATE INDEX IF NOT EXISTS idx_technical_indicators ON other_scanners_trades USING GIN (technical_indicators);
CREATE INDEX IF NOT EXISTS idx_scanner_signals ON other_scanners_trades USING GIN (scanner_signals);
CREATE INDEX IF NOT EXISTS idx_feature_vector ON other_scanners_trades USING GIN (feature_vector);

-- Trigger for automatic updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trades_updated_at BEFORE UPDATE
    ON other_scanners_trades FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for automatic PnL calculation
CREATE OR REPLACE FUNCTION calculate_pnl()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.exit_price IS NOT NULL AND NEW.entry_price IS NOT NULL AND NEW.quantity IS NOT NULL THEN
        IF NEW.side IN ('BUY', 'LONG') THEN
            NEW.pnl = (NEW.exit_price - NEW.entry_price) * NEW.quantity;
        ELSIF NEW.side IN ('SELL', 'SHORT') THEN
            NEW.pnl = (NEW.entry_price - NEW.exit_price) * NEW.quantity;
        END IF;
        
        IF NEW.entry_price > 0 THEN
            NEW.pnl_percentage = (NEW.pnl / (NEW.entry_price * NEW.quantity)) * 100;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_trade_pnl BEFORE INSERT OR UPDATE
    ON other_scanners_trades FOR EACH ROW
    EXECUTE FUNCTION calculate_pnl();

-- Create scanner_backtest_results table for aggregated performance metrics
CREATE TABLE IF NOT EXISTS scanner_backtest_results (
    id SERIAL PRIMARY KEY,
    scanner_name VARCHAR(50) NOT NULL,
    scanner_version VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    backtest_period VARCHAR(20) NOT NULL,  -- '30d', '90d', '1y'
    
    -- Performance statistics
    total_trades INTEGER DEFAULT 0,
    win_rate_pct DECIMAL(5, 2),
    avg_win_pct DECIMAL(8, 4),
    avg_loss_pct DECIMAL(8, 4),
    profit_factor DECIMAL(8, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown_pct DECIMAL(8, 4),
    
    -- Detailed results
    backtest_data JSONB DEFAULT '{}',  -- Full backtest details
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint to prevent duplicate backtest results
    UNIQUE(scanner_name, scanner_version, timeframe, backtest_period)
);

-- Indexes for backtest results
CREATE INDEX IF NOT EXISTS idx_backtest_scanner ON scanner_backtest_results(scanner_name);
CREATE INDEX IF NOT EXISTS idx_backtest_timeframe ON scanner_backtest_results(timeframe);
CREATE INDEX IF NOT EXISTS idx_backtest_timestamp ON scanner_backtest_results(timestamp DESC);

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON SCHEMA other_scanners TO bbscanner;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA other_scanners TO bbscanner;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA other_scanners TO bbscanner;

-- Insert initial scan marker to verify table creation
INSERT INTO other_scanners_trades (scanner_name, scanner_version, timeframe, symbol, status, execution_metadata)
VALUES ('SYSTEM', '1.0.0', '1h', 'SCHEMA_CREATED', 'CLOSED', 
        '{"action": "schema_creation", "timestamp": "' || CURRENT_TIMESTAMP || '", "version": "1.0.0", "schema": "other_scanners"}')
ON CONFLICT DO NOTHING;

-- Display table information
SELECT 
    'other_scanners_trades' as table_name,
    COUNT(*) as total_records,
    MAX(created_at) as latest_record
FROM other_scanners_trades
UNION ALL
SELECT 
    'scanner_backtest_results' as table_name,
    COUNT(*) as total_records,
    MAX(timestamp) as latest_record
FROM scanner_backtest_results;

-- Reset search_path to default
SET search_path TO public;
