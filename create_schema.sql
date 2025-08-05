-- Crypto Scanner Database Schema
-- Supports BB, ICT, Wyckoff, Elliott Waves, and all scanner types

-- Main scan results table
CREATE TABLE IF NOT EXISTS scan_results (
    id SERIAL PRIMARY KEY,
    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scan_type VARCHAR(50),  -- 'bb_main', 'ict_15min', 'wyckoff_4h', etc.
    total_coins_analyzed INT,
    premium_trades_found INT,
    execution_time_seconds FLOAT,
    server_ip VARCHAR(50),
    scanner_version VARCHAR(20)
);

-- Individual trade opportunities
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
    
    -- Scanner specific data (JSON for flexibility)
    scanner_specific_data JSONB,
    
    -- Status tracking
    trade_taken BOOLEAN DEFAULT FALSE,
    trade_result VARCHAR(20),  -- 'pending', 'win', 'loss', 'breakeven'
    actual_exit_price DECIMAL(20,10),
    actual_exit_time TIMESTAMP,
    profit_loss_percent FLOAT
);

-- Backtest results storage
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
    parameters JSONB,  -- Store strategy parameters
    detailed_results JSONB  -- Store detailed trade-by-trade results
);

-- Daily performance tracking
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

-- Scanner configuration history
CREATE TABLE IF NOT EXISTS scanner_configs (
    id SERIAL PRIMARY KEY,
    scanner_type VARCHAR(50),
    version VARCHAR(20),
    config_data JSONB,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade_opportunities(timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_opportunities(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_probability ON trade_opportunities(probability DESC);
CREATE INDEX IF NOT EXISTS idx_trade_scanner ON trade_opportunities(scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_results(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_scan_type ON scan_results(scan_type);
CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name, timeframe);

-- Create views for easy querying
CREATE VIEW IF NOT EXISTS high_probability_trades AS
SELECT 
    t.*,
    s.scan_type,
    s.scan_timestamp
FROM trade_opportunities t
JOIN scan_results s ON t.scan_id = s.id
WHERE t.probability >= 70
ORDER BY t.timestamp DESC;

CREATE VIEW IF NOT EXISTS scanner_performance AS
SELECT 
    scan_type,
    COUNT(*) as total_scans,
    SUM(total_coins_analyzed) as total_coins,
    SUM(premium_trades_found) as total_opportunities,
    AVG(execution_time_seconds) as avg_execution_time,
    MAX(scan_timestamp) as last_scan
FROM scan_results
GROUP BY scan_type;

-- Stored procedure to mark trade result
CREATE OR REPLACE FUNCTION update_trade_result(
    trade_id INT,
    result VARCHAR(20),
    exit_price DECIMAL(20,10)
) RETURNS VOID AS 66216
BEGIN
    UPDATE trade_opportunities
    SET 
        trade_result = result,
        actual_exit_price = exit_price,
        actual_exit_time = CURRENT_TIMESTAMP,
        profit_loss_percent = ((exit_price - entry_price) / entry_price) * 100
    WHERE id = trade_id;
END;
66216 LANGUAGE plpgsql;

-- Function to get win rate by scanner type
CREATE OR REPLACE FUNCTION get_scanner_win_rate(scanner VARCHAR(50))
RETURNS TABLE(
    total_trades BIGINT,
    wins BIGINT,
    losses BIGINT,
    win_rate NUMERIC
) AS 66216
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_trades,
        COUNT(CASE WHEN trade_result = 'win' THEN 1 END) as wins,
        COUNT(CASE WHEN trade_result = 'loss' THEN 1 END) as losses,
        ROUND(
            COUNT(CASE WHEN trade_result = 'win' THEN 1 END)::NUMERIC / 
            NULLIF(COUNT(CASE WHEN trade_result IN ('win', 'loss') THEN 1 END), 0) * 100, 
            2
        ) as win_rate
    FROM trade_opportunities t
    JOIN scan_results s ON t.scan_id = s.id
    WHERE s.scan_type = scanner
    AND t.trade_taken = TRUE;
END;
66216 LANGUAGE plpgsql;
