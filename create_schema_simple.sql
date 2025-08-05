-- Crypto Scanner Database Schema
-- Supports BB, ICT, Wyckoff, Elliott Waves, and all scanner types

-- Main scan results table
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

-- Individual trade opportunities
CREATE TABLE IF NOT EXISTS trade_opportunities (
    id SERIAL PRIMARY KEY,
    scan_id INT REFERENCES scan_results(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(20),
    exchange VARCHAR(50),
    timeframe VARCHAR(10),
    bb_score FLOAT,
    probability FLOAT,
    risk_reward_ratio FLOAT,
    current_price DECIMAL(20,10),
    entry_price DECIMAL(20,10),
    stop_loss DECIMAL(20,10),
    target_1 DECIMAL(20,10),
    target_2 DECIMAL(20,10),
    target_3 DECIMAL(20,10),
    rsi FLOAT,
    mfi FLOAT,
    stochastic_k FLOAT,
    volume_surge FLOAT,
    macd_signal VARCHAR(20),
    pattern_type VARCHAR(100),
    pattern_quality VARCHAR(20),
    confluence_score FLOAT,
    historical_win_rate FLOAT,
    category_win_rate FLOAT,
    similar_setups_count INT,
    market_cap DECIMAL(20,2),
    volume_24h DECIMAL(20,2),
    price_change_24h FLOAT,
    scanner_specific_data JSONB,
    trade_taken BOOLEAN DEFAULT FALSE,
    trade_result VARCHAR(20),
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
    parameters JSONB,
    detailed_results JSONB
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
CREATE OR REPLACE VIEW high_probability_trades AS
SELECT 
    t.*,
    s.scan_type,
    s.scan_timestamp
FROM trade_opportunities t
JOIN scan_results s ON t.scan_id = s.id
WHERE t.probability >= 70
ORDER BY t.timestamp DESC;

CREATE OR REPLACE VIEW scanner_performance AS
SELECT 
    scan_type,
    COUNT(*) as total_scans,
    SUM(total_coins_analyzed) as total_coins,
    SUM(premium_trades_found) as total_opportunities,
    AVG(execution_time_seconds) as avg_execution_time,
    MAX(scan_timestamp) as last_scan
FROM scan_results
GROUP BY scan_type;
