-- Fibonacci Scanner Database Setup
-- Creates isolated fibonacci_signals table in other_scanners schema
-- Follows exact pattern from Elliott Wave integration

-- Ensure other_scanners schema exists
CREATE SCHEMA IF NOT EXISTS other_scanners;

-- Create fibonacci_signals table with complete isolation
CREATE TABLE IF NOT EXISTS other_scanners.fibonacci_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    signal_id VARCHAR(50) UNIQUE NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    fibonacci_level DECIMAL(6,4) NOT NULL,
    price_level DECIMAL(19,4) NOT NULL,
    current_price DECIMAL(19,4) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    volume_confirmation BOOLEAN DEFAULT FALSE,
    momentum_confirmation BOOLEAN DEFAULT FALSE,
    swing_high DECIMAL(19,4),
    swing_low DECIMAL(19,4),
    trend_direction VARCHAR(10),
    validation_rules_passed JSONB,
    scanner_version VARCHAR(20),
    algorithm_parameters JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_fib_symbol_time ON other_scanners.fibonacci_signals(symbol, timeframe, detected_at);
CREATE INDEX IF NOT EXISTS idx_fib_confidence ON other_scanners.fibonacci_signals(confidence_score);
CREATE INDEX IF NOT EXISTS idx_fib_signal_type ON other_scanners.fibonacci_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_fib_detected_at ON other_scanners.fibonacci_signals(detected_at);

-- Create GIN index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_fib_validation_rules ON other_scanners.fibonacci_signals USING GIN (validation_rules_passed);
CREATE INDEX IF NOT EXISTS idx_fib_algorithm_params ON other_scanners.fibonacci_signals USING GIN (algorithm_parameters);

-- Grant permissions (adjust user as needed)
-- GRANT SELECT, INSERT, UPDATE ON other_scanners.fibonacci_signals TO your_user;
-- GRANT USAGE, SELECT ON SEQUENCE other_scanners.fibonacci_signals_id_seq TO your_user;

-- Add comments for documentation
COMMENT ON TABLE other_scanners.fibonacci_signals IS 'Fibonacci scanner signals with complete isolation from other scanners';
COMMENT ON COLUMN other_scanners.fibonacci_signals.symbol IS 'Trading symbol (e.g., BTC, ETH)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.timeframe IS 'Chart timeframe (e.g., 5m, 1h, 4h)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.signal_id IS 'Unique identifier for this signal';
COMMENT ON COLUMN other_scanners.fibonacci_signals.signal_type IS 'Type of signal (SUPPORT, RESISTANCE, BREAKOUT, BOUNCE)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.fibonacci_level IS 'Fibonacci level (0.236, 0.382, 0.500, 0.618, 0.786)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.price_level IS 'Price at the Fibonacci level';
COMMENT ON COLUMN other_scanners.fibonacci_signals.current_price IS 'Current market price at signal detection';
COMMENT ON COLUMN other_scanners.fibonacci_signals.confidence_score IS 'Signal confidence (0.0 to 1.0)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.volume_confirmation IS 'Whether volume confirms the signal';
COMMENT ON COLUMN other_scanners.fibonacci_signals.momentum_confirmation IS 'Whether momentum confirms the signal';
COMMENT ON COLUMN other_scanners.fibonacci_signals.swing_high IS 'Swing high price used for Fibonacci calculation';
COMMENT ON COLUMN other_scanners.fibonacci_signals.swing_low IS 'Swing low price used for Fibonacci calculation';
COMMENT ON COLUMN other_scanners.fibonacci_signals.trend_direction IS 'Overall trend direction (BULLISH, BEARISH, NEUTRAL)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.validation_rules_passed IS 'JSONB object containing validation results';
COMMENT ON COLUMN other_scanners.fibonacci_signals.scanner_version IS 'Version of the Fibonacci scanner';
COMMENT ON COLUMN other_scanners.fibonacci_signals.algorithm_parameters IS 'JSONB object containing algorithm configuration';
COMMENT ON COLUMN other_scanners.fibonacci_signals.detected_at IS 'Timestamp when signal was detected';
COMMENT ON COLUMN other_scanners.fibonacci_signals.created_at IS 'Timestamp when record was created';

-- Verify table creation
SELECT 
    table_name, 
    column_name, 
    data_type, 
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'other_scanners' 
AND table_name = 'fibonacci_signals'
ORDER BY ordinal_position;

-- Show table statistics
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'other_scanners' 
AND tablename = 'fibonacci_signals';
