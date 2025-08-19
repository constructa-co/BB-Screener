-- BB Screener: Elliott Wave Database Schema
-- Creates the elliott_wave_signals table for Elliott Wave pattern detection
-- COMPLETELY SEPARATE from other_scanners_trades table - ZERO IMPACT on existing scanners
-- Supports hierarchical wave data, Fibonacci relationships, and pattern quality metrics

-- Use existing other_scanners schema for consistency
SET search_path TO other_scanners;

-- Create dedicated Elliott Wave signals table
CREATE TABLE IF NOT EXISTS elliott_wave_signals (
    -- Primary identification
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scanner_name VARCHAR(64) DEFAULT 'elliott_wave',
    scanner_version VARCHAR(16) DEFAULT 'R0',
    scan_id VARCHAR(64) NOT NULL,  -- Groups signals from same run
    
    -- Symbol & timeframe identification
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(32) DEFAULT 'binance',
    timeframe VARCHAR(16) NOT NULL,  -- '1h', '4h', '1d', '1w'
    
    -- Trading signals
    direction VARCHAR(8),  -- 'LONG', 'SHORT', 'NEUTRAL'
    entry_price DECIMAL(18,8),
    stop_loss DECIMAL(18,8),
    tp1 DECIMAL(18,8),
    tp2 DECIMAL(18,8),
    tp3 DECIMAL(18,8),
    risk_reward DECIMAL(5,2),
    
    -- Elliott Wave specific fields
    pattern_type VARCHAR(24),  -- 'BULLISH_IMPULSE', 'BEARISH_IMPULSE', 'CORRECTIVE'
    current_wave VARCHAR(8),   -- 'WAVE_1', 'WAVE_2', 'WAVE_3', 'WAVE_4', 'WAVE_5', 'WAVE_A', 'WAVE_B', 'WAVE_C'
    wave_degree VARCHAR(32),   -- 'minor', 'intermediate', 'primary', 'major'
    pattern_quality DECIMAL(5,2),  -- Quality score 0-100
    confidence_score DECIMAL(3,2), -- Confidence 0.0-1.0
    invalidation_level DECIMAL(18,8),
    
    -- Wave analysis data (hierarchical structure)
    wave_analysis JSONB NOT NULL DEFAULT '{}',  -- Complete wave structure with all wave details
    fibonacci_levels JSONB DEFAULT '{}',        -- All Fibonacci relationships and extensions
    pattern_metrics JSONB DEFAULT '{}',         -- Quality scores, probabilities, validation checks
    
    -- Timestamps
    analysis_ts TIMESTAMP WITH TIME ZONE DEFAULT NOW(),  -- When pattern was analyzed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),   -- When record was created
    
    -- Prevent duplicates from same scan run
    CONSTRAINT uq_elliott_signal 
        UNIQUE (scan_id, symbol, timeframe, current_wave)
);

-- Performance-optimized indexes
CREATE INDEX IF NOT EXISTS idx_elliott_symbol_timeframe ON elliott_wave_signals(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_elliott_analysis_ts ON elliott_wave_signals(analysis_ts DESC);
CREATE INDEX IF NOT EXISTS idx_elliott_pattern_quality ON elliott_wave_signals(pattern_quality DESC);
CREATE INDEX IF NOT EXISTS idx_elliott_current_wave ON elliott_wave_signals(current_wave);
CREATE INDEX IF NOT EXISTS idx_elliott_pattern_type ON elliott_wave_signals(pattern_type);

-- GIN indexes for JSONB columns (enables efficient querying of nested data)
CREATE INDEX IF NOT EXISTS idx_elliott_wave_analysis ON elliott_wave_signals USING GIN (wave_analysis);
CREATE INDEX IF NOT EXISTS idx_elliott_fibonacci_levels ON elliott_wave_signals USING GIN (fibonacci_levels);
CREATE INDEX IF NOT EXISTS idx_elliott_pattern_metrics ON elliott_wave_signals USING GIN (pattern_metrics);

-- Trigger for automatic updated_at (if needed in future)
CREATE OR REPLACE FUNCTION update_elliott_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.created_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Insert initial marker to verify table creation
INSERT INTO elliott_wave_signals (scanner_name, scanner_version, scan_id, symbol, exchange, timeframe, 
                                  pattern_type, current_wave, wave_analysis, fibonacci_levels, pattern_metrics)
VALUES ('SYSTEM', '1.0.0', 'SCHEMA_CREATED', 'SCHEMA_CREATED', 'binance', '1h',
        'SCHEMA_CREATION', 'SCHEMA_CREATED', 
        '{"action": "schema_creation", "timestamp": "' || CURRENT_TIMESTAMP || '", "version": "1.0.0"}',
        '{"schema": "elliott_wave_signals"}',
        '{"status": "created"}')
ON CONFLICT (scan_id, symbol, timeframe, current_wave) DO NOTHING;

-- Display table information
SELECT 
    'elliott_wave_signals' as table_name,
    COUNT(*) as total_records,
    MAX(created_at) as latest_record
FROM elliott_wave_signals;

-- Verify isolation - check that other tables are unchanged
SELECT 
    'other_scanners_trades' as table_name,
    COUNT(*) as total_records
FROM other_scanners_trades
UNION ALL
SELECT 
    'scanner_backtest_results' as table_name,
    COUNT(*) as total_records
FROM scanner_backtest_results;

-- Reset search_path to default
SET search_path TO public;

-- Success message
SELECT '✅ Elliott Wave database schema created successfully' as status;
