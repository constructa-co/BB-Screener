-- Fair Value Gap Scanner Database Schema - Production Version
-- Compatible with PostgreSQL 10+ (no GENERATED ALWAYS columns)

CREATE SCHEMA IF NOT EXISTS other_scanners;

-- Drop existing table if needed (for clean install)
-- DROP TABLE IF EXISTS other_scanners.fvg_signals CASCADE;

CREATE TABLE IF NOT EXISTS other_scanners.fvg_signals (
    id BIGSERIAL PRIMARY KEY,
    
    -- Core Identity
    signal_id VARCHAR(120) UNIQUE NOT NULL,
    symbol VARCHAR(24) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- FVG Pattern Data
    gap_type VARCHAR(20) NOT NULL CHECK (gap_type IN ('BULLISH', 'BEARISH')),
    gap_high DECIMAL(20,8) NOT NULL,
    gap_low DECIMAL(20,8) NOT NULL,
    gap_size DECIMAL(20,8),
    gap_size_pct DECIMAL(10,4),
    
    -- Price Context
    current_price DECIMAL(20,8),
    entry_price DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    target_1 DECIMAL(20,8),
    target_2 DECIMAL(20,8),
    target_3 DECIMAL(20,8),
    risk_reward_1 DECIMAL(10,4),
    risk_reward_2 DECIMAL(10,4),
    risk_reward_3 DECIMAL(10,4),
    
    -- Fibonacci Integration
    fib_level DECIMAL(10,4),
    fib_confluence BOOLEAN DEFAULT FALSE,
    fib_confluence_score INTEGER,
    
    -- Quality Metrics
    setup_score INTEGER CHECK (setup_score >= 0 AND setup_score <= 100),
    volume_at_gap BIGINT,
    volume_confirmation BOOLEAN DEFAULT FALSE,
    momentum_confirmation BOOLEAN DEFAULT FALSE,
    
    -- Status Tracking
    gap_status VARCHAR(20) DEFAULT 'OPEN' CHECK (gap_status IN ('OPEN', 'PARTIAL', 'FILLED', 'EXPIRED')),
    fill_percentage DECIMAL(10,4) DEFAULT 0,
    gap_age_minutes INTEGER,
    expires_at TIMESTAMPTZ,
    
    -- Metadata
    scanner_version VARCHAR(20) DEFAULT '1.0.0',
    algorithm_parameters JSONB,
    source VARCHAR(50) DEFAULT 'fvg_scanner'
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_fvg_symbol_timeframe 
    ON other_scanners.fvg_signals(symbol, timeframe, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_fvg_active 
    ON other_scanners.fvg_signals(gap_status, symbol) 
    WHERE gap_status = 'OPEN';
CREATE INDEX IF NOT EXISTS idx_fvg_quality 
    ON other_scanners.fvg_signals(setup_score DESC, detected_at DESC) 
    WHERE setup_score >= 60;
CREATE INDEX IF NOT EXISTS idx_fvg_gap_type 
    ON other_scanners.fvg_signals(gap_type, detected_at DESC);

-- View for computed midpoint (instead of GENERATED column)
CREATE OR REPLACE VIEW other_scanners.v_fvg_signals AS
SELECT 
    *,
    (gap_high + gap_low) / 2.0 AS gap_midpoint,
    CASE 
        WHEN gap_high + gap_low <> 0 
        THEN ((gap_high - gap_low) / ((gap_high + gap_low) / 2.0)) * 100.0
        ELSE NULL
    END AS gap_width_pct_computed
FROM other_scanners.fvg_signals;

-- Grant permissions
GRANT ALL ON SCHEMA other_scanners TO bb_user;
GRANT ALL ON ALL TABLES IN SCHEMA other_scanners TO bb_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA other_scanners TO bb_user;
