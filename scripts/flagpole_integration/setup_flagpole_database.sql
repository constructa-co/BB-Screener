-- Flagpole Scanner Database Schema
-- Comprehensive schema for flagpole and triangle pattern signals

CREATE SCHEMA IF NOT EXISTS other_scanners;

CREATE TABLE IF NOT EXISTS other_scanners.flagpole_signals (
    id BIGSERIAL PRIMARY KEY,
    signal_id VARCHAR(120) UNIQUE NOT NULL,
    
    -- Core identifiers
    symbol VARCHAR(24) NOT NULL,
    timeframe VARCHAR(10) DEFAULT '5m',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Pattern classification (parsed from output)
    pattern_type VARCHAR(50) NOT NULL,  -- 'Pennant', 'Flag', 'Triangle'
    pattern_details VARCHAR(100),        -- Full string like 'Pennant (7.1% pole)'
    direction VARCHAR(20) NOT NULL,      -- 'Bullish', 'Bearish'
    
    -- Price levels (all from terminal output)
    current_price DECIMAL(20,8),
    breakout_level DECIMAL(20,8),
    target_price DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    
    -- Performance metrics
    potential_pct DECIMAL(10,2),        -- "Potential: 12.5%"
    risk_pct DECIMAL(10,2),              -- "Risk: 0.8%"
    risk_reward DECIMAL(10,2),           -- "R:R: 15.9:1"
    
    -- Pattern characteristics
    pole_pct DECIMAL(10,2),              -- "Pole: 7.1%"
    vol_decline_pct DECIMAL(10,2),       -- "Vol Decline: 57%"
    slope_pct DECIMAL(10,2),             -- "Slope: 3.9%"
    age_candles INTEGER,                 -- "Age: 17 candles"
    
    -- Quality indicators
    score INTEGER CHECK (score BETWEEN 0 AND 100),
    quality_indicators TEXT[],           -- Array: ['Strong Vol', 'Ready']
    quality_raw TEXT,                    -- Full quality string
    
    -- Status flags
    is_ready BOOLEAN DEFAULT FALSE,      -- Has '⚡ Ready' indicator
    has_strong_vol BOOLEAN DEFAULT FALSE, -- Has '📊 Strong Vol'
    has_fast_pole BOOLEAN DEFAULT FALSE,  -- Has '🚀 Fast Pole'
    
    -- Metadata
    expires_at TIMESTAMPTZ,
    scanner_version VARCHAR(20) DEFAULT '1.0.0',
    raw_output TEXT,                     -- Store complete scanner output
    source VARCHAR(50) DEFAULT 'flagpole_5m_scanner'
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_flagpole_symbol_time ON other_scanners.flagpole_signals(symbol, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_flagpole_score ON other_scanners.flagpole_signals(score DESC, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_flagpole_direction ON other_scanners.flagpole_signals(direction, pattern_type);
CREATE INDEX IF NOT EXISTS idx_flagpole_quality ON other_scanners.flagpole_signals(symbol) 
    WHERE score >= 90 AND is_ready = TRUE;

-- Permissions
GRANT ALL ON SCHEMA other_scanners TO bb_user;
GRANT ALL ON ALL TABLES IN SCHEMA other_scanners TO bb_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA other_scanners TO bb_user;
