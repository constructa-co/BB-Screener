-- Wyckoff Scanner Database Schema
-- File: scripts/wyckoff_integration/setup_wyckoff_database.sql
-- Created: 2025-01-24
-- Purpose: Database schema for Wyckoff scanner signals

-- Ensure the schema exists
CREATE SCHEMA IF NOT EXISTS other_scanners;

-- Create the Wyckoff signals table with all relevant fields
CREATE TABLE IF NOT EXISTS other_scanners.wyckoff_signals (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    signal_id       VARCHAR(120) UNIQUE NOT NULL,    -- deterministic ID to de-dupe signals
    phase           VARCHAR(30) NOT NULL CHECK (phase IN ('ACCUMULATION','DISTRIBUTION','MARKUP','MARKDOWN')),
    pattern_type    VARCHAR(30) NOT NULL CHECK (pattern_type IN ('SPRING','UPTHRUST','TEST','BREAKOUT','ACCUMULATION_RANGE','DISTRIBUTION_RANGE')),
    pattern_duration INTEGER,                        -- hours of accumulation/distribution range
    trade_direction VARCHAR(10) CHECK (trade_direction IN ('LONG','SHORT','NEUTRAL')),
    entry_price     DECIMAL(19,8),
    stop_loss       DECIMAL(19,8),
    target_1        DECIMAL(19,8),
    target_2        DECIMAL(19,8),
    risk_reward_1   DECIMAL(10,4),
    risk_reward_2   DECIMAL(10,4),
    setup_score     INTEGER CHECK (setup_score >= 0 AND setup_score <= 100),
    volume_confirmation DECIMAL(10,4),               -- volume_ratio from scanner
    strength_score  DECIMAL(10,4),                   -- spring_strength or upthrust_strength
    entry_signal    VARCHAR(50),                     -- e.g. 'IMMEDIATE' or 'WAIT'
    wait_condition  TEXT,                            -- e.g. description of what to wait for
    current_price   DECIMAL(19,8),
    support_level   DECIMAL(19,8),
    resistance_level DECIMAL(19,8),
    range_size_pct  DECIMAL(10,4),
    spring_detected     BOOLEAN,
    upthrust_detected   BOOLEAN,
    accumulation_duration INTEGER,
    distribution_duration INTEGER,
    hold_time_estimate VARCHAR(50),
    source_scanner  VARCHAR(50) DEFAULT 'wyckoff_1h_scanner_r1',
    scanner_version VARCHAR(20) DEFAULT '1.0.0',
    algorithm_parameters JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes to optimize lookups on symbol/timeframe and phase/score
CREATE INDEX IF NOT EXISTS wyckoff_signals_symbol_timeframe_idx
    ON other_scanners.wyckoff_signals (symbol, timeframe);
CREATE INDEX IF NOT EXISTS wyckoff_signals_phase_score_idx
    ON other_scanners.wyckoff_signals (phase, setup_score DESC);
-- Unique index for signal_id to prevent duplicate entries of the same signal
CREATE UNIQUE INDEX IF NOT EXISTS wyckoff_signals_signal_uniq
    ON other_scanners.wyckoff_signals (signal_id);

-- Add comment to table
COMMENT ON TABLE other_scanners.wyckoff_signals IS 'Wyckoff scanner signals for accumulation and distribution patterns';
COMMENT ON COLUMN other_scanners.wyckoff_signals.signal_id IS 'Deterministic unique identifier for each Wyckoff setup';
COMMENT ON COLUMN other_scanners.wyckoff_signals.setup_score IS 'Quality score 0-100, only scores >= 60 are logged';
COMMENT ON COLUMN other_scanners.wyckoff_signals.phase IS 'Wyckoff phase: ACCUMULATION, DISTRIBUTION, MARKUP, MARKDOWN';
COMMENT ON COLUMN other_scanners.wyckoff_signals.pattern_type IS 'Specific pattern: SPRING, UPTHRUST, TEST, BREAKOUT, etc.';
