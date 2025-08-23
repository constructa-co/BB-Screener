-- File: scripts/trend_following_integration/setup_trend_following_database.sql
-- Trend Following Scanner Database Schema - Optimized Hybrid Approach
-- Combines best practices from all research reviews

-- Create schema (idempotent)
CREATE SCHEMA IF NOT EXISTS other_scanners;

-- Main table (idempotent)
CREATE TABLE IF NOT EXISTS other_scanners.trend_following_signals (
    id BIGSERIAL PRIMARY KEY,
    
    -- Core Signal Data (ChatGPT's simplicity)
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    signal_id VARCHAR(80) UNIQUE NOT NULL,
    signal_type VARCHAR(20) NOT NULL CHECK (signal_type IN ('BULLISH', 'BEARISH', 'NEUTRAL')),
    
    -- Trend Metrics (from base scanner)
    trend_direction VARCHAR(20) NOT NULL,
    trend_strength DECIMAL(5,2) CHECK (trend_strength >= 0 AND trend_strength <= 100),
    momentum_score DECIMAL(5,2),
    volume_trend VARCHAR(20),
    
    -- Moving Averages (if available)
    ma_20 DECIMAL(19,8),
    ma_50 DECIMAL(19,8),
    ma_100 DECIMAL(19,8),
    price_to_ma50_distance DECIMAL(10,4),
    
    -- Price Levels
    current_price DECIMAL(19,8) NOT NULL,
    atr_value DECIMAL(19,8),
    
    -- Trade Setup (from scanner output)
    entry_price DECIMAL(19,8),
    stop_loss DECIMAL(19,8),
    target_1 DECIMAL(19,8),
    target_2 DECIMAL(19,8),
    target_3 DECIMAL(19,8),
    risk_reward_1 DECIMAL(10,4),
    risk_reward_2 DECIMAL(10,4),
    risk_reward_3 DECIMAL(10,4),
    risk_pct DECIMAL(10,4),
    entry_timing VARCHAR(32),
    
    -- Quality Metrics
    confidence_score DECIMAL(5,2) DEFAULT 0,
    quality_score INTEGER DEFAULT 0 CHECK (quality_score >= 0 AND quality_score <= 100),
    
    -- Metadata
    scanner_version VARCHAR(20) DEFAULT '1.0.0',
    algorithm_parameters JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    -- Post-mortem
    post_mortem_status VARCHAR(20) DEFAULT 'PENDING',
    actual_outcome VARCHAR(20),
    profit_loss_percentage DECIMAL(10,4),
    post_mortem_notes TEXT
);

-- Optimized Indexes (Perplexity's performance focus)
CREATE INDEX IF NOT EXISTS idx_tf_symbol_timeframe ON other_scanners.trend_following_signals(symbol, timeframe, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_tf_active_signals ON other_scanners.trend_following_signals(symbol, confidence_score DESC) 
    WHERE expires_at IS NULL OR expires_at > NOW();
CREATE INDEX IF NOT EXISTS idx_tf_signal_type ON other_scanners.trend_following_signals(signal_type, trend_strength DESC);
CREATE INDEX IF NOT EXISTS idx_tf_pending_postmortem ON other_scanners.trend_following_signals(detected_at) 
    WHERE post_mortem_status = 'PENDING';

-- Performance optimization indexes
CREATE INDEX IF NOT EXISTS idx_tf_quality_active ON other_scanners.trend_following_signals(quality_score DESC, detected_at DESC)
    WHERE quality_score >= 60 AND (expires_at IS NULL OR expires_at > NOW());

-- JSONB optimization for algorithm parameters
CREATE INDEX IF NOT EXISTS idx_tf_algorithm_gin ON other_scanners.trend_following_signals 
    USING gin (algorithm_parameters) 
    WHERE algorithm_parameters IS NOT NULL;

-- Verify table creation
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'other_scanners' AND table_name = 'trend_following_signals') THEN
        RAISE NOTICE '✅ Trend Following signals table created successfully';
    ELSE
        RAISE EXCEPTION '❌ Failed to create trend_following_signals table';
    END IF;
END $$;
