-- Enhanced Market Data Tables for ML Analysis
-- Based on Claude's superior approach

-- Create market regime table for daily market conditions
CREATE TABLE IF NOT EXISTS market_regime (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER REFERENCES scan_results(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Key market indicators
    btc_dominance DECIMAL(5,2),
    eth_dominance DECIMAL(5,2),
    fear_greed_index INTEGER,
    alt_season_indicator BOOLEAN,
    market_health_score DECIMAL(5,2),
    regime_type VARCHAR(50),
    regime_confidence DECIMAL(5,2),
    position_multiplier DECIMAL(4,2),
    
    -- Traditional market correlation
    spy_correlation DECIMAL(5,2),
    vix_level DECIMAL(6,2),
    dxy_trend VARCHAR(20),
    
    -- Comprehensive regime data as JSON
    regime_data JSONB,
    
    -- Indexes for querying
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create market overview table for aggregate statistics
CREATE TABLE IF NOT EXISTS market_overview (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER REFERENCES scan_results(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Overall statistics
    total_bounces INTEGER,
    coins_analyzed INTEGER,
    overall_success_rate DECIMAL(5,2),
    overall_win_rate DECIMAL(5,2),
    avg_profit_pct DECIMAL(6,2),
    avg_loss_pct DECIMAL(6,2),
    profit_factor DECIMAL(6,2),
    
    -- BB-specific metrics
    bb_squeeze_effectiveness DECIMAL(5,2),
    bb_expansion_effectiveness DECIMAL(5,2),
    bb_reversal_effectiveness DECIMAL(5,2),
    
    -- Timing metrics
    avg_time_to_target DECIMAL(6,2),
    avg_time_to_peak DECIMAL(6,2),
    
    -- Complete overview data as JSON
    bb_performance_metrics JSONB,
    technical_indicators_performance JSONB,
    timing_analysis JSONB,
    overview_data JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create market metadata table for sector/tier analysis
CREATE TABLE IF NOT EXISTS market_metadata (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER REFERENCES scan_results(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Summary counts
    large_cap_count INTEGER,
    mid_cap_count INTEGER,
    small_cap_count INTEGER,
    micro_cap_count INTEGER,
    
    -- Sector counts
    defi_count INTEGER,
    layer1_count INTEGER,
    layer2_count INTEGER,
    
    -- Performance by category (JSON)
    sector_performance JSONB,
    market_cap_tier_performance JSONB,
    liquidity_analysis JSONB,
    expected_performance_by_tier JSONB,
    metadata_details JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_market_regime_scan_id ON market_regime(scan_id);
CREATE INDEX idx_market_regime_timestamp ON market_regime(timestamp);
CREATE INDEX idx_market_regime_fear_greed ON market_regime(fear_greed_index);

CREATE INDEX idx_market_overview_scan_id ON market_overview(scan_id);
CREATE INDEX idx_market_overview_timestamp ON market_overview(timestamp);
CREATE INDEX idx_market_overview_success_rate ON market_overview(overall_success_rate);

CREATE INDEX idx_market_metadata_scan_id ON market_metadata(scan_id);
CREATE INDEX idx_market_metadata_timestamp ON market_metadata(timestamp);

-- Create a view for easy ML data access
CREATE OR REPLACE VIEW ml_training_data AS
SELECT 
    t.*,
    mr.btc_dominance,
    mr.fear_greed_index,
    mr.alt_season_indicator,
    mr.market_health_score,
    mr.regime_type,
    mo.overall_success_rate as market_success_rate,
    mo.bb_squeeze_effectiveness,
    mm.sector_performance,
    mm.market_cap_tier_performance
FROM trade_opportunities t
LEFT JOIN market_regime mr ON t.scan_id = mr.scan_id
LEFT JOIN market_overview mo ON t.scan_id = mo.scan_id
LEFT JOIN market_metadata mm ON t.scan_id = mm.scan_id;
