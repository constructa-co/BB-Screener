-- Supply & Demand Scanner Database Setup
-- Following Claude's Research Report Schema
-- Run this to create the Supply & Demand scanner database infrastructure

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS other_scanners;

-- Main Supply & Demand Zones Table
CREATE TABLE IF NOT EXISTS other_scanners.supply_demand_zones (
    id BIGSERIAL PRIMARY KEY,
    
    -- Core Zone Data
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    zone_id VARCHAR(50) UNIQUE NOT NULL,
    zone_type VARCHAR(20) NOT NULL CHECK (zone_type IN ('SUPPLY', 'DEMAND')),
    
    -- Zone Boundaries (8 decimal precision for crypto)
    zone_top DECIMAL(19,8) NOT NULL,
    zone_bottom DECIMAL(19,8) NOT NULL,
    zone_width DECIMAL(19,8) GENERATED ALWAYS AS (zone_top - zone_bottom) STORED,
    zone_midpoint DECIMAL(19,8) GENERATED ALWAYS AS ((zone_top + zone_bottom) / 2) STORED,
    
    -- Zone Characteristics
    zone_strength DECIMAL(5,2) NOT NULL CHECK (zone_strength >= 0 AND zone_strength <= 100),
    touch_count INTEGER DEFAULT 1,
    breakout_count INTEGER DEFAULT 0,
    respect_count INTEGER DEFAULT 0,
    
    -- Price Action Context
    current_price DECIMAL(19,8) NOT NULL,
    distance_to_zone DECIMAL(19,8),
    distance_percentage DECIMAL(10,4),
    price_position VARCHAR(20) CHECK (price_position IN ('ABOVE', 'BELOW', 'INSIDE')),
    
    -- Volume Profile
    zone_volume BIGINT,
    average_volume BIGINT,
    volume_ratio DECIMAL(10,4),
    volume_confirmation BOOLEAN DEFAULT FALSE,
    
    -- Trading Signals (Pre-calculated for efficiency)
    entry_price DECIMAL(19,8),
    stop_loss DECIMAL(19,8),
    target_1 DECIMAL(19,8),
    target_2 DECIMAL(19,8),
    target_3 DECIMAL(19,8),
    risk_reward_1 DECIMAL(10,4),
    risk_reward_2 DECIMAL(10,4),
    risk_reward_3 DECIMAL(10,4),
    
    -- Zone Formation
    formation_type VARCHAR(50) CHECK (formation_type IN 
        ('RALLY_BASE_RALLY', 'DROP_BASE_DROP', 'RALLY_BASE_DROP', 'DROP_BASE_RALLY')),
    formation_candles INTEGER,
    formation_start TIMESTAMP,
    formation_end TIMESTAMP,
    
    -- Quality Metrics
    quality_score INTEGER DEFAULT 0 CHECK (quality_score >= 0 AND quality_score <= 100),
    freshness_score INTEGER DEFAULT 0,
    reliability_score INTEGER DEFAULT 0,
    
    -- Metadata
    scanner_version VARCHAR(20) DEFAULT '1.0.0',
    algorithm_parameters JSONB,
    validation_status VARCHAR(20) DEFAULT 'ACTIVE',
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_tested_at TIMESTAMP,
    expires_at TIMESTAMP,
    
    -- Post-mortem tracking
    post_mortem_status VARCHAR(20) DEFAULT 'PENDING',
    actual_bounce_price DECIMAL(19,8),
    actual_penetration_depth DECIMAL(10,4),
    zone_held BOOLEAN,
    post_mortem_notes TEXT
);

-- Zone Test History Table for Post-Mortem Analysis
CREATE TABLE IF NOT EXISTS other_scanners.supply_demand_zone_tests (
    id BIGSERIAL PRIMARY KEY,
    zone_id VARCHAR(50) REFERENCES other_scanners.supply_demand_zones(zone_id),
    test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    test_price DECIMAL(19,8),
    test_result VARCHAR(20) CHECK (test_result IN ('RESPECTED', 'BROKEN', 'PARTIAL')),
    penetration_depth DECIMAL(10,4),
    volume_on_test BIGINT,
    subsequent_move DECIMAL(10,4),
    notes TEXT
);

-- Optimized Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_sd_symbol_timeframe ON other_scanners.supply_demand_zones(symbol, timeframe, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_sd_active_zones ON other_scanners.supply_demand_zones(symbol, timeframe, quality_score DESC) 
    WHERE validation_status = 'ACTIVE' AND expires_at > NOW();
CREATE INDEX IF NOT EXISTS idx_sd_near_price ON other_scanners.supply_demand_zones(symbol, distance_percentage) 
    WHERE validation_status = 'ACTIVE' AND distance_percentage <= 2.0;
CREATE INDEX IF NOT EXISTS idx_sd_pending_postmortem ON other_scanners.supply_demand_zones(detected_at) 
    WHERE post_mortem_status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_sd_params_gin ON other_scanners.supply_demand_zones USING gin(algorithm_parameters);

-- Zone test history indexes
CREATE INDEX IF NOT EXISTS idx_sd_test_zone_id ON other_scanners.supply_demand_zone_tests(zone_id, test_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sd_test_result ON other_scanners.supply_demand_zone_tests(test_result, test_timestamp DESC);

-- Grant permissions (adjust username as needed)
-- GRANT ALL PRIVILEGES ON SCHEMA other_scanners TO your_db_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA other_scanners TO your_db_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA other_scanners TO your_db_user;

-- Verify creation
SELECT 'Supply & Demand database setup complete' as status;
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'other_scanners' 
AND table_name LIKE 'supply_demand%';
