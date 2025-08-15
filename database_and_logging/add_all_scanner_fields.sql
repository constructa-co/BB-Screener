-- Add fields for ALL scanner types to future-proof the database
ALTER TABLE trade_opportunities 

-- ICT Scanner Gap fields
ADD COLUMN IF NOT EXISTS gap_high DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS gap_low DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS gap_size_pct DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS gap_type VARCHAR(20),
ADD COLUMN IF NOT EXISTS gap_fill_percentage DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS gap_age_hours INTEGER,

-- ICT Levels and Structure
ADD COLUMN IF NOT EXISTS swing_high DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS swing_low DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS order_block_high DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS order_block_low DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS breaker_block_high DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS breaker_block_low DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS fvg_high DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS fvg_low DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS liquidity_sweep_level DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS equilibrium_level DECIMAL(20,8),

-- Fibonacci Levels
ADD COLUMN IF NOT EXISTS fib_236 DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS fib_382 DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS fib_500 DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS fib_618 DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS fib_786 DECIMAL(20,8),

-- Volume Scanner fields
ADD COLUMN IF NOT EXISTS volume_surge_multiplier DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS relative_volume DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS vwap DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS vwap_deviation DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS volume_profile_poc DECIMAL(20,8),

-- Pattern Recognition fields
ADD COLUMN IF NOT EXISTS pattern_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS pattern_target DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS pattern_reliability DECIMAL(5,2),

-- Support/Resistance Levels
ADD COLUMN IF NOT EXISTS major_resistance_1 DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS major_support_1 DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS pivot_point DECIMAL(20,8);

-- Create indexes for frequently queried fields
CREATE INDEX IF NOT EXISTS idx_gap_levels ON trade_opportunities(gap_high, gap_low);
CREATE INDEX IF NOT EXISTS idx_gap_size ON trade_opportunities(gap_size_pct);
CREATE INDEX IF NOT EXISTS idx_swing_levels ON trade_opportunities(swing_high, swing_low);
CREATE INDEX IF NOT EXISTS idx_order_blocks ON trade_opportunities(order_block_high, order_block_low);
CREATE INDEX IF NOT EXISTS idx_fib_618 ON trade_opportunities(fib_618);
CREATE INDEX IF NOT EXISTS idx_volume_surge ON trade_opportunities(volume_surge_multiplier);
CREATE INDEX IF NOT EXISTS idx_pattern_type ON trade_opportunities(pattern_type);
CREATE INDEX IF NOT EXISTS idx_vwap ON trade_opportunities(vwap);
CREATE INDEX IF NOT EXISTS idx_sr_levels ON trade_opportunities(major_resistance_1, major_support_1);
