-- Enhanced Fibonacci Scanner Database Schema
-- Adds missing columns to capture complete trading setup data

-- Add new columns to capture all trading data
ALTER TABLE other_scanners.fibonacci_signals 
ADD COLUMN IF NOT EXISTS quality_score INTEGER,
ADD COLUMN IF NOT EXISTS move_percentage DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS stop_loss_price DECIMAL(19,4),
ADD COLUMN IF NOT EXISTS entry_timing_status VARCHAR(20),
ADD COLUMN IF NOT EXISTS target_1_price DECIMAL(19,4),
ADD COLUMN IF NOT EXISTS target_1_percentage DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS target_1_risk_reward DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS target_2_price DECIMAL(19,4),
ADD COLUMN IF NOT EXISTS target_2_percentage DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS target_2_risk_reward DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS target_3_price DECIMAL(19,4),
ADD COLUMN IF NOT EXISTS target_3_percentage DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS target_3_risk_reward DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS entry_price DECIMAL(19,4),
ADD COLUMN IF NOT EXISTS risk_percentage DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS setup_stage VARCHAR(30),
ADD COLUMN IF NOT EXISTS trading_metadata JSONB;

-- Add comments for documentation
COMMENT ON COLUMN other_scanners.fibonacci_signals.quality_score IS 'Quality score (0-100) from original scanner';
COMMENT ON COLUMN other_scanners.fibonacci_signals.move_percentage IS 'Percentage move from swing low to swing high';
COMMENT ON COLUMN other_scanners.fibonacci_signals.stop_loss_price IS 'Stop loss price level';
COMMENT ON COLUMN other_scanners.fibonacci_signals.entry_timing_status IS 'Entry timing: NOW, WAIT, CLOSE';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_1_price IS 'Target 1 price (38.2% level)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_1_percentage IS 'Target 1 gain percentage';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_1_risk_reward IS 'Target 1 risk/reward ratio';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_2_price IS 'Target 2 price (50% level)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_2_percentage IS 'Target 2 gain percentage';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_2_risk_reward IS 'Target 2 risk/reward ratio';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_3_price IS 'Target 3 price (61.8% level)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_3_percentage IS 'Target 3 gain percentage';
COMMENT ON COLUMN other_scanners.fibonacci_signals.target_3_risk_reward IS 'Target 3 risk/reward ratio';
COMMENT ON COLUMN other_scanners.fibonacci_signals.entry_price IS 'Entry price at Fibonacci level';
COMMENT ON COLUMN other_scanners.fibonacci_signals.risk_percentage IS 'Risk percentage (stop loss distance)';
COMMENT ON COLUMN other_scanners.fibonacci_signals.setup_stage IS 'Setup stage: immediate_entry, approaching_entry, waiting_for_pullback';
COMMENT ON COLUMN other_scanners.fibonacci_signals.trading_metadata IS 'Additional trading metadata in JSONB format';

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_fib_quality_score ON other_scanners.fibonacci_signals(quality_score);
CREATE INDEX IF NOT EXISTS idx_fib_entry_timing ON other_scanners.fibonacci_signals(entry_timing_status);
CREATE INDEX IF NOT EXISTS idx_fib_setup_stage ON other_scanners.fibonacci_signals(setup_stage);
CREATE INDEX IF NOT EXISTS idx_fib_trading_metadata ON other_scanners.fibonacci_signals USING GIN (trading_metadata);

-- Verify the enhanced schema
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
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
AND tablename = 'fibonacci_signals'
ORDER BY attname;
