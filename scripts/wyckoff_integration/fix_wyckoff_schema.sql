-- Fix Wyckoff Database Schema
-- Add missing columns to match the expected table structure

-- Add missing columns
ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS position_size DECIMAL(10,4) DEFAULT 1.0;

ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS detected_at TIMESTAMPTZ DEFAULT NOW();

ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '12 hours');

ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS post_mortem_status VARCHAR(20) DEFAULT 'PENDING';

ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS actual_outcome VARCHAR(20);

ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS profit_loss_percentage DECIMAL(10,4);

ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS pattern_held BOOLEAN;

ALTER TABLE other_scanners.wyckoff_signals 
ADD COLUMN IF NOT EXISTS post_mortem_notes TEXT;

-- Verify the table structure
\d other_scanners.wyckoff_signals
