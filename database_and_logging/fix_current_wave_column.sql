-- Fix current_wave column length issue
-- Change from VARCHAR(8) to VARCHAR(32) to accommodate longer wave names

SET search_path TO other_scanners;

-- Alter the current_wave column to allow longer values
ALTER TABLE elliott_wave_signals 
ALTER COLUMN current_wave TYPE VARCHAR(32);

-- Verify the change
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns 
WHERE table_schema = 'other_scanners' 
AND table_name = 'elliott_wave_signals' 
AND column_name = 'current_wave';

-- Success message
SELECT '✅ current_wave column updated to VARCHAR(32)' as status;
