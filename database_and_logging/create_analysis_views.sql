-- View to see ALL trades with their scanner type
CREATE OR REPLACE VIEW all_trades_with_scanner AS
SELECT 
    t.*,
    s.scanner_type,
    s.scan_timestamp,
    s.version as scanner_version
FROM trade_opportunities t
JOIN scan_results s ON t.scan_id = s.id;

-- View for BB trades only
CREATE OR REPLACE VIEW bb_trades AS
SELECT t.*
FROM trade_opportunities t
JOIN scan_results s ON t.scan_id = s.id
WHERE s.scanner_type = 'bb_scanner';

-- View for ICT trades only  
CREATE OR REPLACE VIEW ict_trades AS
SELECT t.*
FROM trade_opportunities t
JOIN scan_results s ON t.scan_id = s.id
WHERE s.scanner_type LIKE 'ict_scanner%';

-- Cross-scanner confluence view
CREATE OR REPLACE VIEW multi_scanner_signals AS
SELECT 
    COALESCE(bb.symbol, ict.symbol) as symbol,
    bb.probability as bb_probability,
    bb.entry_price as bb_entry,
    ict.probability as ict_probability,
    ict.gap_high,
    ict.gap_low,
    ict.fib_618,
    CASE 
        WHEN bb.probability > 70 AND ict.probability > 70 THEN 'Strong Confluence'
        WHEN bb.probability > 70 OR ict.probability > 70 THEN 'Moderate Confluence'
        ELSE 'Low Confluence'
    END as signal_confluence
FROM bb_trades bb
FULL OUTER JOIN ict_trades ict 
    ON bb.symbol = ict.symbol 
    AND DATE(bb.created_at) = DATE(ict.created_at);

-- Scanner performance summary
CREATE OR REPLACE VIEW scanner_performance AS
SELECT 
    s.scanner_type,
    DATE(s.scan_timestamp) as scan_date,
    COUNT(t.id) as trade_count,
    AVG(t.probability) as avg_probability,
    COUNT(CASE WHEN t.probability > 70 THEN 1 END) as high_prob_trades,
    COUNT(CASE WHEN t.probability > 80 THEN 1 END) as very_high_prob_trades
FROM scan_results s
LEFT JOIN trade_opportunities t ON s.id = t.scan_id
GROUP BY s.scanner_type, DATE(s.scan_timestamp)
ORDER BY scan_date DESC, scanner_type;
