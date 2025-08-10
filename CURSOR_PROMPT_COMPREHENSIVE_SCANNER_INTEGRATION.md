# CURSOR PROMPT: Comprehensive Scanner Integration & Database Enhancement

## 🎯 TASK OVERVIEW
Implement a comprehensive database enhancement to support ALL scanner types with proper field indexing, while fixing any labeling issues. This is a one-time implementation to future-proof the database.

## 📊 CURRENT SITUATION
- BB Scanner: Working with enhanced market context tables ✅
- ICT Scanner: Has 9,466+ trades but unique fields buried in JSON
- Other Scanners: Need proper field indexing for performance
- All trades correctly stored in single `trade_opportunities` table
- Need to add scanner-specific columns for better querying

## 🔧 REQUIRED CHANGES

### STEP 1: Add All Scanner-Specific Fields to Database

Create file: `add_all_scanner_fields.sql`

```sql
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
```

### STEP 2: Create Analysis Views

Create file: `create_analysis_views.sql`

```sql
-- View to see ALL trades with their scanner type
CREATE OR REPLACE VIEW all_trades_with_scanner AS
SELECT 
    t.*,
    s.scanner_type,
    s.scan_timestamp,
    s.version as scanner_version
FROM trade_opportunities t
JOIN scans s ON t.scan_id = s.id;

-- View for BB trades only
CREATE OR REPLACE VIEW bb_trades AS
SELECT t.*
FROM trade_opportunities t
JOIN scans s ON t.scan_id = s.id
WHERE s.scanner_type = 'bb_scanner';

-- View for ICT trades only  
CREATE OR REPLACE VIEW ict_trades AS
SELECT t.*
FROM trade_opportunities t
JOIN scans s ON t.scan_id = s.id
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
FROM scans s
LEFT JOIN trade_opportunities t ON s.id = t.scan_id
GROUP BY s.scanner_type, DATE(s.scan_timestamp)
ORDER BY scan_date DESC, scanner_type;
```

### STEP 3: Update trade_logger.py

Modify the `log_trade_opportunity` method in `trade_logger.py`:

```python
def log_trade_opportunity(self, scan_id, trade_data):
    """Enhanced to handle all scanner-specific fields"""
    
    # Define scanner-specific fields that get their own columns
    scanner_specific_columns = {
        # ICT fields
        'gap_high', 'gap_low', 'gap_size_pct', 'gap_type',
        'swing_high', 'swing_low', 'order_block_high', 'order_block_low',
        'breaker_block_high', 'breaker_block_low',
        'fvg_high', 'fvg_low', 'liquidity_sweep_level',
        'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786',
        'equilibrium_level',
        # Volume fields
        'volume_surge_multiplier', 'relative_volume', 'vwap', 'vwap_deviation',
        'volume_profile_poc',
        # Pattern fields
        'pattern_type', 'pattern_target', 'pattern_reliability',
        # S/R fields
        'major_resistance_1', 'major_support_1', 'pivot_point'
    }
    
    # Separate scanner-specific fields from general data
    column_values = {}
    json_data = {}
    
    for key, value in trade_data.items():
        if key in scanner_specific_columns and value is not None:
            column_values[key] = value
        elif key not in ['symbol', 'exchange', 'probability', 'entry_price', 
                         'stop_loss', 'target_1', 'target_2', 'target_3']:
            json_data[key] = value
    
    # Build dynamic INSERT query
    columns = ['scan_id', 'symbol', 'exchange', 'probability', 
              'entry_price', 'stop_loss', 'target_1']
    values = [scan_id, trade_data.get('symbol'), trade_data.get('exchange'),
             trade_data.get('probability', 0), trade_data.get('entry_price'),
             trade_data.get('stop_loss'), trade_data.get('target_1')]
    
    # Add scanner-specific columns
    for col, val in column_values.items():
        columns.append(col)
        values.append(val)
    
    # Add JSON data
    columns.append('scanner_specific_data')
    values.append(json.dumps(json_data, default=str))
    
    # Execute INSERT
    placeholders = ','.join(['%s'] * len(values))
    query = f"INSERT INTO trade_opportunities ({','.join(columns)}) VALUES ({placeholders})"
    
    self.cursor.execute(query, values)
```

### STEP 4: Update Enhanced Export Script

Modify `enhanced_export.py` to properly label sheets:

```python
def _export_trades(self):
    """Export trade opportunities with parsed scanner_specific_data"""
    try:
        # Get all trades with scanner type
        self.logger.cursor.execute("""
            SELECT t.*, s.scanner_type
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            ORDER BY t.id DESC
        """)
        
        trades = self.logger.cursor.fetchall()
        
        if not trades:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Parse scanner_specific_data
        parsed_data = []
        for _, row in df.iterrows():
            trade_data = dict(row)
            
            # Parse JSON data
            if trade_data.get('scanner_specific_data'):
                try:
                    if isinstance(trade_data['scanner_specific_data'], str):
                        extra_data = json.loads(trade_data['scanner_specific_data'])
                    else:
                        extra_data = trade_data['scanner_specific_data']
                    
                    # Add extra data to trade_data
                    for key, value in extra_data.items():
                        trade_data[f'extra_{key}'] = value
                        
                except Exception as e:
                    print(f"Warning: Could not parse scanner_specific_data: {e}")
            
            parsed_data.append(trade_data)
        
        return pd.DataFrame(parsed_data)
        
    except Exception as e:
        print(f"Error exporting trades: {e}")
        return pd.DataFrame()

def _create_summary_sheet(self, writer, trades_df, regime_df, overview_df, metadata_df):
    """Create a summary sheet with key statistics"""
    
    # Group trades by scanner type
    scanner_stats = trades_df.groupby('scanner_type').agg({
        'id': 'count',
        'probability': ['mean', 'count']
    }).round(2)
    
    summary_data = {
        'Metric': [
            'Total Trades (All Scanners)',
            'BB Scanner Trades',
            'ICT Scanner Trades',
            'Other Scanner Trades',
            'High Probability Trades (>70%)',
            'Market Regime Records',
            'Market Overview Records',
            'Market Metadata Records',
            'Latest Market Regime',
            'Latest Fear & Greed Index',
            'Latest BTC Dominance',
            'Export Timestamp'
        ],
        'Value': [
            len(trades_df),
            len(trades_df[trades_df['scanner_type'] == 'bb_scanner']),
            len(trades_df[trades_df['scanner_type'].str.contains('ict', na=False)]),
            len(trades_df[~trades_df['scanner_type'].isin(['bb_scanner']) & 
                         ~trades_df['scanner_type'].str.contains('ict', na=False)]),
            len(trades_df[trades_df['probability'] > 70]) if len(trades_df) > 0 else 0,
            len(regime_df),
            len(overview_df),
            len(metadata_df),
            regime_df['regime_type'].iloc[0] if len(regime_df) > 0 else 'N/A',
            regime_df['fear_greed_index'].iloc[0] if len(regime_df) > 0 else 'N/A',
            f"{regime_df['btc_dominance'].iloc[0]}%" if len(regime_df) > 0 else 'N/A',
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
```

### STEP 5: Create Execution Script

Create file: `implement_scanner_integration.py`

```python
#!/usr/bin/env python3
"""
Execute all scanner integration changes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def run_sql_file(filename):
    """Execute SQL file"""
    try:
        with open(filename, 'r') as f:
            sql_content = f.read()
        
        logger = TradeLogger()
        
        # Split by semicolon and execute each statement
        statements = sql_content.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement:
                logger.cursor.execute(statement)
                print(f"✅ Executed: {statement[:50]}...")
        
        logger.connection.commit()
        logger.close()
        print(f"✅ Successfully executed {filename}")
        
    except Exception as e:
        print(f"❌ Error executing {filename}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 Implementing Comprehensive Scanner Integration")
    print("=" * 60)
    
    # Step 1: Add all scanner fields
    print("\n📊 Step 1: Adding scanner-specific fields...")
    run_sql_file('add_all_scanner_fields.sql')
    
    # Step 2: Create analysis views
    print("\n📊 Step 2: Creating analysis views...")
    run_sql_file('create_analysis_views.sql')
    
    print("\n✅ Scanner integration complete!")
    print("\n📋 Next steps:")
    print("1. Update trade_logger.py with enhanced log_trade_opportunity method")
    print("2. Update enhanced_export.py with proper labeling")
    print("3. Test with: SELECT * FROM all_trades_with_scanner LIMIT 5;")

if __name__ == "__main__":
    main()
```

## 🎯 IMPLEMENTATION ORDER

1. **Create all SQL files** (`add_all_scanner_fields.sql`, `create_analysis_views.sql`)
2. **Update `trade_logger.py`** with enhanced method
3. **Update `enhanced_export.py`** with proper labeling
4. **Create execution script** (`implement_scanner_integration.py`)
5. **Run the implementation** on Digital Ocean
6. **Test the results**

## ✅ EXPECTED BENEFITS

- **One-time schema change** - Never need to modify again
- **All trades in one table** - Easy cross-scanner analysis  
- **Proper labeling** - Clear which scanner found what
- **Indexed fields** - Fast queries on gaps, fibs, patterns
- **Future-proof** - Ready for any new scanner
- **Cross-scanner analysis** - Find confluence between BB and ICT signals

## 🔍 TESTING

After implementation, verify with:

```sql
-- Check all scanner types
SELECT DISTINCT scanner_type, COUNT(*) as trade_count 
FROM all_trades_with_scanner 
GROUP BY scanner_type;

-- Test ICT-specific queries
SELECT symbol, gap_size_pct, fib_618 
FROM ict_trades 
WHERE gap_size_pct > 2 
LIMIT 10;

-- Test cross-scanner confluence
SELECT * FROM multi_scanner_signals 
WHERE signal_confluence = 'Strong Confluence' 
LIMIT 5;
```

This comprehensive approach handles everything in one go and future-proofs your database! 🚀
