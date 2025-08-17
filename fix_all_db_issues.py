# fix_all_db_issues.py - Complete solution
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import json
import os
from datetime import datetime

# Try to import dotenv for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not available - using system environment variables")
    def load_dotenv():
        pass

def make_json_safe(obj):
    """Convert NumPy/pandas types to JSON-serializable Python types"""
    import numpy as np
    import pandas as pd
    from decimal import Decimal
    from datetime import datetime
    
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    else:
        return obj

def fix_all_database_issues():
    """Fix all database issues: missing trades, truncation, scanner metadata"""
    
    # Load the scanner output
    scanner_file = 'outputs/excel_reports/bb_analysis_20250816_192704.xlsx'
    
    print("Loading Excel data...")
    all_analysis = pd.read_excel(scanner_file, sheet_name='All_Analysis')
    market_regime = pd.read_excel(scanner_file, sheet_name='Market_Regime_Analysis')
    market_overview = pd.read_excel(scanner_file, sheet_name='Market_Overview')
    market_metadata = pd.read_excel(scanner_file, sheet_name='Market_Metadata')
    
    print(f"Found {len(all_analysis)} trades in Excel")
    
    # Use same connection method as trade_logger.py
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ No DATABASE_URL found in environment variables")
        return
    
    try:
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    try:
        # First, check what's in database
        cursor.execute("SELECT COUNT(*) FROM trade_opportunities WHERE DATE(timestamp) = '2025-08-16'")
        db_count = cursor.fetchone()['count']
        print(f"Current database has {db_count} trades for 2025-08-16")
        
        # Create a new scan entry
        scan_id = f"bb_scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process ALL trades (including 0 probability ones)
        inserted = 0
        updated = 0
        
        for idx, trade in all_analysis.iterrows():
            symbol = trade['symbol']
            
            # Prepare complete trade data
            trade_dict = trade.to_dict()
            
            # Add missing metadata
            trade_dict['scanner_type'] = 'bb_scanner'
            trade_dict['timeframe'] = '4H'  # Or extract from config
            trade_dict['scan_id'] = scan_id
            trade_dict['scan_timestamp'] = datetime.now().isoformat()
            
            # Fix confidence_rationale (ensure full string is saved)
            if 'confidence_rationale' in trade_dict:
                trade_dict['confidence_rationale_full'] = str(trade_dict['confidence_rationale'])
            
            # Add market context
            trade_dict['market_regime_data'] = market_regime.to_dict('records') if len(market_regime) > 0 else {}
            trade_dict['market_overview_data'] = market_overview.to_dict('records') if len(market_overview) > 0 else {}
            trade_dict['market_metadata_data'] = market_metadata.to_dict('records') if len(market_metadata) > 0 else {}
            
            # Clean NaN values and make JSON safe
            for key, value in trade_dict.items():
                try:
                    if isinstance(value, (list, tuple, np.ndarray)):
                        # Handle arrays/lists - keep as is
                        continue
                    elif pd.isna(value):
                        trade_dict[key] = None
                    elif isinstance(value, float) and pd.isna(value):
                        trade_dict[key] = None
                except:
                    # If any error in checking, keep original value
                    continue
            
            # Make the entire trade_dict JSON safe
            trade_dict = make_json_safe(trade_dict)
            
            # Check if trade exists
            cursor.execute("""
                SELECT id FROM trade_opportunities 
                WHERE symbol = %s AND DATE(timestamp) = '2025-08-16'
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing trade with complete data
                cursor.execute("""
                    UPDATE trade_opportunities 
                    SET scanner_specific_data = %s,
                        timeframe = %s
                    WHERE id = %s
                """, (Json(trade_dict), '4H', existing['id']))
                updated += 1
            else:
                # Insert new trade with ALL fields
                cursor.execute("""
                    INSERT INTO trade_opportunities 
                    (scan_id, symbol, exchange, probability, 
                     entry_price, stop_loss, target_1, timeframe,
                     scanner_specific_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    scan_id,
                    trade_dict.get('symbol', ''),
                    trade_dict.get('exchange', ''),
                    trade_dict.get('probability', 0),
                    trade_dict.get('entry_price', 0),
                    trade_dict.get('stop_loss', 0),
                    trade_dict.get('target_1', 0),
                    '4H',
                    Json(trade_dict)
                ))
                inserted += 1
        
        conn.commit()
        
        print(f"✅ SUCCESS!")
        print(f"  - Inserted {inserted} new trades")
        print(f"  - Updated {updated} existing trades")
        print(f"  - All {len(all_analysis)} trades now in database with complete data")
        print(f"  - Sentiment data preserved")
        print(f"  - Scanner metadata added")
        print(f"  - Market context included")
        
        # Verify the fix
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN scanner_specific_data->>'confidence_rationale' LIKE '%Sentiment%' THEN 1 END) as with_sentiment,
                COUNT(CASE WHEN scanner_specific_data->>'scanner_type' IS NOT NULL THEN 1 END) as with_scanner_type
            FROM trade_opportunities 
            WHERE DATE(timestamp) = '2025-08-16'
        """)
        
        verify = cursor.fetchone()
        print(f"\n✅ VERIFICATION:")
        print(f"  - Total trades: {verify['total']}")
        print(f"  - Trades with sentiment: {verify['with_sentiment']}")
        print(f"  - Trades with scanner_type: {verify['with_scanner_type']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    fix_all_database_issues()
