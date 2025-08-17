#!/usr/bin/env python3
"""
Complete Database Export Script
Exports ALL 122 fields from trade_opportunities table including JSONB data
"""

from trade_logger import TradeLogger
import pandas as pd
import json
from datetime import datetime

def export_complete_database():
    """Export complete database with all 122 fields to Excel"""
    
    print("🔍 Connecting to database...")
    logger = TradeLogger()
    
    print("📊 Fetching all trade opportunities...")
    logger.cursor.execute("""
        SELECT *, scanner_specific_data 
        FROM trade_opportunities 
        ORDER BY timestamp DESC
    """)
    
    print("🔄 Processing trades and expanding JSONB data...")
    trades = []
    for row in logger.cursor.fetchall():
        # Convert row to dict
        trade = dict(row)
        
        # Expand JSONB data if it exists
        if trade['scanner_specific_data']:
            try:
                # JSONB data is already a dict from psycopg2
                jsonb_data = trade['scanner_specific_data']
                if isinstance(jsonb_data, str):
                    # If it's a string, parse it
                    jsonb_data = json.loads(jsonb_data)
                
                # Merge all JSONB fields into the main trade dict
                trade.update(jsonb_data)
                print(f"✅ Expanded {len(jsonb_data)} JSONB fields for trade {trade.get('symbol', 'Unknown')}")
            except Exception as e:
                print(f"❌ Error processing JSONB for trade {trade.get('symbol', 'Unknown')}: {e}")
        else:
            print(f"⚠️  No JSONB data for trade {trade.get('symbol', 'Unknown')}")
        
        trades.append(trade)
    
    if not trades:
        print("❌ No trades found in database")
        return
    
    print(f"📋 Creating DataFrame with {len(trades)} trades...")
    df = pd.DataFrame(trades)
    
    # Show field count
    total_fields = len(df.columns)
    print(f"📊 Total fields in export: {total_fields}")
    print(f"📋 Field names: {list(df.columns)}")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'complete_database_export_{timestamp}.xlsx'
    
    print(f"💾 Exporting to {filename}...")
    # Use openpyxl engine to avoid xlsxwriter issues
    df.to_excel(filename, index=False, engine='openpyxl')
    
    print(f"✅ Complete database exported to {filename}")
    print(f"📊 Summary:")
    print(f"   • Trades exported: {len(trades)}")
    print(f"   • Total fields: {total_fields}")
    print(f"   • File size: {filename}")
    
    # Show sample of the data
    print(f"\n📋 Sample data (first 3 trades, first 10 fields):")
    print(df.head(3)[df.columns[:10]].to_string())
    
    logger.close()

if __name__ == "__main__":
    export_complete_database()
