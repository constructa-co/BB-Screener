#!/usr/bin/env python3
"""
Quick database export for verification
"""
from trade_logger import TradeLogger
import pandas as pd
from datetime import datetime

def export_database():
    """Export current database state"""
    logger = TradeLogger()
    
    # Export trade_opportunities
    logger.cursor.execute("""
        SELECT * FROM trade_opportunities 
        WHERE scanner_specific_data IS NOT NULL 
        ORDER BY timestamp DESC 
        LIMIT 50
    """)
    
    trades = logger.cursor.fetchall()
    trades_df = pd.DataFrame(trades)
    
    # Export market_data
    logger.cursor.execute("SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 100")
    market_data = logger.cursor.fetchall()
    market_df = pd.DataFrame(market_data)
    
    # Save to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"database_export_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        trades_df.to_excel(writer, sheet_name='Trade_Opportunities', index=False)
        market_df.to_excel(writer, sheet_name='Market_Data', index=False)
    
    print(f"✅ Database exported to {filename}")
    print(f"📊 Trade opportunities: {len(trades_df)} rows")
    print(f"📊 Market data: {len(market_df)} rows")
    
    # Check field counts
    if not trades_df.empty and 'scanner_specific_data' in trades_df.columns:
        import json
        field_counts = []
        for _, row in trades_df.head(10).iterrows():
            if row['scanner_specific_data']:
                data = json.loads(row['scanner_specific_data']) if isinstance(row['scanner_specific_data'], str) else row['scanner_specific_data']
                field_counts.append(len(data))
        
        print(f"🔍 Field counts in latest trades: {field_counts}")
    
    logger.close()

if __name__ == "__main__":
    export_database()
