#!/usr/bin/env python3
"""
Enhanced Database Export Script
Connects directly to PostgreSQL and exports ALL JSONB fields expanded
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv

def export_full_database():
    """Export database with ALL JSONB fields expanded"""
    
    print("🔍 Connecting to PostgreSQL database...")
    load_dotenv()
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in environment variables")
        return None
    
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        print("✅ Connected to database successfully")
        
        # Get all trades from today
        cursor.execute("""
            SELECT * FROM trade_opportunities
            WHERE DATE(timestamp) = CURRENT_DATE
            ORDER BY timestamp DESC
        """)
        
        trades = cursor.fetchall()
        print(f"📊 Found {len(trades)} trades from today")
        
        # Expand JSONB for each trade
        print("📈 Expanding JSONB data...")
        expanded_trades = []
        
        for i, trade in enumerate(trades):
            trade_dict = dict(trade)
            
            # Expand scanner_specific_data JSONB
            if trade_dict.get('scanner_specific_data'):
                jsonb_data = trade_dict['scanner_specific_data']
                
                # Remove the JSONB column and merge its contents
                del trade_dict['scanner_specific_data']
                trade_dict.update(jsonb_data)
                
                print(f"✅ Trade {i+1}: Expanded {len(jsonb_data)} JSONB fields")
            else:
                print(f"⚠️  Trade {i+1}: No JSONB data found")
            
            expanded_trades.append(trade_dict)
        
        # Create DataFrame with ALL fields
        df = pd.DataFrame(expanded_trades)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'FULL_database_export_{timestamp}.xlsx'
        
        print(f"💾 Saving expanded data to {filename}...")
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main trades sheet
            df.to_excel(writer, sheet_name='All_Trades', index=False)
            
            # High probability trades
            if 'probability' in df.columns:
                df_high = df[df['probability'] > 70].sort_values('probability', ascending=False)
                df_high.to_excel(writer, sheet_name='High_Probability', index=False)
            
            # Summary statistics
            summary_data = {
                'Metric': [
                    'Total Trades',
                    'Total Fields',
                    'High Probability Trades (>70%)',
                    'Average Probability',
                    'Date Range',
                    'Export Time'
                ],
                'Value': [
                    len(df),
                    len(df.columns),
                    len(df_high) if 'probability' in df.columns else 0,
                    f"{df['probability'].mean():.1f}%" if 'probability' in df.columns else 'N/A',
                    f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A',
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Field list sheet
            field_list = pd.DataFrame({
                'Field_Number': range(1, len(df.columns) + 1),
                'Field_Name': df.columns.tolist(),
                'Data_Type': [str(df[col].dtype) for col in df.columns],
                'Non_Null_Count': [df[col].count() for col in df.columns]
            })
            field_list.to_excel(writer, sheet_name='Field_List', index=False)
        
        print(f"✅ SUCCESS! Exported {len(df)} trades with {len(df.columns)} columns")
        print(f"📁 File saved as: {filename}")
        print(f"📊 High probability trades: {len(df_high) if 'probability' in df.columns else 0}")
        
        # Show sample of expanded fields
        print("\n🔍 Sample of expanded fields:")
        sample_fields = df.columns[:20].tolist()
        for i, field in enumerate(sample_fields, 1):
            print(f"  {i:2d}. {field}")
        
        if len(df.columns) > 20:
            print(f"  ... and {len(df.columns) - 20} more fields")
        
        cursor.close()
        conn.close()
        
        return filename
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    export_full_database()
