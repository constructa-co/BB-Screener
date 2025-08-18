#!/usr/bin/env python3
"""
Export ICT Scanner Results to Excel
Exports the latest ICT scanner trades with all JSONB fields expanded
"""

import os
import sys
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import json

def get_database_url():
    """Get the database URL for other scanners"""
    DATABASE_URL = os.getenv('OTHER_SCANNERS_DATABASE_URL')
    
    if not DATABASE_URL:
        # Fallback to main DATABASE_URL with schema
        main_db_url = os.getenv('DATABASE_URL')
        if main_db_url:
            if '?' in main_db_url:
                DATABASE_URL = main_db_url + '&options=-csearch_path=other_scanners'
            else:
                DATABASE_URL = main_db_url + '?options=-csearch_path=other_scanners'
        else:
            print("❌ No database URL found")
            return None
    
    return DATABASE_URL

def expand_jsonb_column(df, column_name):
    """Expand a JSONB column into separate columns"""
    if column_name not in df.columns:
        return df
    
    # Get all unique keys from the JSONB column
    all_keys = set()
    for item in df[column_name].dropna():
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    # Create new columns for each key
    for key in sorted(all_keys):
        new_column_name = f"{column_name}_{key}"
        df[new_column_name] = df[column_name].apply(
            lambda x: x.get(key) if isinstance(x, dict) else None
        )
    
    # Drop the original JSONB column
    df = df.drop(columns=[column_name])
    return df

def export_ict_results():
    """Export ICT scanner results to Excel"""
    DATABASE_URL = get_database_url()
    if not DATABASE_URL:
        return
    
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Set schema
        cur.execute("SET search_path TO other_scanners")
        
        # Get ICT scanner trades from the last 24 hours
        cur.execute("""
            SELECT 
                id,
                scanner_name,
                scanner_version,
                symbol,
                timeframe,
                side,
                entry_price,
                exit_price,
                quantity,
                stop_loss,
                take_profit,
                status,
                created_at,
                updated_at,
                technical_indicators,
                scanner_signals,
                market_conditions,
                feature_vector,
                execution_metadata
            FROM other_scanners_trades
            WHERE scanner_name = 'ict_4h_scanner'
            AND created_at > NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
        """)
        
        # Fetch all results
        rows = cur.fetchall()
        
        if not rows:
            print("❌ No ICT scanner trades found in the last 24 hours")
            return
        
        # Convert to DataFrame
        columns = [
            'id', 'scanner_name', 'scanner_version', 'symbol', 'timeframe', 'side',
            'entry_price', 'exit_price', 'quantity', 'stop_loss', 'take_profit',
            'status', 'created_at', 'updated_at', 'technical_indicators',
            'scanner_signals', 'market_conditions', 'feature_vector', 'execution_metadata'
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Expand JSONB columns
        print("📊 Expanding JSONB columns...")
        df = expand_jsonb_column(df, 'technical_indicators')
        df = expand_jsonb_column(df, 'scanner_signals')
        df = expand_jsonb_column(df, 'market_conditions')
        df = expand_jsonb_column(df, 'execution_metadata')
        
        # Handle feature_vector (array)
        if 'feature_vector' in df.columns:
            df['feature_vector_length'] = df['feature_vector'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            # Extract first few elements if they exist
            for i in range(5):
                df[f'feature_vector_{i}'] = df['feature_vector'].apply(
                    lambda x: x[i] if isinstance(x, list) and len(x) > i else None
                )
            df = df.drop(columns=['feature_vector'])
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ict_scanner_results_{timestamp}.xlsx"
        
        # Export to Excel
        print(f"📈 Exporting {len(df)} trades to {output_file}...")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='ICT_Scanner_Results', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Total Trades',
                    'BUY Trades',
                    'SELL Trades',
                    'Average Entry Price',
                    'Average Stop Loss',
                    'Average Take Profit',
                    'Average Quality Score',
                    'Average Risk/Reward',
                    'Most Common Symbol',
                    'Scan Time'
                ],
                'Value': [
                    len(df),
                    len(df[df['side'] == 'BUY']),
                    len(df[df['side'] == 'SELL']),
                    df['entry_price'].mean(),
                    df['stop_loss'].mean(),
                    df['take_profit'].mean(),
                    df.get('technical_indicators_final_quality', pd.Series()).mean(),
                    df.get('technical_indicators_risk_reward', pd.Series()).mean(),
                    df['symbol'].mode().iloc[0] if not df['symbol'].mode().empty else 'N/A',
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"✅ Successfully exported {len(df)} ICT scanner trades to {output_file}")
        print(f"📊 Total columns: {len(df.columns)}")
        print(f"📋 Summary: {len(df)} trades, {len(df[df['side'] == 'BUY'])} BUY, {len(df[df['side'] == 'SELL'])} SELL")
        
        # Show sample of the data
        print("\n📋 Sample of exported data:")
        print(df[['symbol', 'side', 'entry_price', 'stop_loss', 'take_profit', 'technical_indicators_final_quality']].head())
        
        cur.close()
        conn.close()
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error exporting ICT results: {e}")
        return None

if __name__ == "__main__":
    output_file = export_ict_results()
    if output_file:
        print(f"\n🎉 Export complete! File: {output_file}")
    else:
        print("\n❌ Export failed!")
