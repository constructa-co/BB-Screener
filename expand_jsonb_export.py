#!/usr/bin/env python3
"""
Expand JSONB Export Script
Expands the scanner_specific_data JSONB column to show ALL 2,172 fields in Excel
"""

import pandas as pd
import json
import os
from datetime import datetime

def expand_jsonb_export(input_file='database_export_20250817.xlsx', output_file=None):
    """Expand JSONB data from database export to show all fields"""
    
    print("🔍 Loading database export...")
    df = pd.read_excel(input_file)
    print(f"✅ Loaded {len(df)} trades with {len(df.columns)} base columns")
    
    # Expand the scanner_specific_data JSONB column
    print("📊 Expanding JSONB data...")
    expanded_data = []
    
    for idx, row in df.iterrows():
        trade = row.to_dict()
        
        # Parse the JSONB data
        if 'scanner_specific_data' in trade and trade['scanner_specific_data']:
            try:
                # Handle both string and dict formats
                if isinstance(trade['scanner_specific_data'], str):
                    jsonb_data = json.loads(trade['scanner_specific_data'])
                else:
                    jsonb_data = trade['scanner_specific_data']
                
                # Remove the JSONB column and merge its contents
                del trade['scanner_specific_data']
                trade.update(jsonb_data)
                
                print(f"✅ Trade {idx+1}: Expanded {len(jsonb_data)} JSONB fields")
                
            except Exception as e:
                print(f"⚠️  Trade {idx+1}: Error expanding JSONB - {e}")
                # Keep original data if expansion fails
        else:
            print(f"⚠️  Trade {idx+1}: No JSONB data found")
        
        expanded_data.append(trade)
    
    # Create new DataFrame with ALL fields
    df_expanded = pd.DataFrame(expanded_data)
    
    # Generate output filename if not provided
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'database_export_FULL_FIELDS_{timestamp}.xlsx'
    
    # Save to new Excel with ALL fields visible
    print(f"💾 Saving expanded data to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main trades sheet with ALL fields
        df_expanded.to_excel(writer, sheet_name='All_Trades', index=False)
        
        # High probability trades sheet
        if 'probability' in df_expanded.columns:
            df_high = df_expanded[df_expanded['probability'] > 70].sort_values('probability', ascending=False)
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
                len(df_expanded),
                len(df_expanded.columns),
                len(df_high) if 'probability' in df_expanded.columns else 0,
                f"{df_expanded['probability'].mean():.1f}%" if 'probability' in df_expanded.columns else 'N/A',
                f"{df_expanded['timestamp'].min()} to {df_expanded['timestamp'].max()}" if 'timestamp' in df_expanded.columns else 'N/A',
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Field list sheet
        field_list = pd.DataFrame({
            'Field_Number': range(1, len(df_expanded.columns) + 1),
            'Field_Name': df_expanded.columns.tolist(),
            'Data_Type': [str(df_expanded[col].dtype) for col in df_expanded.columns],
            'Non_Null_Count': [df_expanded[col].count() for col in df_expanded.columns]
        })
        field_list.to_excel(writer, sheet_name='Field_List', index=False)
    
    print(f"✅ SUCCESS! Exported {len(df_expanded)} trades with {len(df_expanded.columns)} columns")
    print(f"📁 File saved as: {output_file}")
    print(f"📊 High probability trades: {len(df_high) if 'probability' in df_expanded.columns else 0}")
    
    # Show sample of expanded fields
    print("\n🔍 Sample of expanded fields:")
    sample_fields = df_expanded.columns[:20].tolist()
    for i, field in enumerate(sample_fields, 1):
        print(f"  {i:2d}. {field}")
    
    if len(df_expanded.columns) > 20:
        print(f"  ... and {len(df_expanded.columns) - 20} more fields")
    
    return output_file

if __name__ == "__main__":
    # Check if input file exists
    input_file = 'database_export_20250817.xlsx'
    if not os.path.exists(input_file):
        print(f"❌ Input file '{input_file}' not found!")
        print("Please make sure the database export file is in the current directory.")
    else:
        expand_jsonb_export(input_file)
