#!/usr/bin/env python3
"""
Export BB Scanner Database Data
"""

import pandas as pd
import json
from datetime import datetime
from trade_logger import TradeLogger

def export_bb_database():
    """Export all BB scanner data from database"""
    logger = TradeLogger()
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        # Get all BB scanner trades
        logger.cursor.execute("""
            SELECT t.*, s.scan_type
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type = 'bb_scanner'
            ORDER BY t.id DESC
        """)
        
        trades = logger.cursor.fetchall()
        print(f"📊 Found {len(trades)} BB scanner trades in database")
        
        if len(trades) == 0:
            print("❌ No BB scanner trades found")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Parse scanner_specific_data
        print("🔍 Parsing scanner_specific_data...")
        parsed_data = []
        
        for idx, row in df.iterrows():
            trade_dict = dict(row)
            
            # Parse JSON data
            if trade_dict.get('scanner_specific_data'):
                try:
                    if isinstance(trade_dict['scanner_specific_data'], str):
                        extra_data = json.loads(trade_dict['scanner_specific_data'])
                    else:
                        extra_data = trade_dict['scanner_specific_data']
                    
                    # Add all extra fields to the trade dict
                    for key, value in extra_data.items():
                        trade_dict[f'extra_{key}'] = value
                        
                except Exception as e:
                    print(f"⚠️ Error parsing data for trade {idx}: {e}")
            
            parsed_data.append(trade_dict)
        
        # Create final DataFrame
        final_df = pd.DataFrame(parsed_data)
        
        # Save to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bb_database_export_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main trades sheet
            final_df.to_excel(writer, sheet_name='All_BB_Trades', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total BB Scanner Trades',
                    'High Probability Trades (>70%)',
                    'Medium Probability Trades (50-70%)',
                    'Low Probability Trades (<50%)',
                    'Average Probability',
                    'Most Recent Trade',
                    'Database Export Time'
                ],
                'Value': [
                    len(final_df),
                    len(final_df[final_df['probability'] > 70]),
                    len(final_df[(final_df['probability'] >= 50) & (final_df['probability'] <= 70)]),
                    len(final_df[final_df['probability'] < 50]),
                    final_df['probability'].mean(),
                    'N/A',
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # High probability trades sheet
            high_prob_df = final_df[final_df['probability'] > 70].copy()
            if len(high_prob_df) > 0:
                high_prob_df.to_excel(writer, sheet_name='High_Probability_Trades', index=False)
        
        print(f"✅ Database export saved to: {filename}")
        print(f"📊 Export contains {len(final_df)} trades with {len(final_df.columns)} columns")
        
        # Show sample of high probability trades
        high_prob_trades = final_df[final_df['probability'] > 70]
        if len(high_prob_trades) > 0:
            print(f"\n🎯 Sample High Probability Trades:")
            for _, trade in high_prob_trades.head(5).iterrows():
                print(f"   • {trade['symbol']}: {trade['probability']}% probability")
        
        return filename
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        logger.close()

if __name__ == "__main__":
    export_bb_database()
