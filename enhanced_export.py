#!/usr/bin/env python3
"""
Enhanced Database Export with Secondary Sheet Data
Exports both trade data and market context data
"""

import pandas as pd
import json
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

class EnhancedDatabaseExport:
    def __init__(self):
        self.logger = TradeLogger()
        
    def export_complete_database(self):
        """Export complete database with all tables"""
        try:
            print("📊 Enhanced Database Export")
            print("="*50)
            
            # 1. Export trade opportunities
            print("🔍 Exporting trade opportunities...")
            trades_df = self._export_trades()
            print(f"✅ Exported {len(trades_df)} trades")
            
            # 2. Export market regime data
            print("🔍 Exporting market regime data...")
            regime_df = self._export_market_regime()
            print(f"✅ Exported {len(regime_df)} market regime records")
            
            # 3. Export market overview data
            print("🔍 Exporting market overview data...")
            overview_df = self._export_market_overview()
            print(f"✅ Exported {len(overview_df)} market overview records")
            
            # 4. Export market metadata
            print("🔍 Exporting market metadata...")
            metadata_df = self._export_market_metadata()
            print(f"✅ Exported {len(metadata_df)} market metadata records")
            
            # 5. Create Excel file with all sheets
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_bb_database_export_{timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main trade data
                trades_df.to_excel(writer, sheet_name='All_Trades', index=False)
                
                # Market context sheets
                if len(regime_df) > 0:
                    regime_df.to_excel(writer, sheet_name='Market_Regime_Analysis', index=False)
                
                if len(overview_df) > 0:
                    overview_df.to_excel(writer, sheet_name='Market_Overview', index=False)
                
                if len(metadata_df) > 0:
                    metadata_df.to_excel(writer, sheet_name='Market_Metadata', index=False)
                
                # Create summary sheet
                self._create_summary_sheet(writer, trades_df, regime_df, overview_df, metadata_df)
            
            print(f"✅ Enhanced export saved to: {filename}")
            print(f"📊 Export contains:")
            print(f"   • {len(trades_df)} trades in All_Trades")
            print(f"   • {len(regime_df)} records in Market_Regime_Analysis")
            print(f"   • {len(overview_df)} records in Market_Overview")
            print(f"   • {len(metadata_df)} records in Market_Metadata")
            
            return filename
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _export_trades(self):
        """Export trade opportunities with parsed scanner_specific_data"""
        try:
            # Get all trades with scanner type
            self.logger.cursor.execute("""
                SELECT t.*, s.scan_type
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
    
    def _export_market_regime(self):
        """Export market regime data"""
        try:
            self.logger.cursor.execute("""
                SELECT mr.*, s.scan_type
                FROM market_regime mr
                JOIN scan_results s ON mr.scan_id = s.id
                WHERE s.scan_type = 'bb_scanner'
                ORDER BY mr.id DESC
            """)
            
            regimes = self.logger.cursor.fetchall()
            
            if not regimes:
                return pd.DataFrame()
            
            df = pd.DataFrame(regimes)
            
            # Parse regime_data JSON
            parsed_data = []
            for _, row in df.iterrows():
                regime_data = dict(row)
                
                if regime_data.get('regime_data'):
                    try:
                        if isinstance(regime_data['regime_data'], str):
                            extra_data = json.loads(regime_data['regime_data'])
                        else:
                            extra_data = regime_data['regime_data']
                        
                        # Add extra data
                        for key, value in extra_data.items():
                            regime_data[f'regime_{key}'] = value
                            
                    except Exception as e:
                        print(f"Warning: Could not parse regime_data: {e}")
                
                parsed_data.append(regime_data)
            
            return pd.DataFrame(parsed_data)
            
        except Exception as e:
            print(f"Error exporting market regime: {e}")
            return pd.DataFrame()
    
    def _export_market_overview(self):
        """Export market overview data"""
        try:
            self.logger.cursor.execute("""
                SELECT mo.*, s.scan_type
                FROM market_overview mo
                JOIN scan_results s ON mo.scan_id = s.id
                WHERE s.scan_type = 'bb_scanner'
                ORDER BY mo.id DESC
            """)
            
            overviews = self.logger.cursor.fetchall()
            
            if not overviews:
                return pd.DataFrame()
            
            df = pd.DataFrame(overviews)
            
            # Parse overview_data JSON
            parsed_data = []
            for _, row in df.iterrows():
                overview_data = dict(row)
                
                if overview_data.get('overview_data'):
                    try:
                        if isinstance(overview_data['overview_data'], str):
                            extra_data = json.loads(overview_data['overview_data'])
                        else:
                            extra_data = overview_data['overview_data']
                        
                        # Add extra data
                        for key, value in extra_data.items():
                            overview_data[f'overview_{key}'] = value
                            
                    except Exception as e:
                        print(f"Warning: Could not parse overview_data: {e}")
                
                parsed_data.append(overview_data)
            
            return pd.DataFrame(parsed_data)
            
        except Exception as e:
            print(f"Error exporting market overview: {e}")
            return pd.DataFrame()
    
    def _export_market_metadata(self):
        """Export market metadata"""
        try:
            self.logger.cursor.execute("""
                SELECT mm.*, s.scan_type
                FROM market_metadata mm
                JOIN scan_results s ON mm.scan_id = s.id
                WHERE s.scan_type = 'bb_scanner'
                ORDER BY mm.id DESC
            """)
            
            metadata = self.logger.cursor.fetchall()
            
            if not metadata:
                return pd.DataFrame()
            
            df = pd.DataFrame(metadata)
            
            # Parse JSON fields
            parsed_data = []
            for _, row in df.iterrows():
                meta_data = dict(row)
                
                # Parse sector_performance
                if meta_data.get('sector_performance'):
                    try:
                        if isinstance(meta_data['sector_performance'], str):
                            sector_data = json.loads(meta_data['sector_performance'])
                        else:
                            sector_data = meta_data['sector_performance']
                        
                        for key, value in sector_data.items():
                            meta_data[f'sector_{key}'] = value
                            
                    except Exception as e:
                        print(f"Warning: Could not parse sector_performance: {e}")
                
                # Parse market_cap_tier_performance
                if meta_data.get('market_cap_tier_performance'):
                    try:
                        if isinstance(meta_data['market_cap_tier_performance'], str):
                            tier_data = json.loads(meta_data['market_cap_tier_performance'])
                        else:
                            tier_data = meta_data['market_cap_tier_performance']
                        
                        for key, value in tier_data.items():
                            meta_data[f'tier_{key}'] = value
                            
                    except Exception as e:
                        print(f"Warning: Could not parse market_cap_tier_performance: {e}")
                
                parsed_data.append(meta_data)
            
            return pd.DataFrame(parsed_data)
            
        except Exception as e:
            print(f"Error exporting market metadata: {e}")
            return pd.DataFrame()
    
    def _create_summary_sheet(self, writer, trades_df, regime_df, overview_df, metadata_df):
        """Create a summary sheet with key statistics"""
        
        # Group trades by scanner type
        scanner_stats = trades_df.groupby('scan_type').agg({
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
                len(trades_df[trades_df['scan_type'] == 'bb_scanner']),
                len(trades_df[trades_df['scan_type'].str.contains('ict', na=False)]),
                len(trades_df[~trades_df['scan_type'].isin(['bb_scanner']) & 
                             ~trades_df['scan_type'].str.contains('ict', na=False)]),
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
    
    def close(self):
        """Close database connection"""
        if self.logger.connection:
            self.logger.close()

if __name__ == "__main__":
    exporter = EnhancedDatabaseExport()
    filename = exporter.export_complete_database()
    exporter.close()
    
    if filename:
        print(f"\n🎉 Enhanced export completed: {filename}")
    else:
        print("\n❌ Enhanced export failed")
