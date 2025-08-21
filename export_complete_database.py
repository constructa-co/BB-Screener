#!/usr/bin/env python3
"""
Complete Database Export - ICT + Elliott Wave Data
Exports all data from both other_scanners_trades and elliott_wave_signals tables.
"""

import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json

def connect_to_database():
    """Connect to the database."""
    try:
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            print("❌ DATABASE_URL not found in environment")
            return None
            
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def export_ict_data(conn):
    """Export ICT scanner data from other_scanners_trades."""
    try:
        print("📊 Querying ICT scanner data from other_scanners_trades...")
        
        query = """
            SELECT 
                id,
                scanner_name,
                scanner_version,
                timeframe,
                symbol,
                side,
                entry_price,
                quantity,
                stop_loss,
                take_profit,
                exit_price,
                pnl,
                status,
                created_at,
                updated_at,
                market_conditions,
                technical_indicators,
                scanner_signals,
                feature_vector,
                execution_metadata
            FROM other_scanners.other_scanners_trades
            ORDER BY created_at DESC
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        
        print(f"📈 Found {len(rows)} ICT scanner trades")
        return pd.DataFrame(rows) if rows else pd.DataFrame()
        
    except Exception as e:
        print(f"❌ ICT data export failed: {e}")
        return pd.DataFrame()

def export_elliott_wave_data(conn):
    """Export Elliott Wave data from elliott_wave_signals."""
    try:
        print("🌊 Querying Elliott Wave data from elliott_wave_signals...")
        
        query = """
            SELECT 
                id,
                scanner_name,
                scanner_version,
                scan_id,
                symbol,
                exchange,
                timeframe,
                direction,
                entry_price,
                stop_loss,
                tp1,
                tp2,
                tp3,
                risk_reward,
                pattern_type,
                current_wave,
                wave_degree,
                pattern_quality,
                confidence_score,
                invalidation_level,
                wave_analysis,
                fibonacci_levels,
                pattern_metrics,
                analysis_ts,
                created_at
            FROM other_scanners.elliott_wave_signals
            ORDER BY created_at DESC
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        
        print(f"🌊 Found {len(rows)} Elliott Wave patterns")
        return pd.DataFrame(rows) if rows else pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Elliott Wave data export failed: {e}")
        return pd.DataFrame()

def get_database_summary(conn):
    """Get summary statistics for both tables."""
    try:
        print("📊 Getting database summary...")
        
        summary_queries = [
            ("other_scanners_trades", "SELECT COUNT(*) as count FROM other_scanners.other_scanners_trades"),
            ("elliott_wave_signals", "SELECT COUNT(*) as count FROM other_scanners.elliott_wave_signals"),
            ("ict_scanners", "SELECT COUNT(*) as count FROM other_scanners.other_scanners_trades WHERE scanner_name LIKE 'ict%'"),
            ("recent_ict", "SELECT COUNT(*) as count FROM other_scanners.other_scanners_trades WHERE created_at > NOW() - INTERVAL '24 hours'"),
            ("recent_elliott", "SELECT COUNT(*) as count FROM other_scanners.elliott_wave_signals WHERE created_at > NOW() - INTERVAL '24 hours'")
        ]
        
        summary = {}
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for name, query in summary_queries:
                cursor.execute(query)
                result = cursor.fetchone()
                summary[name] = result['count'] if result else 0
        
        return summary
        
    except Exception as e:
        print(f"❌ Summary query failed: {e}")
        return {}

def export_to_excel():
    """Export complete database to Excel."""
    try:
        print("🔍 Connecting to database...")
        conn = connect_to_database()
        if not conn:
            return False
        
        # Get summary first
        summary = get_database_summary(conn)
        print(f"\n📊 Database Summary:")
        print(f"   ICT Scanner Trades: {summary.get('ict_scanners', 0)}")
        print(f"   Elliott Wave Patterns: {summary.get('elliott_wave_signals', 0)}")
        print(f"   Recent ICT (24h): {summary.get('recent_ict', 0)}")
        print(f"   Recent Elliott (24h): {summary.get('recent_elliott', 0)}")
        print(f"   Total Trades: {summary.get('other_scanners_trades', 0)}")
        
        # Export data
        ict_df = export_ict_data(conn)
        elliott_df = export_elliott_wave_data(conn)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_database_export_{timestamp}.xlsx"
        
        print(f"\n💾 Exporting to Excel: {filename}")
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Database_Summary', index=False)
            
            # ICT data sheet
            if not ict_df.empty:
                ict_df.to_excel(writer, sheet_name='ICT_Scanner_Data', index=False)
                print(f"   ✅ ICT data: {len(ict_df)} rows")
            else:
                print("   ⚠️ No ICT data found")
            
            # Elliott Wave data sheet
            if not elliott_df.empty:
                elliott_df.to_excel(writer, sheet_name='Elliott_Wave_Data', index=False)
                print(f"   ✅ Elliott Wave data: {len(elliott_df)} rows")
            else:
                print("   ⚠️ No Elliott Wave data found")
            
            # Recent data sheets
            if not ict_df.empty:
                recent_ict = ict_df[ict_df['created_at'] >= pd.Timestamp.now() - pd.Timedelta(days=1)]
                if not recent_ict.empty:
                    recent_ict.to_excel(writer, sheet_name='Recent_ICT_24h', index=False)
                    print(f"   ✅ Recent ICT: {len(recent_ict)} rows")
            
            if not elliott_df.empty:
                recent_elliott = elliott_df[elliott_df['created_at'] >= pd.Timestamp.now() - pd.Timedelta(days=1)]
                if not recent_elliott.empty:
                    recent_elliott.to_excel(writer, sheet_name='Recent_Elliott_24h', index=False)
                    print(f"   ✅ Recent Elliott: {len(recent_elliott)} rows")
        
        print(f"\n✅ Complete export completed: {filename}")
        
        # Show sample data
        if not ict_df.empty:
            print(f"\n🔍 Sample ICT Data (last 3):")
            sample_ict = ict_df[['scanner_name', 'symbol', 'timeframe', 'created_at']].head(3)
            print(sample_ict.to_string(index=False))
        
        if not elliott_df.empty:
            print(f"\n🌊 Sample Elliott Wave Data (last 3):")
            sample_elliott = elliott_df[['symbol', 'timeframe', 'current_wave', 'pattern_type', 'pattern_quality', 'created_at']].head(3)
            print(sample_elliott.to_string(index=False))
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Complete Database Export...")
    success = export_to_excel()
    if success:
        print("✅ Export completed successfully!")
    else:
        print("❌ Export failed!")
        sys.exit(1)
