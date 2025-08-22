#!/usr/bin/env python3
"""
Fibonacci Scanner Data Export
Exports all Fibonacci signals data in JSONB and Excel formats
"""

import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json

def connect_to_database():
    """Connect to the database"""
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in environment")
        return None
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def export_fibonacci_data_jsonb(conn):
    """Export Fibonacci data in JSONB format"""
    query = """
    SELECT 
        id,
        symbol,
        timeframe,
        signal_id,
        signal_type,
        fibonacci_level,
        price_level,
        current_price,
        confidence_score,
        volume_confirmation,
        momentum_confirmation,
        swing_high,
        swing_low,
        trend_direction,
        validation_rules_passed,
        scanner_version,
        algorithm_parameters,
        detected_at,
        created_at
    FROM other_scanners.fibonacci_signals
    ORDER BY detected_at DESC
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    
    # Convert to list of dictionaries
    data = []
    for row in rows:
        row_dict = dict(row)
        # Convert datetime objects to strings for JSON serialization
        if row_dict['detected_at']:
            row_dict['detected_at'] = row_dict['detected_at'].isoformat()
        if row_dict['created_at']:
            row_dict['created_at'] = row_dict['created_at'].isoformat()
        data.append(row_dict)
    
    return data

def export_fibonacci_summary_stats(conn):
    """Export summary statistics"""
    query = """
    SELECT 
        COUNT(*) as total_signals,
        ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
        COUNT(CASE WHEN confidence_score >= 0.75 THEN 1 END) as high_confidence_signals,
        COUNT(CASE WHEN confidence_score >= 0.80 THEN 1 END) as premium_signals,
        COUNT(DISTINCT symbol) as unique_symbols,
        MIN(detected_at) as first_signal,
        MAX(detected_at) as last_signal
    FROM other_scanners.fibonacci_signals
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query)
        return dict(cursor.fetchone())

def export_fibonacci_by_symbol(conn):
    """Export data grouped by symbol"""
    query = """
    SELECT 
        symbol,
        COUNT(*) as signal_count,
        ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
        MAX(confidence_score) as max_confidence,
        COUNT(CASE WHEN confidence_score >= 0.75 THEN 1 END) as high_conf_signals,
        COUNT(CASE WHEN confidence_score >= 0.80 THEN 1 END) as premium_signals,
        MIN(detected_at) as first_signal,
        MAX(detected_at) as last_signal
    FROM other_scanners.fibonacci_signals
    GROUP BY symbol
    ORDER BY avg_confidence DESC, signal_count DESC
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    
    data = []
    for row in rows:
        row_dict = dict(row)
        if row_dict['first_signal']:
            row_dict['first_signal'] = row_dict['first_signal'].isoformat()
        if row_dict['last_signal']:
            row_dict['last_signal'] = row_dict['last_signal'].isoformat()
        data.append(row_dict)
    
    return data

def export_fibonacci_by_level(conn):
    """Export data grouped by Fibonacci level"""
    query = """
    SELECT 
        fibonacci_level,
        COUNT(*) as signal_count,
        ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
        COUNT(CASE WHEN confidence_score >= 0.75 THEN 1 END) as high_conf_signals,
        COUNT(CASE WHEN confidence_score >= 0.80 THEN 1 END) as premium_signals
    FROM other_scanners.fibonacci_signals
    GROUP BY fibonacci_level
    ORDER BY signal_count DESC
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    
    return [dict(row) for row in rows]

def export_fibonacci_by_hour(conn):
    """Export data grouped by hour"""
    query = """
    SELECT 
        DATE_TRUNC('hour', detected_at) as hour,
        COUNT(*) as signals_per_hour,
        ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
        COUNT(DISTINCT symbol) as unique_symbols,
        COUNT(CASE WHEN confidence_score >= 0.75 THEN 1 END) as high_conf_signals
    FROM other_scanners.fibonacci_signals
    GROUP BY DATE_TRUNC('hour', detected_at)
    ORDER BY hour DESC
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    
    data = []
    for row in rows:
        row_dict = dict(row)
        if row_dict['hour']:
            row_dict['hour'] = row_dict['hour'].isoformat()
        data.append(row_dict)
    
    return data

def export_to_excel():
    """Export all Fibonacci data to Excel with multiple sheets"""
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        print("📊 Exporting Fibonacci scanner data...")
        
        # Get all data
        all_data = export_fibonacci_data_jsonb(conn)
        summary_stats = export_fibonacci_summary_stats(conn)
        symbol_data = export_fibonacci_by_symbol(conn)
        level_data = export_fibonacci_by_level(conn)
        hourly_data = export_fibonacci_by_hour(conn)
        
        # Create Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fibonacci_scanner_export_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # All signals data
            if all_data:
                df_all = pd.DataFrame(all_data)
                df_all.to_excel(writer, sheet_name='All_Signals', index=False)
                print(f"✅ Exported {len(all_data)} signals to 'All_Signals' sheet")
            
            # Summary statistics
            if summary_stats:
                df_summary = pd.DataFrame([summary_stats])
                df_summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
                print(f"✅ Exported summary statistics to 'Summary_Stats' sheet")
            
            # Symbol analysis
            if symbol_data:
                df_symbols = pd.DataFrame(symbol_data)
                df_symbols.to_excel(writer, sheet_name='Symbol_Analysis', index=False)
                print(f"✅ Exported {len(symbol_data)} symbols to 'Symbol_Analysis' sheet")
            
            # Fibonacci level analysis
            if level_data:
                df_levels = pd.DataFrame(level_data)
                df_levels.to_excel(writer, sheet_name='Level_Analysis', index=False)
                print(f"✅ Exported {len(level_data)} levels to 'Level_Analysis' sheet")
            
            # Hourly analysis
            if hourly_data:
                df_hourly = pd.DataFrame(hourly_data)
                df_hourly.to_excel(writer, sheet_name='Hourly_Analysis', index=False)
                print(f"✅ Exported {len(hourly_data)} hours to 'Hourly_Analysis' sheet")
        
        print(f"✅ Excel export complete: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None
    finally:
        conn.close()

def export_to_jsonb():
    """Export all Fibonacci data to JSONB format"""
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        print("📊 Exporting Fibonacci scanner data to JSONB...")
        
        # Get all data
        all_data = export_fibonacci_data_jsonb(conn)
        summary_stats = export_fibonacci_summary_stats(conn)
        symbol_data = export_fibonacci_by_symbol(conn)
        level_data = export_fibonacci_by_level(conn)
        hourly_data = export_fibonacci_by_hour(conn)
        
        # Create comprehensive JSONB structure
        jsonb_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_signals": len(all_data),
                "export_type": "fibonacci_scanner_complete"
            },
            "summary_statistics": summary_stats,
            "all_signals": all_data,
            "symbol_analysis": symbol_data,
            "level_analysis": level_data,
            "hourly_analysis": hourly_data
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fibonacci_scanner_export_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(jsonb_data, f, indent=2, default=str)
        
        print(f"✅ JSONB export complete: {filename}")
        print(f"📊 Total signals exported: {len(all_data)}")
        print(f"📈 Summary: {summary_stats['total_signals']} signals, {summary_stats['avg_confidence']} avg confidence")
        
        return filename
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None
    finally:
        conn.close()

def main():
    """Main export function"""
    print("=== FIBONACCI SCANNER DATA EXPORT ===")
    print("Exporting all Fibonacci signals data...")
    print()
    
    # Export to both formats
    excel_file = export_to_excel()
    jsonb_file = export_to_jsonb()
    
    print()
    print("=== EXPORT COMPLETE ===")
    if excel_file:
        print(f"📊 Excel file: {excel_file}")
    if jsonb_file:
        print(f"📄 JSONB file: {jsonb_file}")
    
    print()
    print("Files contain:")
    print("- All individual signals with full details")
    print("- Summary statistics")
    print("- Symbol-by-symbol analysis")
    print("- Fibonacci level performance")
    print("- Hourly signal generation rates")

if __name__ == "__main__":
    main()
