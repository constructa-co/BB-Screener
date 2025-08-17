#!/usr/bin/env python3
"""
Inspect JSONB Structure Script
Analyzes what's actually stored in the scanner_specific_data JSONB field
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from collections import defaultdict
from dotenv import load_dotenv

def inspect_jsonb_structure():
    """Inspect the JSONB structure of recent trades"""
    
    print("🔍 Connecting to PostgreSQL database...")
    load_dotenv()
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in environment variables")
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        print("✅ Connected to database successfully")
        
        # Get trades from different times today
        cursor.execute("""
            SELECT 
                symbol, 
                timestamp,
                LENGTH(scanner_specific_data::text) as json_size,
                scanner_specific_data
            FROM trade_opportunities 
            WHERE DATE(timestamp) = CURRENT_DATE
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        trades = cursor.fetchall()
        
        for i, row in enumerate(trades):
            print(f"\n{'='*80}")
            print(f"Trade {i+1}: {row['symbol']} at {row['timestamp']}")
            print(f"{'='*80}")
            
            # Parse JSONB
            jsonb = json.loads(row['scanner_specific_data']) if isinstance(row['scanner_specific_data'], str) else row['scanner_specific_data']
            
            print(f"JSON size: {row['json_size']} bytes")
            print(f"Field count: {len(jsonb)}")
            
            # Check for market context
            has_market_context = 'market_context' in jsonb or 'market_regime_data' in jsonb
            has_market_overview = 'market_overview_data' in jsonb or 'Market_Overview_data' in jsonb
            has_market_metadata = 'market_metadata_data' in jsonb or 'Market_Metadata_data' in jsonb
            
            print(f"Has market_context: {has_market_context}")
            print(f"Has market_overview: {has_market_overview}")
            print(f"Has market_metadata: {has_market_metadata}")
            
            # Categorize fields
            categories = defaultdict(list)
            for key, value in jsonb.items():
                if 'market' in key.lower():
                    categories['MARKET'].append(key)
                elif 'regime' in key.lower():
                    categories['REGIME'].append(key)
                elif 'sentiment' in key.lower():
                    categories['SENTIMENT'].append(key)
                elif 'confidence' in key.lower():
                    categories['CONFIDENCE'].append(key)
                elif isinstance(value, (dict, list)):
                    categories['NESTED'].append(key)
                else:
                    categories['STANDARD'].append(key)
            
            # Display categories
            for category, fields in categories.items():
                if fields:
                    print(f"\n{category} Fields ({len(fields)}):")
                    for field in fields[:5]:  # Show first 5
                        value = jsonb[field]
                        if isinstance(value, dict):
                            print(f"  {field}: <dict with {len(value)} items>")
                        elif isinstance(value, list):
                            print(f"  {field}: <list with {len(value)} items>")
                        else:
                            print(f"  {field}: {str(value)[:50]}...")
            
            # Check specifically for market sheets
            print("\nMarket Sheet Fields:")
            sheet_fields = [
                'Market_Regime_Analysis_data', 'Market_Overview_data', 
                'Market_Metadata_data', 'market_regime_data', 
                'market_overview_data', 'market_context',
                'market_regime_analysis', 'market_overview',
                'market_metadata', 'confidence_summary',
                'top_10_sentiment'
            ]
            
            for field in sheet_fields:
                if field in jsonb:
                    value = jsonb[field]
                    if isinstance(value, list) and len(value) > 0:
                        print(f"  ✅ {field}: {len(value)} records")
                    elif isinstance(value, dict):
                        print(f"  ✅ {field}: dict with {len(value)} keys")
                    else:
                        print(f"  ⚠️ {field}: empty or invalid")
                else:
                    print(f"  ❌ {field}: NOT FOUND")
            
            # Show first few keys
            print(f"\nFirst 10 fields: {list(jsonb.keys())[:10]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_jsonb_structure()
