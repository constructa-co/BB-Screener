#!/usr/bin/env python3
"""
Extract JSONB Sample Script
Extracts raw JSONB content from the database for inspection
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from datetime import datetime
from dotenv import load_dotenv

def extract_jsonb_sample():
    """Extract sample JSONB content from database"""
    
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
        
        # Get the most recent trade with enhanced data
        cursor.execute("""
            SELECT 
                symbol, 
                timestamp,
                scanner_specific_data
            FROM trade_opportunities 
            WHERE DATE(timestamp) = CURRENT_DATE
            ORDER BY LENGTH(scanner_specific_data::text) DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row:
            print(f"📊 Extracting JSONB from: {row['symbol']} at {row['timestamp']}")
            
            # Parse the JSONB
            jsonb = json.loads(row['scanner_specific_data']) if isinstance(row['scanner_specific_data'], str) else row['scanner_specific_data']
            
            # Save to file
            output_file = f"jsonb_sample_{row['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_file, 'w') as f:
                json.dump(jsonb, f, indent=2, default=str)
            
            print(f"✅ Saved JSONB sample to: {output_file}")
            print(f"📈 JSONB size: {len(str(jsonb))} bytes")
            print(f"🔢 Field count: {len(jsonb)}")
            
            # Show field categories
            market_fields = [k for k in jsonb.keys() if 'market' in k.lower()]
            regime_fields = [k for k in jsonb.keys() if 'regime' in k.lower()]
            sentiment_fields = [k for k in jsonb.keys() if 'sentiment' in k.lower()]
            
            print(f"\n📋 Field Summary:")
            print(f"  Market fields: {len(market_fields)}")
            print(f"  Regime fields: {len(regime_fields)}")
            print(f"  Sentiment fields: {len(sentiment_fields)}")
            print(f"  Total fields: {len(jsonb)}")
            
            # Show first 20 field names
            print(f"\n🔍 First 20 fields:")
            for i, field in enumerate(list(jsonb.keys())[:20]):
                value = jsonb[field]
                if isinstance(value, dict):
                    print(f"  {i+1:2d}. {field}: <dict with {len(value)} items>")
                elif isinstance(value, list):
                    print(f"  {i+1:2d}. {field}: <list with {len(value)} items>")
                else:
                    print(f"  {i+1:2d}. {field}: {str(value)[:50]}...")
            
            return output_file
        else:
            print("❌ No trades found")
            return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    extract_jsonb_sample()
