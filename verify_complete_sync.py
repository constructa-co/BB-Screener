import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from datetime import datetime

def verify_sync():
    """Verify that all data is being captured correctly in the database"""
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️  python-dotenv not available - using system environment variables")
        def load_dotenv():
            pass
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ No DATABASE_URL found in environment variables")
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    print("="*60)
    print("DATABASE SYNC VERIFICATION")
    print("="*60)
    
    # Check latest sync
    cursor.execute("""
        SELECT COUNT(*) as total,
               MAX(timestamp) as latest,
               MIN(timestamp) as earliest
        FROM trade_opportunities
        WHERE DATE(timestamp) = CURRENT_DATE
    """)
    result = cursor.fetchone()
    print(f"✓ Total trades today: {result['total']}")
    print(f"✓ Latest update: {result['latest']}")
    print(f"✓ Earliest update: {result['earliest']}")
    
    if result['total'] == 0:
        print("⚠️  No trades found for today - checking yesterday...")
        cursor.execute("""
            SELECT COUNT(*) as total,
                   MAX(timestamp) as latest
            FROM trade_opportunities
            WHERE DATE(timestamp) = CURRENT_DATE - INTERVAL '1 day'
        """)
        result = cursor.fetchone()
        print(f"✓ Total trades yesterday: {result['total']}")
        print(f"✓ Latest update: {result['latest']}")
    
    # Check for all critical fields in latest trade
    cursor.execute("""
        SELECT scanner_specific_data::text
        FROM trade_opportunities
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    
    latest_row = cursor.fetchone()
    if not latest_row or not latest_row['scanner_specific_data']:
        print("❌ No trades found with scanner_specific_data")
        return
    
    latest_trade = json.loads(latest_row['scanner_specific_data'])
    
    # Check field counts
    print(f"\n📊 FIELD COUNT ANALYSIS:")
    print(f"✓ Total fields in JSONB: {len(latest_trade)}")
    
    # Check for market context
    has_market = 'market_context' in latest_trade
    print(f"{'✓' if has_market else '✗'} Market context: {'Present' if has_market else 'MISSING'}")
    
    if has_market:
        market_context = latest_trade['market_context']
        print(f"  Market context breakdown:")
        for key, value in market_context.items():
            if isinstance(value, list):
                print(f"    ✓ {key}: {len(value)} records")
            else:
                print(f"    ⚠️ {key}: {type(value)} (expected list)")
    
    # Check for sentiment
    has_sentiment = any('sentiment' in str(v).lower() for v in latest_trade.values() if v)
    print(f"{'✓' if has_sentiment else '✗'} Sentiment data: {'Present' if has_sentiment else 'MISSING'}")
    
    # Check critical fields
    critical_fields = [
        'scanner_type', 'timeframe', 'confidence_rationale',
        'market_regime_data', 'tier_base_bb', 'component_1',
        'probability', 'entry_price', 'stop_loss', 'target_1'
    ]
    
    print(f"\n🔍 CRITICAL FIELDS CHECK:")
    missing_fields = []
    for field in critical_fields:
        present = field in latest_trade or any(field in str(k) for k in latest_trade.keys())
        status = '✓' if present else '✗'
        print(f"  {status} {field}: {'Present' if present else 'MISSING'}")
        if not present:
            missing_fields.append(field)
    
    # Check for 139 fields (Excel standard)
    print(f"\n📈 COMPREHENSIVE DATA CHECK:")
    if len(latest_trade) >= 139:
        print(f"✓ EXCELLENT: {len(latest_trade)} fields (≥139 target)")
    elif len(latest_trade) >= 120:
        print(f"⚠️  GOOD: {len(latest_trade)} fields (120-138 range)")
    else:
        print(f"❌ INCOMPLETE: {len(latest_trade)} fields (<120)")
    
    # Check for specific data types
    print(f"\n🎯 DATA QUALITY CHECK:")
    
    # Check for market regime data
    has_regime = any('regime' in str(k).lower() for k in latest_trade.keys())
    print(f"{'✓' if has_regime else '✗'} Market regime data: {'Present' if has_regime else 'MISSING'}")
    
    # Check for confidence data
    has_confidence = any('confidence' in str(k).lower() for k in latest_trade.keys())
    print(f"{'✓' if has_confidence else '✗'} Confidence data: {'Present' if has_confidence else 'MISSING'}")
    
    # Check for technical indicators
    has_technical = any(indicator in str(latest_trade.keys()) for indicator in ['rsi', 'macd', 'bb_', 'volume'])
    print(f"{'✓' if has_technical else '✗'} Technical indicators: {'Present' if has_technical else 'MISSING'}")
    
    # Check for metadata
    has_metadata = any('metadata' in str(k).lower() for k in latest_trade.keys())
    print(f"{'✓' if has_metadata else '✗'} Market metadata: {'Present' if has_metadata else 'MISSING'}")
    
    # Summary
    print(f"\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if len(latest_trade) >= 139 and has_market and has_sentiment and len(missing_fields) == 0:
        print("🎉 EXCELLENT: All data captured correctly!")
        print("   - All 139+ fields present")
        print("   - Market context included")
        print("   - Sentiment data captured")
        print("   - No critical fields missing")
    elif len(latest_trade) >= 120 and has_market:
        print("✅ GOOD: Most data captured correctly")
        print("   - 120+ fields present")
        print("   - Market context included")
        if missing_fields:
            print(f"   - Missing fields: {', '.join(missing_fields)}")
    else:
        print("❌ ISSUES DETECTED:")
        if len(latest_trade) < 120:
            print(f"   - Insufficient fields: {len(latest_trade)}")
        if not has_market:
            print("   - Market context missing")
        if missing_fields:
            print(f"   - Missing critical fields: {', '.join(missing_fields)}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*60)
    print("Verification complete!")

if __name__ == "__main__":
    verify_sync()

