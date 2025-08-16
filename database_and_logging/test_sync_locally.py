"""
Test Excel to Database Sync Locally
Uses existing Excel files - no need to run scanner
Location: database_and_logging/test_sync_locally.py
"""
import pandas as pd
from sqlalchemy import create_engine
import glob
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Add parent directory to path so we can import trade_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get project root directory (one level up from database_and_logging)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

def test_sync_with_existing_excel():
    """Test with Excel files you already have"""
    
    print("="*60)
    print("EXCEL TO DATABASE SYNC TESTER")
    print("="*60)
    
    # Find Excel files in outputs folder
    excel_pattern = os.path.join(PROJECT_ROOT, 'outputs', 'excel_reports', 'bb_analysis_*.xlsx')
    excel_files = glob.glob(excel_pattern)
    
    if not excel_files:
        print(f"❌ No Excel files found in {excel_pattern}")
        return
    
    print(f"\n📁 Found {len(excel_files)} Excel files:")
    # Show last 5 files
    for f in sorted(excel_files)[-5:]:
        file_size = os.path.getsize(f) / 1024  # Size in KB
        file_name = os.path.basename(f)
        print(f"   - {file_name} ({file_size:.1f} KB)")
    
    # Use the latest one
    latest_excel = max(excel_files)
    latest_name = os.path.basename(latest_excel)
    print(f"\n📊 Using latest: {latest_name}")
    
    # Test reading the Excel file
    try:
        excel_file = pd.ExcelFile(latest_excel)
        print(f"📑 Sheets found: {excel_file.sheet_names}")
    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        return
    
    # Show preview of each sheet
    print("\n📋 Sheet Contents:")
    for sheet in excel_file.sheet_names:
        try:
            df = pd.read_excel(latest_excel, sheet_name=sheet)
            print(f"\n   📌 {sheet}:")
            print(f"      - Rows: {len(df)}")
            print(f"      - Columns: {len(df.columns)}")
            if len(df) > 0 and len(df.columns) > 0:
                print(f"      - First 5 columns: {list(df.columns)[:5]}")
                if 'symbol' in df.columns:
                    print(f"      - Sample symbols: {df['symbol'].head(3).tolist()}")
        except Exception as e:
            print(f"      ❌ Error reading sheet: {e}")
    
    # Database connection test
    print("\n" + "="*60)
    print("DATABASE CONNECTION TEST")
    print("="*60)
    
    # Check if we have database credentials
    # Check if we have database credentials
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ Missing DATABASE_URL in .env file")
        return

    # Parse the URL to show connection info
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    db_host = parsed.hostname
    db_name = parsed.path[1:]  # Remove leading /
    db_user = parsed.username
    
    if not all([db_host, db_name, db_user]):
        print("❌ Missing database credentials in .env file")
        print("   Required: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT")
        return
    
    print(f"📡 Connecting to: {db_user}@{db_host}/{db_name}")
    
    try:
        # Use the DATABASE_URL directly
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT version()")
            version = result.scalar()
            print(f"✅ Connected successfully!")
            print(f"   PostgreSQL version: {version[:40]}...")
            
            # Check existing tables
            result = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in result]
            print(f"\n📊 Existing tables in database: {len(tables)}")
            for table in tables[:10]:  # Show first 10
                print(f"   - {table}")
            
    except Exception as e:
        print(f"\n❌ Database connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has correct credentials")
        print("2. Ensure database is accessible from your location")
        print("3. Check if you need VPN or SSH tunnel")
        return
    
    # Sheet to table mapping
    sheet_mapping = {
        'All_Analysis': 'trade_opportunities',
        'Premium_High_Only': 'trade_opportunities_premium',
        'Trade_Recommendations': 'trade_opportunities_recommended',
        'Low_Risk_Trades': 'trade_opportunities_low_risk',
        'Monitoring_List': 'trade_opportunities_monitoring',
        'Market_Regime_Analysis': 'market_regime_data',
        'Market_Overview': 'market_overview_data',
        'Market_Metadata': 'market_metadata',
        'Confidence_Summary': 'confidence_data',
        'Top_10_Sentiment': 'sentiment_data'
    }
    
    print("\n" + "="*60)
    print("SYNC PLAN")
    print("="*60)
    print("Sheet → Table mapping:")
    for sheet, table in sheet_mapping.items():
        if sheet in excel_file.sheet_names:
            df_temp = pd.read_excel(latest_excel, sheet_name=sheet)
            print(f"   {sheet} ({len(df_temp)} rows) → {table}")
    
    # Ask for confirmation
    print("\n" + "="*60)
    response = input("🔄 Ready to sync to database? (y/n): ")
    if response.lower() != 'y':
        print("❌ Sync cancelled")
        return
    
    # Perform the sync
    print("\n" + "="*60)
    print("SYNCING DATA")
    print("="*60)
    
    success_count = 0
    error_count = 0
    
    for sheet_name in excel_file.sheet_names:
        if sheet_name not in sheet_mapping:
            print(f"⚠️  Skipping unmapped sheet: {sheet_name}")
            continue
            
        try:
            df = pd.read_excel(latest_excel, sheet_name=sheet_name)
            
            if len(df) == 0:
                print(f"⚠️  {sheet_name}: Empty sheet, skipping")
                continue
            
            # Add metadata
            df['import_timestamp'] = datetime.now()
            df['source_file'] = latest_name
            df['source_sheet'] = sheet_name
            
            # Get table name
            table_name = sheet_mapping[sheet_name]
            
            # For testing, use 'replace' to overwrite existing data
            # For production, use 'append' to keep history
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            
            print(f"✅ {sheet_name} → {table_name}: {len(df)} rows synced")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {sheet_name}: Error - {str(e)[:100]}")
            error_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("SYNC COMPLETE")
    print("="*60)
    print(f"✅ Successful: {success_count} sheets")
    print(f"❌ Errors: {error_count} sheets")
    
    # Verify data in database
    print("\n📊 Verifying data in database:")
    with engine.connect() as conn:
        for table in sheet_mapping.values():
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = result.scalar()
                if count > 0:
                    print(f"   ✓ {table}: {count} rows")
            except:
                pass  # Table might not exist
    
    print("\n🎉 Done! Your Excel data is now in the database")
    print("   Run again anytime to update with latest Excel")

if __name__ == "__main__":
    try:
        test_sync_with_existing_excel()
    except KeyboardInterrupt:
        print("\n\n❌ Sync interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()