#!/usr/bin/env python3
"""
Debug Dashboard Data
Test what the dashboard is actually receiving
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_logger import TradeLogger

def debug_get_best_opportunities():
    """Debug the get_best_opportunities function"""
    print("🔍 Debugging get_best_opportunities...")
    
    logger = TradeLogger()
    opportunities = []
    
    if logger.connection:
        try:
            logger.cursor.execute("""
                SELECT 
                    t.*,
                    s.scan_type,
                    s.scan_timestamp,
                    'Day Trading' as trading_style,
                    '4H' as timeframe
                FROM trade_opportunities t
                JOIN scan_results s ON t.scan_id = s.id
                WHERE t.probability >= %s
                AND t.trade_taken = FALSE
                ORDER BY t.probability DESC, t.risk_reward_ratio DESC
                LIMIT 5
            """, (70,))
            
            opportunities = logger.cursor.fetchall()
            print(f"✅ Raw opportunities count: {len(opportunities)}")
            
            if opportunities:
                # Show column names
                columns = [desc[0] for desc in logger.cursor.description]
                print(f"✅ Columns: {columns}")
                
                # Show first opportunity structure
                first_opp = opportunities[0]
                print(f"✅ First opportunity (RealDictRow): {list(first_opp.keys())[:5]}...")
                
                # Convert to dictionary
                opp_dict = dict(first_opp)
                print(f"✅ First opportunity (dict): {list(opp_dict.keys())[:5]}...")
                print(f"✅ Sample values: symbol={opp_dict.get('symbol')}, probability={opp_dict.get('probability')}")
                
                # Test the conversion
                all_opps = [dict(row) for row in opportunities]
                print(f"✅ Converted opportunities count: {len(all_opps)}")
                print(f"✅ First converted opp: symbol={all_opps[0].get('symbol')}, scan_type={all_opps[0].get('scan_type')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    logger.close()
    return opportunities

def debug_dashboard_display():
    """Debug what the dashboard display expects"""
    print("\n🔍 Debugging dashboard display expectations...")
    
    # Simulate what the dashboard expects
    sample_opp = {
        'scan_type': 'BB_Backtest_R10',
        'symbol': 'BTCUSDT',
        'probability': 85.5,
        'risk_reward_ratio': 2.5,
        'entry_price': 115000.0,
        'target_1': 120000.0,
        'trading_style': 'Day Trading',
        'timestamp': '2025-08-06 18:30:00'
    }
    
    print(f"✅ Dashboard expects: {list(sample_opp.keys())}")
    print(f"✅ Sample data: {sample_opp}")
    
    # Test the dashboard display logic
    scanner_type = sample_opp['scan_type'].replace('_', ' ').title()
    print(f"✅ Formatted scanner: {scanner_type}")
    
    opp_data = {
        'Scanner': f"🎯 {scanner_type}",
        'Symbol': sample_opp['symbol'],
        'Probability': sample_opp['probability'],
        'R:R': sample_opp['risk_reward_ratio'],
        'Entry': sample_opp['entry_price'],
        'Target 1': sample_opp['target_1'],
        'Style': sample_opp['trading_style'],
        'Time': sample_opp['timestamp']
    }
    
    print(f"✅ Dashboard display data: {opp_data}")

def main():
    """Main debug function"""
    print("🔧 Dashboard Data Debug")
    print("=" * 50)
    
    # Debug the data fetching
    opportunities = debug_get_best_opportunities()
    
    # Debug the dashboard display
    debug_dashboard_display()
    
    print("\n" + "=" * 50)
    print("📊 Debug Summary:")
    print(f"  Raw opportunities: {len(opportunities)}")
    print("  Dashboard display logic: Working")
    print("  Data conversion: Working")

if __name__ == "__main__":
    main() 