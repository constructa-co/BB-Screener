#!/usr/bin/env python3
"""
Test Data Fetching
Test the data fetching functions without streamlit dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_logger import TradeLogger
from datetime import datetime, timedelta

def test_get_best_opportunities():
    """Test getting best opportunities"""
    print("🔍 Testing get_best_opportunities...")
    
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
                LIMIT 50
            """, (70,))
            
            opportunities = logger.cursor.fetchall()
            print(f"✅ Found {len(opportunities)} opportunities")
            
            if opportunities:
                # Show first few opportunities
                for i, opp in enumerate(opportunities[:3]):
                    print(f"  {i+1}. Symbol: {opp[3]}, Probability: {opp[7]}%, R/R: {opp[8]}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    logger.close()
    return len(opportunities)

def test_get_current_metrics():
    """Test getting current metrics"""
    print("\n🔍 Testing get_current_metrics...")
    
    logger = TradeLogger()
    metrics = {
        'opportunities': 0,
        'scans': 0,
        'high_prob': 0,
        'win_rate': 0
    }
    
    if logger.connection:
        try:
            # Get metrics from last 24 hours
            logger.cursor.execute("""
                SELECT 
                    COUNT(DISTINCT s.id) as scans,
                    COUNT(t.id) as opportunities,
                    COUNT(CASE WHEN t.probability >= 80 THEN 1 END) as high_prob
                FROM scan_results s
                LEFT JOIN trade_opportunities t ON s.id = t.scan_id
                WHERE s.scan_timestamp > NOW() - INTERVAL '24 hours'
            """)
            
            result = logger.cursor.fetchone()
            print(f"Debug - Raw result: {result}")
            if result:
                # Handle both tuple and dict-like results
                if hasattr(result, '__getitem__') and not isinstance(result, (list, tuple)):
                    # RealDictRow or similar
                    metrics['scans'] = result['scans'] if result['scans'] is not None else 0
                    metrics['opportunities'] = result['opportunities'] if result['opportunities'] is not None else 0
                    metrics['high_prob'] = result['high_prob'] if result['high_prob'] is not None else 0
                else:
                    # Tuple result
                    metrics['scans'] = result[0] if result[0] is not None else 0
                    metrics['opportunities'] = result[1] if result[1] is not None else 0
                    metrics['high_prob'] = result[2] if result[2] is not None else 0
                
                print(f"✅ Scans: {metrics['scans']}")
                print(f"✅ Opportunities: {metrics['opportunities']}")
                print(f"✅ High Probability: {metrics['high_prob']}")
            else:
                print("⚠️ No result from query")
                
                # Calculate win rate from completed trades
                logger.cursor.execute("""
                    SELECT 
                        COUNT(CASE WHEN profit_loss_percent > 0 THEN 1 END) as wins,
                        COUNT(*) as total_trades
                    FROM trade_opportunities 
                    WHERE trade_taken = TRUE 
                    AND profit_loss_percent IS NOT NULL
                    AND timestamp > NOW() - INTERVAL '24 hours'
                """)
                
                win_result = logger.cursor.fetchone()
                if win_result and win_result[1] > 0:
                    metrics['win_rate'] = (win_result[0] / win_result[1]) * 100
                    print(f"✅ Win Rate: {metrics['win_rate']:.1f}% ({win_result[0]}/{win_result[1]} trades)")
                else:
                    print("⚠️ No completed trades for win rate calculation")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    logger.close()
    return metrics

def test_get_latest_opportunities():
    """Test getting latest opportunities"""
    print("\n🔍 Testing get_latest_opportunities...")
    
    logger = TradeLogger()
    opportunities = []
    
    if logger.connection:
        try:
            logger.cursor.execute("""
                SELECT 
                    t.*,
                    s.scan_type as scanner_type
                FROM trade_opportunities t
                JOIN scan_results s ON t.scan_id = s.id
                WHERE t.timestamp > NOW() - INTERVAL '24 hours'
                ORDER BY t.timestamp DESC
                LIMIT 10
            """)
            
            opportunities = logger.cursor.fetchall()
            print(f"✅ Found {len(opportunities)} latest opportunities")
            
            if opportunities:
                # Show first few opportunities
                for i, opp in enumerate(opportunities[:3]):
                    print(f"  {i+1}. Symbol: {opp[3]}, Scanner: {opp[-1]}, Time: {opp[2]}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    logger.close()
    return len(opportunities)

def main():
    """Main test function"""
    print("🔧 Data Fetching Test")
    print("=" * 50)
    
    # Test all functions
    opp_count = test_get_best_opportunities()
    metrics = test_get_current_metrics()
    latest_count = test_get_latest_opportunities()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Opportunities: {opp_count}")
    print(f"  Latest Opportunities: {latest_count}")
    print(f"  Scans (24h): {metrics['scans']}")
    print(f"  Total Opportunities (24h): {metrics['opportunities']}")
    print(f"  High Probability: {metrics['high_prob']}")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")

if __name__ == "__main__":
    main() 