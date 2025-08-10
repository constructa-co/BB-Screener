#!/usr/bin/env python3
"""
Verify enhanced data capture
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def verify_enhanced_data():
    logger = TradeLogger()
    
    # Check market_regime table
    logger.cursor.execute("""
        SELECT COUNT(*) as count FROM market_regime
    """)
    regime_count = logger.cursor.fetchone()['count']
    print(f"📊 Market regime records: {regime_count}")
    
    if regime_count > 0:
        logger.cursor.execute("""
            SELECT btc_dominance, fear_greed_index, regime_type, regime_confidence, regime_data
            FROM market_regime ORDER BY id DESC LIMIT 1
        """)
        regime = logger.cursor.fetchone()
        print(f"   • Latest regime: {regime['regime_type']} ({regime['regime_confidence']}% confidence)")
        print(f"   • BTC Dominance: {regime['btc_dominance']}%")
        print(f"   • Fear & Greed: {regime['fear_greed_index']}")
    
    # Check market_overview table
    logger.cursor.execute("""
        SELECT COUNT(*) as count FROM market_overview
    """)
    overview_count = logger.cursor.fetchone()['count']
    print(f"📊 Market overview records: {overview_count}")
    
    if overview_count > 0:
        logger.cursor.execute("""
            SELECT total_bounces, coins_analyzed, overall_success_rate, bb_squeeze_effectiveness
            FROM market_overview ORDER BY id DESC LIMIT 1
        """)
        overview = logger.cursor.fetchone()
        print(f"   • Total bounces: {overview['total_bounces']}")
        print(f"   • Coins analyzed: {overview['coins_analyzed']}")
        print(f"   • Success rate: {overview['overall_success_rate']}%")
        print(f"   • BB squeeze effectiveness: {overview['bb_squeeze_effectiveness']}%")
    
    # Check market_metadata table
    logger.cursor.execute("""
        SELECT COUNT(*) as count FROM market_metadata
    """)
    metadata_count = logger.cursor.fetchone()['count']
    print(f"📊 Market metadata records: {metadata_count}")
    
    if metadata_count > 0:
        logger.cursor.execute("""
            SELECT large_cap_count, mid_cap_count, small_cap_count, micro_cap_count
            FROM market_metadata ORDER BY id DESC LIMIT 1
        """)
        metadata = logger.cursor.fetchone()
        print(f"   • Large cap: {metadata['large_cap_count']}")
        print(f"   • Mid cap: {metadata['mid_cap_count']}")
        print(f"   • Small cap: {metadata['small_cap_count']}")
        print(f"   • Micro cap: {metadata['micro_cap_count']}")
    
    # Check total trades
    logger.cursor.execute("""
        SELECT COUNT(*) as count FROM trade_opportunities t
        JOIN scan_results s ON t.scan_id = s.id
        WHERE s.scan_type = 'bb_scanner'
    """)
    trade_count = logger.cursor.fetchone()['count']
    print(f"📊 Total BB scanner trades: {trade_count}")
    
    # Check ML view
    try:
        logger.cursor.execute("""
            SELECT COUNT(*) as count FROM ml_training_data
        """)
        ml_count = logger.cursor.fetchone()['count']
        print(f"📊 ML training data records: {ml_count}")
    except:
        print("⚠️ ML view not accessible")
    
    logger.close()

if __name__ == "__main__":
    verify_enhanced_data()
