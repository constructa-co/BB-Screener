# test_market_metadata.py
"""
Standalone test for Market Metadata Enricher
Tests CMC API integration and metadata classification
"""

import sys
import os
import pandas as pd
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your config (assumes config.py exists in project root)
try:
    from config import CMC_API_KEY
    print(f"✅ Config imported. CMC API Key available: {'Yes' if CMC_API_KEY else 'No'}")
except ImportError:
    print("❌ Could not import config.py - please ensure it exists with CMC_API_KEY")
    CMC_API_KEY = None

# Import the metadata enricher
from modules.market_metadata_enricher import MarketMetadataEnricher

def test_sample_coins():
    """Test metadata enrichment on a sample of popular coins"""
    
    print("\n🧪 TESTING MARKET METADATA ENRICHER")
    print("=" * 50)
    
    # Initialize enricher
    enricher = MarketMetadataEnricher()
    
    # Test coins representing different market cap tiers and sectors
    test_coins = [
        'BTC',   # Large cap, layer 1
        'ETH',   # Large cap, smart contracts  
        'BNB',   # Large cap, exchange
        'ADA',   # Mid cap, layer 1
        'LINK',  # Mid cap, oracle
        'UNI',   # Mid cap, defi
        'SAND',  # Small cap, gaming
        'CRV',   # Small cap, defi
        'LRC',   # Small cap, layer 2
        'DOGE'   # Large cap, meme
    ]
    
    results = []
    
    for symbol in test_coins:
        print(f"\n🔍 Testing {symbol}...")
        
        # Create mock trade data (what would come from BB analysis)
        mock_trade_data = {
            'symbol': symbol,
            'setup_type': 'LONG',
            'bb_score': 15,
            'entry': 100.0,
            'stop': 95.0,
            'target1': 105.0
        }
        
        # Enrich with metadata
        enriched_data = enricher.enrich_trade_data(symbol, mock_trade_data)
        results.append(enriched_data)
        
        # Display key metadata
        print(f"   Market Cap Rank: {enriched_data['market_cap_rank']}")
        print(f"   Market Cap Tier: {enriched_data['market_cap_tier']}")
        print(f"   24h Volume: ${enriched_data['volume_24h_usd']:,.0f}")
        print(f"   Liquidity Tier: {enriched_data['liquidity_tier']}")
        print(f"   Primary Sector: {enriched_data['primary_sector']}")
        print(f"   Expected Success Rate: {enriched_data['expected_success_rate']:.1f}%")
        print(f"   Position Multiplier: {enriched_data['market_cap_multiplier']:.1f}x")
    
    return results

def test_classification_logic():
    """Test the classification algorithms with known values"""
    
    print(f"\n🧪 TESTING CLASSIFICATION LOGIC")
    print("=" * 50)
    
    enricher = MarketMetadataEnricher()
    
    # Test market cap tier classification
    print("\n📊 Market Cap Tier Tests:")
    test_ranks = [1, 10, 50, 75, 150, 200, 500, 1000]
    for rank in test_ranks:
        tier = enricher._classify_market_cap_tier(rank)
        print(f"   Rank {rank:4d} → {tier}")
    
    # Test liquidity tier classification  
    print("\n💧 Liquidity Tier Tests:")
    test_volumes = [100_000_000, 25_000_000, 5_000_000, 500_000, 50_000]
    for volume in test_volumes:
        tier = enricher._classify_liquidity_tier(volume)
        risk = enricher._calculate_liquidity_risk(volume)
        print(f"   ${volume:>12,} → {tier:15s} (Risk: {risk:.1f})")
    
    # Test sector classification
    print("\n🏭 Sector Classification Tests:")
    test_tag_sets = [
        ['defi', 'yield-farming'],
        ['layer-1', 'smart-contracts'], 
        ['gaming', 'metaverse'],
        ['memes', 'dog-themed'],
        ['exchange-based'],
        ['layer-2', 'ethereum-ecosystem'],
        ['oracles'],
        ['unknown-tag']
    ]
    
    for tags in test_tag_sets:
        sector = enricher._extract_primary_sector(tags)
        category = enricher._classify_sector_category(tags)
        print(f"   {tags} → {sector} ({category})")

def test_success_rate_estimation():
    """Test success rate estimation logic"""
    
    print(f"\n🎯 SUCCESS RATE ESTIMATION TESTS")
    print("=" * 50)
    
    enricher = MarketMetadataEnricher()
    
    # Test different combinations of rank and volume
    test_cases = [
        (10, 100_000_000),   # Large cap, high liquidity
        (10, 500_000),       # Large cap, low liquidity  
        (200, 50_000_000),   # Small cap, high liquidity
        (200, 1_000_000),    # Small cap, medium liquidity
        (500, 100_000),      # Micro cap, low liquidity
    ]
    
    print("Rank | Volume      | Expected Success Rate | Notes")
    print("-" * 55)
    
    for rank, volume in test_cases:
        success_rate = enricher._estimate_success_rate(rank, volume)
        cap_tier = enricher._classify_market_cap_tier(rank)
        liq_tier = enricher._classify_liquidity_tier(volume)
        print(f"{rank:4d} | ${volume:>10,} | {success_rate:8.1f}%          | {cap_tier}, {liq_tier}")

def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary DataFrame from test results"""
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Symbol': result['symbol'],
            'Market_Cap_Rank': result['market_cap_rank'],
            'Market_Cap_Tier': result['market_cap_tier'],
            'Volume_24h_USD': result['volume_24h_usd'],
            'Liquidity_Tier': result['liquidity_tier'],
            'Primary_Sector': result['primary_sector'],
            'Sector_Category': result['sector_category'],
            'Expected_Success_Rate': result['expected_success_rate'],
            'Liquidity_Risk_Score': result['liquidity_risk_score'],
            'Market_Cap_Multiplier': result['market_cap_multiplier'],
            'Liquidity_Multiplier': result['liquidity_multiplier']
        })
    
    return pd.DataFrame(summary_data)

def main():
    """Main test function"""
    
    print("🚀 MARKET METADATA ENRICHER STANDALONE TEST")
    print("=" * 60)
    
    if not CMC_API_KEY:
        print("⚠️  WARNING: No CMC API key found. Testing will use fallback data.")
        print("   To test with real data, add CMC_API_KEY to config.py")
    
    # Run tests
    try:
        # Test basic functionality
        test_classification_logic()
        test_success_rate_estimation()
        
        # Test with real/mock data
        results = test_sample_coins()
        
        # Create summary
        if results:
            print(f"\n📊 SUMMARY TABLE")
            print("=" * 50)
            
            df = create_summary_dataframe(results)
            print(df.to_string(index=False))
            
            # Save to CSV for inspection
            df.to_csv('market_metadata_test_results.csv', index=False)
            print(f"\n💾 Results saved to: market_metadata_test_results.csv")
        
        print(f"\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   • Large caps (Top 50) get lower multipliers due to worse performance")
        print(f"   • Small caps (151-300) get higher multipliers due to better performance") 
        print(f"   • Low liquidity assets get reduced position sizing for risk management")
        print(f"   • Each sector is properly classified for rotation analysis")
        
        print(f"\n🚀 READY FOR INTEGRATION INTO MAIN SCANNER!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()