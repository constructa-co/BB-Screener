# test_confidence_live.py
import asyncio
from main_scanner import BBScreener

async def quick_confidence_test():
    scanner = BBScreener()
    # Test with just 10 coins instead of 500+
    scanner.data_fetcher.top_coins_limit = 10
    results = await scanner.scan_all_coins_comprehensive()
    
    if results:
        print(f"✅ Found {len(results)} results")
        print("✅ Testing confidence enhancement...")
        # This will trigger the confidence module
        market_regime = {'regime_type': 'MIXED', 'regime_confidence': 60}
        enhanced = scanner.confidence_module.enhance_all_trades_with_confidence(
            results, market_regime, scanner.confidence_module
        )
        print(f"✅ Confidence enhancement successful: {len(enhanced)} trades")
        return True
    else:
        print("❌ No results found for testing")
        return False

if __name__ == "__main__":
    asyncio.run(quick_confidence_test())