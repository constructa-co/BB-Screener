from main_scanner import ModularBBScanner

if __name__ == "__main__":
    scanner = ModularBBScanner()
    # Run analysis for a single coin (BTC)
    results = scanner.analyze_coin_comprehensive('BTC', market_regime=None)
    
    print("\nTest Results for BTC:")
    for res in results:
        print(f"\nExchange: {res['exchange']}")
        print(f"Setup Type: {res['setup_type']}")
        print(f"BB Score: {res['bb_score']}")
        print(f"BB PCT: {res['bb_pct']}")
        print(f"RSI: {res['rsi']}")
        print(f"Volume Ratio: {res['volume_ratio']}")
        print(f"ATR PCT: {res['atr_pct']}")
        print("-" * 50) 