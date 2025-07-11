#!/usr/bin/env python3
"""
Quick BB Score Test
==================

This script tests the BB detector directly to see what bb_score values it returns.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the modules directory to the path
sys.path.append('modules')

try:
    from bb_detector import BBDetector
    print("✅ Successfully imported BB detector")
except ImportError as e:
    print(f"❌ Failed to import BB detector: {e}")
    sys.exit(1)

def create_sample_data():
    """Create sample OHLCV data for testing"""
    
    # Create sample data with BB setup
    dates = pd.date_range('2024-01-01', periods=50, freq='4H')
    
    # Create data that should trigger a BB setup
    data = {
        'open': np.random.uniform(100, 110, 50),
        'high': np.random.uniform(105, 115, 50),
        'low': np.random.uniform(95, 105, 50),
        'close': np.random.uniform(100, 110, 50),
        'volume': np.random.uniform(1000, 2000, 50)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Add technical indicators
    df['bb_upper'] = df['close'] * 1.02  # 2% above close
    df['bb_middle'] = df['close'] * 1.00  # At close
    df['bb_lower'] = df['close'] * 0.98   # 2% below close
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Add RSI
    df['rsi'] = np.random.uniform(20, 80, 50)
    
    # Add volume ratio
    df['volume_ratio'] = np.random.uniform(1.0, 2.0, 50)
    
    # Add ATR
    df['atr'] = df['close'] * 0.02  # 2% ATR
    
    # Create a BB setup in the last few candles
    # Make it touch the lower band
    df.loc[df.index[-1], 'low'] = df.loc[df.index[-1], 'bb_lower'] * 0.999
    df.loc[df.index[-1], 'close'] = df.loc[df.index[-1], 'bb_lower'] * 1.001
    df.loc[df.index[-1], 'bb_pct'] = 0.05  # Very low BB position
    df.loc[df.index[-1], 'rsi'] = 25  # Oversold RSI
    df.loc[df.index[-1], 'volume_ratio'] = 1.8  # High volume
    
    return df

def test_bb_detector():
    """Test the BB detector with sample data"""
    
    print("🧪 TESTING BB DETECTOR")
    print("=" * 30)
    
    # Create sample data
    df = create_sample_data()
    print(f"✅ Created sample data with {len(df)} candles")
    
    # Initialize BB detector
    bb_detector = BBDetector()
    print("✅ BB detector initialized")
    
    # Test BB analysis
    print("\n🔍 Testing BB analysis...")
    try:
        bb_analysis = bb_detector.analyze_bb_setup(df)
        print(f"✅ BB analysis successful")
        print(f"   Setup type: {bb_analysis.get('setup_type', 'NOT_FOUND')}")
        print(f"   BB Score: {bb_analysis.get('bb_score', 'NOT_FOUND')}")
        print(f"   Setup quality: {bb_analysis.get('setup_quality', 'NOT_FOUND')}")
        print(f"   Entry: {bb_analysis.get('entry', 'NOT_FOUND')}")
        print(f"   Stop: {bb_analysis.get('stop', 'NOT_FOUND')}")
        print(f"   Target: {bb_analysis.get('target1', 'NOT_FOUND')}")
        print(f"   Risk/Reward: {bb_analysis.get('risk_reward', 'NOT_FOUND')}")
        
        # Check if scoring_details exists
        if 'scoring_details' in bb_analysis:
            scoring = bb_analysis['scoring_details']
            print(f"   Scoring details: {len(scoring.get('breakdown', []))} components")
            print(f"   Tier scores: {scoring.get('tier_scores', {})}")
        else:
            print(f"   No scoring details found")
            
    except Exception as e:
        print(f"❌ BB analysis failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_bb_detector()
    if success:
        print("\n🎉 BB detector test passed!")
    else:
        print("\n❌ BB detector test failed!")
        sys.exit(1) 