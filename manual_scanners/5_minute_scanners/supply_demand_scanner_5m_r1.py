#!/usr/bin/env python3
"""
Supply & Demand Scanner R1 - With Database Integration
File: supply_demand_scanner_5m_r1.py
Extends the R0 scanner with database logging capabilities
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the original scanner (R0) to extend it
import importlib.util
spec = importlib.util.spec_from_file_location("base_scanner", "/opt/bb-screener/manual_scanners/5_min_scanners/supply_&_demand_scanner_5m_r0.py")
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
BaseScanner = base_module.SupplyDemandScanner5M

# Import database logger
try:
    from database_and_logging.supply_demand_logger import SupplyDemandLogger
    DB_LOGGING_ENABLED = True
    print("✅ Database logging enabled for Supply & Demand scanner")
except ImportError as e:
    print(f"❌ Database logging not available: {e}")
    DB_LOGGING_ENABLED = False

class SupplyDemandScanner5MR1(SupplyDemandScanner5M):
    """Enhanced Supply & Demand Scanner with database integration"""
    
    def __init__(self):
        super().__init__()
        self.timeframe = '5m'
        
        # Initialize database logger if available
        if DB_LOGGING_ENABLED:
            self.db_logger = SupplyDemandLogger(timeframe=self.timeframe)
        else:
            self.db_logger = None
    
    def identify_zones(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Enhanced zone identification with quality scoring
        Extends the base scanner's zone detection
        """
        # Call parent method to get base zones
        zones = super().detect_aggressive_moves(df)
        
        # Enhance zones with additional metrics for database
        current_price = float(df['close'].iloc[-1])
        
        enhanced_zones = []
        for zone in zones:
            # Calculate distance metrics
            zone_midpoint = (zone['zone_high'] + zone['zone_low']) / 2
            distance = abs(current_price - zone_midpoint)
            distance_pct = (distance / zone_midpoint) * 100
            
            # Determine price position
            if current_price > zone['zone_high']:
                price_position = 'ABOVE'
            elif current_price < zone['zone_low']:
                price_position = 'BELOW'
            else:
                price_position = 'INSIDE'
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(zone, df)
            
            # Add database-specific fields
            enhanced_zone = {
                'symbol': symbol,
                'zone_type': zone['type'].upper(),
                'zone_top': zone['zone_high'],
                'zone_bottom': zone['zone_low'],
                'zone_strength': zone['move_strength'],
                'touch_count': zone.get('tests', 1),
                'current_price': current_price,
                'distance_to_zone': distance,
                'distance_percentage': distance_pct,
                'price_position': price_position,
                'quality_score': quality_score,
                'formation_type': self._determine_formation_type(df, zone),
                'freshness_score': self._calculate_freshness_score(zone, df),
                'reliability_score': 50,  # Default, will be updated by post-mortem
                'zone_volume': zone.get('volume_ratio', 1.0),
                'average_volume': 1.0,  # Will be calculated from df
                'volume_ratio': zone.get('volume_ratio', 1.0),
                'volume_confirmation': zone.get('volume_ratio', 1.0) >= 1.1,
                'formation_candles': zone.get('zone_range_pct', 0),
                'formation_start': zone.get('creation_timestamp'),
                'formation_end': zone.get('creation_timestamp'),
                'algorithm_parameters': {
                    'timeframe': self.timeframe,
                    'min_move_pct': 1.5,  # Lower threshold for 5M
                    'volume_threshold': 1.1,  # Lower volume requirement for 5M
                    'zone_range_min': 0.3,  # Smaller zones for 5M
                    'zone_range_max': 3.0
                }
            }
            
            enhanced_zones.append(enhanced_zone)
        
        return enhanced_zones
    
    def _calculate_quality_score(self, zone: Dict, df: pd.DataFrame) -> int:
        """Calculate comprehensive quality score for a zone"""
        score = 0
        
        # Base score from move strength (max 40 points)
        move_strength = zone.get('move_strength', 0)
        score += min(40, int(move_strength * 10))
        
        # Volume confirmation (max 30 points)
        volume_ratio = zone.get('volume_ratio', 1.0)
        if volume_ratio >= 1.8:
            score += 30
        elif volume_ratio >= 1.4:
            score += 20
        elif volume_ratio >= 1.1:
            score += 10
        
        # Zone size optimization (max 20 points)
        zone_range = zone.get('zone_range_pct', 0)
        if 0.5 <= zone_range <= 2.0:  # Optimal zone size for 5M
            score += 20
        elif 0.3 <= zone_range <= 2.5:
            score += 10
        
        # Freshness bonus (max 10 points)
        if zone.get('creation_index'):
            bars_ago = len(df) - zone['creation_index']
            if bars_ago <= 40:
                score += 10
            elif bars_ago <= 80:
                score += 5
        
        return min(100, int(score))
    
    def _calculate_freshness_score(self, zone: Dict, df: pd.DataFrame) -> int:
        """Calculate how fresh/recent the zone is"""
        if not zone.get('creation_index'):
            return 50
        
        bars_ago = len(df) - zone['creation_index']
        if bars_ago <= 20:
            return 100
        elif bars_ago <= 50:
            return 80
        elif bars_ago <= 100:
            return 60
        elif bars_ago <= 200:
            return 40
        else:
            return 20
    
    def _determine_formation_type(self, df: pd.DataFrame, zone: Dict) -> str:
        """Determine zone formation pattern"""
        if not zone.get('creation_index'):
            return 'UNKNOWN'
        
        idx = zone['creation_index']
        
        # Look at price action before and after zone
        before_start = max(0, idx - 15)
        before_end = max(0, idx - 2)
        after_start = min(len(df), idx + 2)
        after_end = min(len(df), idx + 15)
        
        if before_end > before_start and after_end > after_start:
            before_trend = df['close'].iloc[before_end] - df['close'].iloc[before_start]
            after_trend = df['close'].iloc[after_end] - df['close'].iloc[after_start]
            
            if zone['type'] == 'demand':
                if before_trend < 0 and after_trend > 0:
                    return 'DROP_BASE_RALLY'
                elif before_trend > 0 and after_trend > 0:
                    return 'RALLY_BASE_RALLY'
            else:  # supply
                if before_trend > 0 and after_trend < 0:
                    return 'RALLY_BASE_DROP'
                elif before_trend < 0 and after_trend < 0:
                    return 'DROP_BASE_DROP'
        
        return 'UNKNOWN'
    
    def scan_and_log(self, symbols: Optional[List[str]] = None) -> Dict:
        """Main scanning method with database logging"""
        if symbols is None:
            symbols = self.get_top_coins()  # Use parent's method
        
        print(f"\n🔍 Scanning {len(symbols)} symbols for Supply & Demand zones")
        print(f"⏰ Timeframe: {self.timeframe}")
        print("=" * 60)
        
        all_zones = []
        logged_count = 0
        
        for symbol in symbols:
            try:
                # Get price data using parent's method
                df = self.get_binance_klines(symbol)
                if df is None or len(df) < 50:
                    continue
                
                # Identify zones with enhanced metrics
                zones = self.identify_zones(symbol, df)
                
                if zones:
                    print(f"\n{symbol}: Found {len(zones)} zones")
                    
                    # Log to database if available
                    if self.db_logger:
                        batch_logged = self.db_logger.log_zones_batch(zones)
                        logged_count += batch_logged
                        print(f"  ✅ Logged {batch_logged} zones to database")
                    
                    all_zones.extend(zones)
                
            except Exception as e:
                print(f"❌ Error scanning {symbol}: {e}")
                continue
        
        print(f"\n{'=' * 60}")
        print(f"✅ Scan complete!")
        print(f"   Total zones found: {len(all_zones)}")
        if self.db_logger:
            print(f"   Zones logged to database: {logged_count}")
        
        return {
            'zones': all_zones,
            'total_found': len(all_zones),
            'total_logged': logged_count
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.db_logger:
            self.db_logger.close()

def main():
    """Main execution"""
    scanner = SupplyDemandScanner5MR1()
    results = scanner.scan_and_log()
    
    # Save results to CSV for backup
    if results['zones']:
        df = pd.DataFrame(results['zones'])
        filename = f"sd_zones_{scanner.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to {filename}")

if __name__ == "__main__":
    main()
