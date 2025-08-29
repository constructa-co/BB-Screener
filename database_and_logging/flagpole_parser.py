#!/usr/bin/env python3
"""
Flagpole Pattern Output Parser
Parses the text output from pattern_scanner.py into structured data
"""

import re
from typing import Dict, List, Optional, Any
from decimal import Decimal

class FlagpoleOutputParser:
    """Parse flagpole scanner text output into structured data"""
    
    def __init__(self):
        # Regex patterns for parsing
        self.header_pattern = re.compile(
            r'(\d+)\.\s+([🏁🚩📐])\s+(\w+)\s+\|\s+Score:\s+(\d+)/100\s+\|\s+(.+?)\s+\|\s+([📈📉⚡])'
        )
        self.price_pattern = re.compile(
            r'Current:\s+\$?([\d.]+)\s+\|\s+Breakout:\s+\$?([\d.]+)'
        )
        self.target_pattern = re.compile(
            r'Target:\s+\$?([\d.]+)\s+\|\s+Potential:\s+([\d.]+)%\s+\|\s+R:R:\s+([\d.]+):1'
        )
        self.stop_pattern = re.compile(
            r'Stop:\s+\$?([\d.]+)\s+\|\s+Risk:\s+([\d.]+)%\s+\|\s+Age:\s+(\d+)\s+candles'
        )
        self.metrics_pattern = re.compile(
            r'Pole:\s+([\d.]+)%\s+\|\s+Vol Decline:\s+([\d.]+)%\s+\|\s+Slope:\s+([\d.]+)%'
        )
        self.quality_pattern = re.compile(
            r'Quality:\s+(.+)'
        )
    
    def parse_scanner_output(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse complete scanner output into list of pattern dictionaries
        """
        patterns = []
        lines = output.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for pattern header
            header_match = self.header_pattern.search(line)
            if header_match:
                pattern = self._parse_single_pattern(lines, i)
                if pattern:
                    patterns.append(pattern)
                i += 6  # Skip to next pattern
            else:
                i += 1
        
        return patterns
    
    def _parse_single_pattern(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Parse a single pattern entry starting at given index"""
        
        pattern = {}
        
        # Parse header line
        header_match = self.header_pattern.search(lines[start_idx])
        if not header_match:
            return None
        
        pattern['rank'] = int(header_match.group(1))
        pattern['icon'] = header_match.group(2)
        pattern['symbol'] = header_match.group(3)
        pattern['score'] = int(header_match.group(4))
        pattern['pattern_details'] = header_match.group(5)
        pattern['direction_icon'] = header_match.group(6)
        
        # Extract pattern type and pole percentage
        pattern_match = re.match(r'(\w+)\s+\(([\d.]+)%\s+pole\)', pattern['pattern_details'])
        if pattern_match:
            pattern['pattern_type'] = pattern_match.group(1)
            pattern['pole_pct'] = float(pattern_match.group(2))
        else:
            pattern['pattern_type'] = pattern['pattern_details'].split()[0]
            pattern['pole_pct'] = None
        
        # Determine direction from icon
        if pattern['direction_icon'] == '📈':
            pattern['direction'] = 'Bullish'
        elif pattern['direction_icon'] == '📉':
            pattern['direction'] = 'Bearish'
        else:
            pattern['direction'] = 'Either direction'
        
        # Parse price line (next line)
        if start_idx + 1 < len(lines):
            price_match = self.price_pattern.search(lines[start_idx + 1])
            if price_match:
                pattern['current_price'] = float(price_match.group(1))
                pattern['breakout_level'] = float(price_match.group(2))
        
        # Parse target line
        if start_idx + 2 < len(lines):
            target_match = self.target_pattern.search(lines[start_idx + 2])
            if target_match:
                pattern['target_price'] = float(target_match.group(1))
                pattern['potential_pct'] = float(target_match.group(2))
                pattern['risk_reward'] = float(target_match.group(3))
        
        # Parse stop line
        if start_idx + 3 < len(lines):
            stop_match = self.stop_pattern.search(lines[start_idx + 3])
            if stop_match:
                pattern['stop_loss'] = float(stop_match.group(1))
                pattern['risk_pct'] = float(stop_match.group(2))
                pattern['age_candles'] = int(stop_match.group(3))
        
        # Parse metrics line
        if start_idx + 4 < len(lines):
            metrics_match = self.metrics_pattern.search(lines[start_idx + 4])
            if metrics_match:
                pattern['pole_pct'] = float(metrics_match.group(1))
                pattern['vol_decline_pct'] = float(metrics_match.group(2))
                pattern['slope_pct'] = float(metrics_match.group(3))
        
        # Parse quality line
        if start_idx + 5 < len(lines):
            quality_match = self.quality_pattern.search(lines[start_idx + 5])
            if quality_match:
                quality_str = quality_match.group(1)
                pattern['quality_raw'] = quality_str
                
                # Extract quality indicators
                indicators = []
                pattern['is_ready'] = '⚡ Ready' in quality_str
                pattern['has_strong_vol'] = '📊 Strong Vol' in quality_str
                pattern['has_fast_pole'] = '🚀 Fast Pole' in quality_str
                
                if pattern['is_ready']:
                    indicators.append('Ready')
                if pattern['has_strong_vol']:
                    indicators.append('Strong Vol')
                if pattern['has_fast_pole']:
                    indicators.append('Fast Pole')
                
                pattern['quality_indicators'] = indicators
        
        return pattern
