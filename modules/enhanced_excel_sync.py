#!/usr/bin/env python3
"""
Enhanced Excel to Database Sync with Proper Secondary Sheet Parsing
Based on Claude's superior approach with structured tables
"""

import pandas as pd
import json
import os
import re
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_logger import TradeLogger

class EnhancedExcelSync:
    def __init__(self):
        self.logger = TradeLogger()
        
    def find_latest_excel(self, scanner_type='bb'):
        """Find the most recent Excel file from the scanner"""
        excel_dir = Path('outputs/excel_reports')
        pattern = f"{scanner_type}_analysis_*.xlsx"
        
        excel_files = list(excel_dir.glob(pattern))
        if not excel_files:
            print(f"No Excel files found matching {pattern}")
            return None
            
        latest_file = max(excel_files, key=os.path.getctime)
        print(f"Found latest Excel: {latest_file}")
        return latest_file
        
    def sync_excel_to_database(self, excel_path=None):
        """Complete sync of all Excel sheets to database"""
        
        # Find Excel file if not provided
        if excel_path is None:
            excel_path = self.find_latest_excel()
            if excel_path is None:
                return False
        
        try:
            # Read ALL sheets from Excel
            excel_file = pd.ExcelFile(excel_path)
            print(f"Excel sheets found: {excel_file.sheet_names}")
            
            # Start database logging
            if not self.logger.connection:
                print("❌ Database connection failed")
                return False
                
            # Create scan record
            scan_id = self.logger.log_scan_start('bb_scanner', version='2.0')
            print(f"Started enhanced database sync with scan_id: {scan_id}")
            
            # 1. SYNC TRADE OPPORTUNITIES (All_Analysis sheet)
            trades_synced = self._sync_trades(excel_path, scan_id)
            
            # 2. SYNC MARKET REGIME
            if 'Market_Regime_Analysis' in excel_file.sheet_names:
                self._sync_market_regime(excel_path, scan_id)
            
            # 3. SYNC MARKET OVERVIEW
            if 'Market_Overview' in excel_file.sheet_names:
                self._sync_market_overview(excel_path, scan_id)
            
            # 4. SYNC MARKET METADATA
            if 'Market_Metadata' in excel_file.sheet_names:
                self._sync_market_metadata(excel_path, scan_id)
            
            # Complete the scan
            high_prob_count = len([t for t in trades_synced if t > 70])
            self.logger.complete_scan(scan_id, len(trades_synced), high_prob_count, 0)
            
            print(f"✅ Enhanced sync finished for scan {scan_id}")
            print(f"   - Trades: {len(trades_synced)}")
            print(f"   - Market context: All sheets synced with structured data")
            
            self.logger.close()
            return True
            
        except Exception as e:
            print(f"❌ Enhanced sync failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _sync_trades(self, excel_path, scan_id):
        """Sync trade opportunities from All_Analysis sheet"""
        try:
            df = pd.read_excel(excel_path, sheet_name=0)
            print(f"Found {len(df)} total trades in Excel")
            
            # Keep ALL trades (including 0% probability for ML training)
            trades_df = df.copy()
            
            success_count = 0
            probabilities = []
            
            for idx, row in trades_df.iterrows():
                try:
                    trade_data = row.to_dict()
                    
                    # Clean NaN values
                    for key, value in trade_data.items():
                        if pd.isna(value):
                            trade_data[key] = None
                    
                    # Main columns (Excel column names)
                    main_columns = [
                        'symbol', 'exchange', 'probability', 
                        'entry', 'stop', 'target1'
                    ]
                    
                    # Everything else in scanner_specific_data
                    scanner_specific = {}
                    for key, value in trade_data.items():
                        if key not in main_columns:
                            if value is not None:
                                if isinstance(value, (np.integer, np.floating)):
                                    value = float(value)
                                elif isinstance(value, np.ndarray):
                                    value = value.tolist()
                                elif isinstance(value, (pd.Timestamp, datetime)):
                                    value = value.isoformat()
                                scanner_specific[key] = value
                    
                    # Create record
                    record = {
                        'symbol': str(trade_data.get('symbol', '')),
                        'exchange': str(trade_data.get('exchange', '')),
                        'probability': float(trade_data.get('probability', 0)),
                        'entry_price': float(trade_data.get('entry', 0)) if trade_data.get('entry') else 0,
                        'stop_loss': float(trade_data.get('stop', 0)) if trade_data.get('stop') else 0,
                        'target_1': float(trade_data.get('target1', 0)) if trade_data.get('target1') else 0,
                        'scanner_specific_data': json.dumps(scanner_specific, default=str)
                    }
                    
                    self.logger.log_trade_opportunity(scan_id, record)
                    success_count += 1
                    probabilities.append(record['probability'])
                    
                    if success_count == 1:
                        print(f"First trade: {record['symbol']} with {len(scanner_specific)} extra fields")
                        
                except Exception as e:
                    print(f"Error logging trade {idx}: {e}")
                    continue
            
            print(f"✅ Synced {success_count} trades (including {len([p for p in probabilities if p == 0])} with 0% probability)")
            return probabilities
            
        except Exception as e:
            print(f"Error syncing trades: {e}")
            return []
    
    def _parse_market_regime(self, df):
        """Parse Market Regime Analysis sheet with proper extraction"""
        regime_data = {}
        
        try:
            # Parse the specific format
            for idx, row in df.iterrows():
                if pd.notna(row.iloc[0]):
                    metric = str(row.iloc[0]).strip()
                    
                    # Extract key metrics using regex patterns
                    
                    # Fear & Greed Index
                    fg_match = re.search(r'F&G (\d+)', metric)
                    if fg_match:
                        regime_data['fear_greed_index'] = int(fg_match.group(1))
                    
                    # BTC Dominance
                    btc_match = re.search(r'BTC Dom (\d+\.?\d*)', metric)
                    if btc_match:
                        regime_data['btc_dominance'] = float(btc_match.group(1))
                    
                    # Market Health Score
                    health_match = re.search(r'Market Health Score (\d+\.?\d*)%', metric)
                    if health_match:
                        regime_data['market_health_score'] = float(health_match.group(1))
                    
                    # Regime Type
                    if 'BULLISH' in metric:
                        regime_data['regime_type'] = 'BULLISH'
                    elif 'BEARISH' in metric:
                        regime_data['regime_type'] = 'BEARISH'
                    elif 'MIXED' in metric:
                        regime_data['regime_type'] = 'MIXED'
                    
                    # Regime Confidence
                    conf_match = re.search(r'(\d+\.?\d*)% confidence', metric)
                    if conf_match:
                        regime_data['regime_confidence'] = float(conf_match.group(1))
                    
                    # Position Multiplier
                    pos_match = re.search(r'Position Multiplier: (\d+\.?\d*)', metric)
                    if pos_match:
                        regime_data['position_multiplier'] = float(pos_match.group(1))
                    
                    # Alt Season Indicator
                    if 'Alt Season' in metric:
                        regime_data['alt_season_indicator'] = True
                    
                    # Store the full metric for JSON
                    if 'regime_data' not in regime_data:
                        regime_data['regime_data'] = {}
                    regime_data['regime_data'][f'metric_{idx}'] = metric
            
            print(f"📊 Parsed regime data: {len(regime_data)} fields")
            return regime_data
            
        except Exception as e:
            print(f"Error parsing market regime: {e}")
            return {}
    
    def _sync_market_regime(self, excel_path, scan_id):
        """Sync Market Regime Analysis sheet to structured table"""
        try:
            df = pd.read_excel(excel_path, sheet_name='Market_Regime_Analysis')
            print(f"Syncing Market Regime data...")
            
            # Parse the regime data
            regime_data = self._parse_market_regime(df)
            
            # Extract structured fields
            btc_dominance = regime_data.get('btc_dominance')
            fear_greed_index = regime_data.get('fear_greed_index')
            alt_season_indicator = regime_data.get('alt_season_indicator', False)
            market_health_score = regime_data.get('market_health_score')
            regime_type = regime_data.get('regime_type')
            regime_confidence = regime_data.get('regime_confidence')
            position_multiplier = regime_data.get('position_multiplier', 1.0)
            
            # Store in enhanced table
            self.logger.cursor.execute("""
                INSERT INTO market_regime 
                (scan_id, btc_dominance, fear_greed_index, alt_season_indicator,
                 market_health_score, regime_type, regime_confidence, 
                 position_multiplier, regime_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (scan_id, btc_dominance, fear_greed_index, alt_season_indicator,
                  market_health_score, regime_type, regime_confidence,
                  position_multiplier, json.dumps(regime_data.get('regime_data', {}), default=str)))
            
            regime_id = self.logger.cursor.fetchone()['id']
            self.logger.connection.commit()
            
            print(f"✅ Market regime logged with ID: {regime_id}")
            print(f"   - Fear & Greed: {fear_greed_index}")
            print(f"   - BTC Dominance: {btc_dominance}%")
            print(f"   - Market Health: {market_health_score}%")
            print(f"   - Regime: {regime_type} ({regime_confidence}% confidence)")
            
        except Exception as e:
            print(f"Error syncing market regime: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_market_overview(self, df):
        """Parse Market Overview sheet with proper extraction"""
        overview_data = {}
        
        try:
            for idx, row in df.iterrows():
                if pd.notna(row.iloc[0]):
                    metric = str(row.iloc[0]).strip()
                    value = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else None
                    
                    # Extract key metrics
                    if 'Total BB Bounces Analyzed' in metric:
                        overview_data['total_bounces'] = int(value) if value else 0
                    elif 'Coins Successfully Analyzed' in metric:
                        overview_data['coins_analyzed'] = int(value) if value else 0
                    elif 'Overall Success Rate' in metric:
                        overview_data['overall_success_rate'] = float(str(value).replace('%', '')) if value else 0
                    elif 'Market Health Score' in metric:
                        overview_data['market_health_score'] = float(str(value).replace('%', '')) if value else 0
                    elif 'BB Squeeze Effectiveness' in metric:
                        overview_data['bb_squeeze_effectiveness'] = float(str(value).replace('%', '')) if value else 0
                    elif 'BB Expansion Effectiveness' in metric:
                        overview_data['bb_expansion_effectiveness'] = float(str(value).replace('%', '')) if value else 0
                    
                    # Store all data for JSON
                    if 'overview_data' not in overview_data:
                        overview_data['overview_data'] = {}
                    overview_data['overview_data'][f'metric_{idx}'] = {
                        'metric': metric,
                        'value': str(value) if value is not None else None
                    }
            
            print(f"📊 Parsed overview data: {len(overview_data)} fields")
            print(f"   • Total bounces: {overview_data.get('total_bounces', 'N/A')}")
            print(f"   • Coins analyzed: {overview_data.get('coins_analyzed', 'N/A')}")
            print(f"   • Success rate: {overview_data.get('overall_success_rate', 'N/A')}%")
            return overview_data
            
        except Exception as e:
            print(f"Error parsing market overview: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _sync_market_overview(self, excel_path, scan_id):
        """Sync Market Overview sheet to structured table"""
        try:
            df = pd.read_excel(excel_path, sheet_name='Market_Overview')
            print(f"Syncing Market Overview data...")
            
            # Parse the overview data
            overview_data = self._parse_market_overview(df)
            
            # Extract structured fields
            total_bounces = overview_data.get('total_bounces', 0)
            coins_analyzed = overview_data.get('coins_analyzed', 0)
            overall_success_rate = overview_data.get('overall_success_rate', 0)
            market_health_score = overview_data.get('market_health_score', 0)
            bb_squeeze_effectiveness = overview_data.get('bb_squeeze_effectiveness', 0)
            bb_expansion_effectiveness = overview_data.get('bb_expansion_effectiveness', 0)
            
            # Store in enhanced table
            self.logger.cursor.execute("""
                INSERT INTO market_overview 
                (scan_id, total_bounces, coins_analyzed, overall_success_rate,
                 bb_squeeze_effectiveness, bb_expansion_effectiveness,
                 overview_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (scan_id, total_bounces, coins_analyzed, overall_success_rate,
                  bb_squeeze_effectiveness, bb_expansion_effectiveness,
                  json.dumps(overview_data.get('overview_data', {}), default=str)))
            
            overview_id = self.logger.cursor.fetchone()['id']
            self.logger.connection.commit()
            
            print(f"✅ Market overview logged with ID: {overview_id}")
            print(f"   - Total bounces: {total_bounces}")
            print(f"   - Coins analyzed: {coins_analyzed}")
            print(f"   - Success rate: {overall_success_rate}%")
            print(f"   - BB squeeze effectiveness: {bb_squeeze_effectiveness}%")
            
        except Exception as e:
            print(f"Error syncing market overview: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_market_metadata(self, df):
        """Parse Market Metadata sheet with proper extraction"""
        metadata = {
            'market_cap_distribution': {},
            'sector_performance': {},
            'tier_performance': {},
            'liquidity_analysis': {}
        }
        
        try:
            current_section = None
            for idx, row in df.iterrows():
                if pd.notna(row.iloc[0]):
                    header = str(row.iloc[0]).strip()
                    value = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else None
                    
                    # Identify sections
                    if 'MARKET CAP DISTRIBUTION' in header:
                        current_section = 'market_cap_distribution'
                    elif 'SECTOR DISTRIBUTION' in header:
                        current_section = 'sector_performance'
                    elif 'PERFORMANCE' in header.upper() and 'TIER' in header.upper():
                        current_section = 'tier_performance'
                    elif 'LIQUIDITY' in header.upper():
                        current_section = 'liquidity_analysis'
                    
                    # Parse data within sections
                    elif current_section and value is not None and header != 'Tier' and header != 'Sector':
                        # Extract count from "X trades" format
                        if isinstance(value, str) and 'trades' in value:
                            count = int(value.split()[0])
                        else:
                            count = int(value) if isinstance(value, (int, float)) else 0
                        
                        # Count market cap tiers
                        if 'large_cap' in header.lower():
                            metadata['market_cap_distribution']['large_cap'] = count
                        elif 'mid_cap' in header.lower():
                            metadata['market_cap_distribution']['mid_cap'] = count
                        elif 'small_cap' in header.lower():
                            metadata['market_cap_distribution']['small_cap'] = count
                        elif 'micro_cap' in header.lower():
                            metadata['market_cap_distribution']['micro_cap'] = count
                        
                        # Count sectors
                        elif 'layer1' in header.lower():
                            metadata['sector_performance']['layer1'] = count
                        elif 'defi' in header.lower():
                            metadata['sector_performance']['defi'] = count
                        elif 'layer2' in header.lower():
                            metadata['sector_performance']['layer2'] = count
                        elif 'other' in header.lower():
                            metadata['sector_performance']['other'] = count
                        
                        # Store in appropriate section
                        if current_section in metadata:
                            metadata[current_section][header.lower().replace(' ', '_')] = value
            
            print(f"📊 Parsed metadata: {len(metadata)} sections")
            print(f"   • Market cap distribution: {metadata['market_cap_distribution']}")
            print(f"   • Sector performance: {metadata['sector_performance']}")
            return metadata
            
        except Exception as e:
            print(f"Error parsing market metadata: {e}")
            import traceback
            traceback.print_exc()
            return metadata
    
    def _sync_market_metadata(self, excel_path, scan_id):
        """Sync Market Metadata sheet to structured table"""
        try:
            df = pd.read_excel(excel_path, sheet_name='Market_Metadata')
            print(f"Syncing Market Metadata...")
            
            # Parse the metadata
            metadata = self._parse_market_metadata(df)
            
            # Extract structured fields
            tier_counts = metadata.get('market_cap_distribution', {})
            large_cap_count = tier_counts.get('large_cap', 0)
            mid_cap_count = tier_counts.get('mid_cap', 0)
            small_cap_count = tier_counts.get('small_cap', 0)
            micro_cap_count = tier_counts.get('micro_cap', 0)
            
            # Store in enhanced table
            self.logger.cursor.execute("""
                INSERT INTO market_metadata 
                (scan_id, large_cap_count, mid_cap_count, 
                 small_cap_count, micro_cap_count,
                 sector_performance, market_cap_tier_performance,
                 liquidity_analysis, metadata_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (scan_id, large_cap_count, mid_cap_count, small_cap_count, micro_cap_count,
                  json.dumps(metadata.get('sector_performance', {}), default=str),
                  json.dumps(metadata.get('tier_performance', {}), default=str),
                  json.dumps(metadata.get('liquidity_analysis', {}), default=str),
                  json.dumps(metadata, default=str)))
            
            metadata_id = self.logger.cursor.fetchone()['id']
            self.logger.connection.commit()
            
            print(f"✅ Market metadata logged with ID: {metadata_id}")
            print(f"   - Large cap: {large_cap_count}")
            print(f"   - Mid cap: {mid_cap_count}")
            print(f"   - Small cap: {small_cap_count}")
            print(f"   - Micro cap: {micro_cap_count}")
            
        except Exception as e:
            print(f"Error syncing market metadata: {e}")
            import traceback
            traceback.print_exc()

    def run_after_scan(self):
        """Run this after BB scanner completes"""
        print("="*60)
        print("ENHANCED EXCEL TO DATABASE SYNC")
        print("="*60)
        
        success = self.sync_excel_to_database()
        
        if success:
            print("✅ Enhanced database sync successful!")
            print("   - All trades captured with structured data")
            print("   - Market regime logged with key metrics")
            print("   - Market overview logged with performance stats")
            print("   - Market metadata logged with sector/tier data")
        else:
            print("❌ Enhanced database sync failed - check logs")
        
        return success

if __name__ == "__main__":
    syncer = EnhancedExcelSync()
    syncer.run_after_scan()
