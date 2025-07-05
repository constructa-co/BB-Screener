"""
Pattern Performance Tracker - ML Data Collection System
Tracks pattern outcomes for machine learning and performance optimization
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class PatternPerformanceTracker:
    """
    Pattern performance tracking system for ML preparation
    Collects comprehensive data on pattern outcomes and success rates
    """
    
    def __init__(self, db_path: str = "outputs/pattern_performance.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking parameters
        self.tracking_params = {
            'outcome_check_hours': [24, 72, 168],  # 1 day, 3 days, 1 week
            'success_threshold': 0.02,             # 2% minimum move for success
            'max_tracking_days': 30,               # Maximum days to track outcomes
            'min_data_points': 50,                 # Minimum data points for statistics
            'confidence_threshold': 0.95           # Statistical confidence level
        }
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info(f"Pattern Performance Tracker initialized with database: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Pattern detections table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS pattern_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        exchange TEXT,
                        timeframe TEXT DEFAULT '4H',
                        
                        -- Pattern Information
                        pattern_name TEXT NOT NULL,
                        pattern_tier INTEGER,
                        pattern_direction TEXT,
                        pattern_quality REAL,
                        pattern_confidence REAL,
                        
                        -- Market Context
                        current_price REAL NOT NULL,
                        bb_position REAL,
                        atr_multiple REAL,
                        volume_multiplier REAL,
                        market_regime TEXT,
                        regime_confidence REAL,
                        
                        -- Filter Results
                        atr_significant BOOLEAN,
                        volume_confirmed BOOLEAN,
                        filters_passed BOOLEAN,
                        filter_score REAL,
                        
                        -- Risk/Reward Data
                        auto_sl REAL,
                        auto_tp REAL,
                        risk_reward_ratio REAL,
                        sl_method TEXT,
                        tp_method TEXT,
                        
                        -- Pattern Clustering
                        has_clustering BOOLEAN,
                        cluster_count INTEGER,
                        clustered_patterns TEXT,
                        
                        -- BB Analysis Context
                        bb_probability REAL,
                        bb_score INTEGER,
                        bb_setup_type TEXT,
                        
                        -- Outcome Tracking (Updated later)
                        outcome_1d TEXT,     -- 'SUCCESS', 'FAILURE', 'PENDING', 'NEUTRAL'
                        outcome_3d TEXT,
                        outcome_7d TEXT,
                        
                        price_1d REAL,
                        price_3d REAL,
                        price_7d REAL,
                        
                        max_favorable_1d REAL,
                        max_adverse_1d REAL,
                        max_favorable_3d REAL,
                        max_adverse_3d REAL,
                        max_favorable_7d REAL,
                        max_adverse_7d REAL,
                        
                        -- Performance Metrics
                        pattern_boost_applied REAL,
                        final_confidence REAL,
                        trade_taken BOOLEAN DEFAULT FALSE,
                        trade_outcome TEXT,
                        
                        -- Metadata
                        data_quality INTEGER DEFAULT 5,  -- 1-10 scale
                        notes TEXT
                    )
                ''')
                
                # Pattern statistics summary table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS pattern_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_name TEXT NOT NULL,
                        pattern_tier INTEGER,
                        timeframe TEXT DEFAULT '4H',
                        
                        -- Sample Size
                        total_detections INTEGER DEFAULT 0,
                        filtered_detections INTEGER DEFAULT 0,
                        
                        -- Success Rates
                        success_rate_1d REAL DEFAULT 0,
                        success_rate_3d REAL DEFAULT 0,
                        success_rate_7d REAL DEFAULT 0,
                        
                        -- Performance Metrics
                        avg_quality_score REAL DEFAULT 0,
                        avg_rr_ratio REAL DEFAULT 0,
                        avg_favorable_move REAL DEFAULT 0,
                        avg_adverse_move REAL DEFAULT 0,
                        
                        -- Market Context Analysis
                        best_regime TEXT,
                        best_regime_success_rate REAL DEFAULT 0,
                        worst_regime TEXT,
                        worst_regime_success_rate REAL DEFAULT 0,
                        
                        -- Filter Effectiveness
                        filter_improvement REAL DEFAULT 0,  -- Improvement from filtering
                        optimal_quality_threshold REAL DEFAULT 60,
                        
                        -- Timestamps
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        calculation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Statistical Confidence
                        confidence_level REAL DEFAULT 0,
                        sample_adequate BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Market regime performance table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS regime_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        regime_type TEXT NOT NULL,
                        regime_confidence_range TEXT,  -- 'HIGH', 'MEDIUM', 'LOW'
                        
                        -- Pattern Performance by Regime
                        total_patterns INTEGER DEFAULT 0,
                        successful_patterns INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0,
                        
                        -- Best/Worst Patterns in this Regime
                        best_pattern_type TEXT,
                        best_pattern_success_rate REAL DEFAULT 0,
                        worst_pattern_type TEXT,
                        worst_pattern_success_rate REAL DEFAULT 0,
                        
                        -- Risk/Reward in this Regime
                        avg_rr_ratio REAL DEFAULT 0,
                        avg_risk_percent REAL DEFAULT 0,
                        avg_reward_percent REAL DEFAULT 0,
                        
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_pattern_symbol ON pattern_detections(pattern_name, symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON pattern_detections(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_outcome ON pattern_detections(outcome_1d, outcome_3d, outcome_7d)')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def log_pattern_detection(self, symbol: str, all_patterns: List[Dict], 
                             filtered_patterns: List[Dict], bb_data: Dict,
                             market_regime: Optional[Dict] = None) -> bool:
        """
        Log pattern detection for future performance tracking
        
        Args:
            symbol: Trading symbol
            all_patterns: All detected patterns
            filtered_patterns: Patterns that passed filters
            bb_data: BB analysis context
            market_regime: Market regime data
            
        Returns:
            Success status
        """
        try:
            if not all_patterns:
                return True  # Nothing to log
            
            with sqlite3.connect(self.db_path) as conn:
                for pattern in all_patterns:
                    # Determine if this pattern passed filters
                    passed_filters = any(fp['pattern_name'] == pattern['pattern_name'] 
                                       for fp in filtered_patterns)
                    
                    # Extract pattern data
                    pattern_data = pattern.get('pattern_data', {})
                    
                    # Prepare data for insertion
                    insert_data = {
                        'symbol': symbol,
                        'exchange': bb_data.get('exchange', 'unknown'),
                        'pattern_name': pattern['pattern_name'],
                        'pattern_tier': int(pattern['tier'].replace('tier_', '')),
                        'pattern_direction': pattern_data.get('direction', 'neutral'),
                        'pattern_quality': pattern['quality_score'],
                        'pattern_confidence': pattern_data.get('confidence', 0),
                        
                        # Market Context
                        'current_price': bb_data.get('current_price', 0),
                        'bb_position': bb_data.get('bb_percent', 0),
                        'atr_multiple': pattern_data.get('atr_multiple', 0),
                        'volume_multiplier': pattern_data.get('volume_multiplier', 1),
                        'market_regime': market_regime.get('regime_type', 'UNKNOWN') if market_regime else 'UNKNOWN',
                        'regime_confidence': market_regime.get('confidence', 0) if market_regime else 0,
                        
                        # Filter Results
                        'atr_significant': pattern_data.get('atr_multiple', 0) >= 1.5,
                        'volume_confirmed': pattern_data.get('volume_confirmed', False),
                        'filters_passed': passed_filters,
                        'filter_score': pattern.get('filter_score', 0),
                        
                        # Risk/Reward (if available)
                        'auto_sl': pattern_data.get('auto_sl', 0),
                        'auto_tp': pattern_data.get('auto_tp', 0),
                        'risk_reward_ratio': pattern_data.get('risk_reward_ratio', 0),
                        'sl_method': pattern_data.get('sl_method', 'unknown'),
                        'tp_method': pattern_data.get('tp_method', 'unknown'),
                        
                        # Pattern Clustering
                        'has_clustering': pattern.get('has_clustering', False),
                        'cluster_count': pattern.get('cluster_count', 1),
                        'clustered_patterns': json.dumps(pattern.get('clustered_patterns', [])),
                        
                        # BB Context
                        'bb_probability': bb_data.get('probability', 0),
                        'bb_score': bb_data.get('bb_score', 0),
                        'bb_setup_type': bb_data.get('setup_type', 'UNKNOWN'),
                        
                        # Performance tracking (initially pending)
                        'outcome_1d': 'PENDING',
                        'outcome_3d': 'PENDING',
                        'outcome_7d': 'PENDING',
                        
                        # Pattern boost
                        'pattern_boost_applied': pattern.get('pattern_boost', 0),
                        'final_confidence': bb_data.get('final_confidence', bb_data.get('probability', 0))
                    }
                    
                    # Insert into database
                    placeholders = ', '.join(['?' for _ in insert_data])
                    columns = ', '.join(insert_data.keys())
                    values = list(insert_data.values())
                    
                    conn.execute(f'''
                        INSERT INTO pattern_detections ({columns})
                        VALUES ({placeholders})
                    ''', values)
                
                conn.commit()
                self.logger.debug(f"Logged {len(all_patterns)} pattern detections for {symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging pattern detection: {str(e)}")
            return False
    
    def update_pattern_outcomes(self, hours_since: int = 24) -> int:
        """
        Update pattern outcomes by checking current prices
        
        Args:
            hours_since: Hours since detection to check
            
        Returns:
            Number of patterns updated
        """
        try:
            # This would normally fetch current prices and update outcomes
            # For now, return placeholder implementation
            
            cutoff_time = datetime.now() - timedelta(hours=hours_since)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get pending patterns from the specified time period
                cursor = conn.execute('''
                    SELECT id, symbol, pattern_name, pattern_direction, current_price, 
                           auto_sl, auto_tp, timestamp
                    FROM pattern_detections 
                    WHERE timestamp <= ? AND (
                        (? = 24 AND outcome_1d = 'PENDING') OR
                        (? = 72 AND outcome_3d = 'PENDING') OR  
                        (? = 168 AND outcome_7d = 'PENDING')
                    )
                ''', (cutoff_time, hours_since, hours_since, hours_since))
                
                patterns_to_update = cursor.fetchall()
                
                # In a real implementation, would fetch current prices here
                # For now, simulate outcomes for demonstration
                updated_count = 0
                
                for pattern in patterns_to_update:
                    pattern_id, symbol, pattern_name, direction, entry_price = pattern[:5]
                    
                    # Simulate price movement (in real implementation, fetch actual prices)
                    simulated_outcome = self._simulate_pattern_outcome(pattern_name, direction)
                    
                    # Determine which outcome column to update
                    if hours_since == 24:
                        outcome_col = 'outcome_1d'
                        price_col = 'price_1d'
                    elif hours_since == 72:
                        outcome_col = 'outcome_3d'
                        price_col = 'price_3d'
                    else:
                        outcome_col = 'outcome_7d'
                        price_col = 'price_7d'
                    
                    # Update the pattern outcome
                    conn.execute(f'''
                        UPDATE pattern_detections 
                        SET {outcome_col} = ?, {price_col} = ?
                        WHERE id = ?
                    ''', (simulated_outcome['status'], simulated_outcome['price'], pattern_id))
                    
                    updated_count += 1
                
                conn.commit()
                self.logger.info(f"Updated outcomes for {updated_count} patterns ({hours_since}h period)")
                return updated_count
                
        except Exception as e:
            self.logger.error(f"Error updating pattern outcomes: {str(e)}")
            return 0
    
    def _simulate_pattern_outcome(self, pattern_name: str, direction: str) -> Dict:
        """Simulate pattern outcome for demonstration (replace with real price fetching)"""
        # Simulate success rates based on pattern tier and direction
        tier_success_rates = {'tier_1': 0.65, 'tier_2': 0.55, 'tier_3': 0.45}
        
        # Get pattern tier (simplified lookup)
        if pattern_name in ['hammer', 'engulfing', 'doji']:
            base_success_rate = tier_success_rates['tier_1']
        elif pattern_name in ['morning_star', 'harami']:
            base_success_rate = tier_success_rates['tier_2'] 
        else:
            base_success_rate = tier_success_rates['tier_3']
        
        # Random outcome based on success rate
        success = np.random.random() < base_success_rate
        
        return {
            'status': 'SUCCESS' if success else 'FAILURE',
            'price': np.random.uniform(0.98, 1.05)  # Simulated price ratio
        }
    
    def calculate_pattern_statistics(self, pattern_name: Optional[str] = None) -> Dict:
        """
        Calculate comprehensive pattern performance statistics
        
        Args:
            pattern_name: Specific pattern to analyze (None for all patterns)
            
        Returns:
            Comprehensive statistics dictionary
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build query condition
                where_clause = "WHERE pattern_name = ?" if pattern_name else ""
                params = [pattern_name] if pattern_name else []
                
                # Get basic statistics
                query = f'''
                    SELECT 
                        pattern_name,
                        pattern_tier,
                        COUNT(*) as total_detections,
                        COUNT(CASE WHEN filters_passed = 1 THEN 1 END) as filtered_detections,
                        AVG(pattern_quality) as avg_quality,
                        AVG(risk_reward_ratio) as avg_rr_ratio,
                        
                        -- Success rates
                        AVG(CASE WHEN outcome_1d = 'SUCCESS' THEN 1.0 ELSE 0.0 END) as success_rate_1d,
                        AVG(CASE WHEN outcome_3d = 'SUCCESS' THEN 1.0 ELSE 0.0 END) as success_rate_3d,
                        AVG(CASE WHEN outcome_7d = 'SUCCESS' THEN 1.0 ELSE 0.0 END) as success_rate_7d,
                        
                        -- Filter effectiveness
                        AVG(CASE WHEN filters_passed = 1 AND outcome_1d = 'SUCCESS' THEN 1.0 ELSE 0.0 END) as filtered_success_rate
                        
                    FROM pattern_detections 
                    {where_clause}
                    GROUP BY pattern_name, pattern_tier
                '''
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return {'error': 'No data available for analysis'}
                
                # Calculate additional metrics
                statistics = {}
                
                for _, row in df.iterrows():
                    pattern_stats = {
                        'pattern_name': row['pattern_name'],
                        'pattern_tier': row['pattern_tier'],
                        'sample_size': {
                            'total_detections': row['total_detections'],
                            'filtered_detections': row['filtered_detections'],
                            'sample_adequate': row['total_detections'] >= self.tracking_params['min_data_points']
                        },
                        'success_rates': {
                            '1_day': round(row['success_rate_1d'], 3),
                            '3_day': round(row['success_rate_3d'], 3),
                            '7_day': round(row['success_rate_7d'], 3)
                        },
                        'quality_metrics': {
                            'avg_quality_score': round(row['avg_quality'], 1),
                            'avg_rr_ratio': round(row['avg_rr_ratio'], 2)
                        },
                        'filter_effectiveness': {
                            'unfiltered_success': round(row['success_rate_1d'], 3),
                            'filtered_success': round(row['filtered_success_rate'], 3),
                            'improvement': round(row['filtered_success_rate'] - row['success_rate_1d'], 3)
                        }
                    }
                    
                    statistics[row['pattern_name']] = pattern_stats
                
                return statistics
                
        except Exception as e:
            self.logger.error(f"Error calculating pattern statistics: {str(e)}")
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary for monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Overall statistics
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_patterns,
                        COUNT(CASE WHEN filters_passed = 1 THEN 1 END) as filtered_patterns,
                        COUNT(CASE WHEN outcome_1d != 'PENDING' THEN 1 END) as evaluated_patterns,
                        AVG(CASE WHEN outcome_1d = 'SUCCESS' THEN 1.0 ELSE 0.0 END) as overall_success_rate,
                        COUNT(DISTINCT pattern_name) as unique_patterns,
                        COUNT(DISTINCT symbol) as unique_symbols
                    FROM pattern_detections
                ''')
                
                overall_stats = cursor.fetchone()
                
                # Best and worst performing patterns
                cursor = conn.execute('''
                    SELECT 
                        pattern_name,
                        COUNT(*) as count,
                        AVG(CASE WHEN outcome_1d = 'SUCCESS' THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM pattern_detections 
                    WHERE outcome_1d != 'PENDING'
                    GROUP BY pattern_name
                    HAVING COUNT(*) >= 10
                    ORDER BY success_rate DESC
                ''')
                
                pattern_performance = cursor.fetchall()
                
                return {
                    'database_path': str(self.db_path),
                    'total_patterns_logged': overall_stats[0],
                    'filtered_patterns': overall_stats[1],
                    'evaluated_patterns': overall_stats[2],
                    'overall_success_rate': round(overall_stats[3] or 0, 3),
                    'unique_patterns_detected': overall_stats[4],
                    'unique_symbols_analyzed': overall_stats[5],
                    'best_patterns': pattern_performance[:5] if pattern_performance else [],
                    'data_collection_status': 'Active',
                    'ml_readiness': overall_stats[2] >= self.tracking_params['min_data_points']
                }
                
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {'error': str(e)}
    
    def export_ml_dataset(self, output_path: str = "outputs/ml_dataset.csv") -> bool:
        """
        Export comprehensive dataset for machine learning training
        
        Args:
            output_path: Path for exported CSV file
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export comprehensive dataset
                query = '''
                    SELECT 
                        symbol, pattern_name, pattern_tier, pattern_direction,
                        pattern_quality, pattern_confidence, current_price,
                        bb_position, atr_multiple, volume_multiplier,
                        market_regime, regime_confidence,
                        atr_significant, volume_confirmed, filters_passed,
                        risk_reward_ratio, has_clustering, cluster_count,
                        bb_probability, bb_score, pattern_boost_applied,
                        final_confidence,
                        
                        -- Outcomes
                        outcome_1d, outcome_3d, outcome_7d,
                        
                        -- Success flags for ML
                        CASE WHEN outcome_1d = 'SUCCESS' THEN 1 ELSE 0 END as success_1d,
                        CASE WHEN outcome_3d = 'SUCCESS' THEN 1 ELSE 0 END as success_3d,
                        CASE WHEN outcome_7d = 'SUCCESS' THEN 1 ELSE 0 END as success_7d,
                        
                        timestamp
                    FROM pattern_detections 
                    WHERE outcome_1d != 'PENDING'
                    ORDER BY timestamp DESC
                '''
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    self.logger.warning("No completed pattern data available for ML export")
                    return False
                
                # Save to CSV
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
                
                self.logger.info(f"ML dataset exported: {len(df)} records to {output_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error exporting ML dataset: {str(e)}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old pattern data to manage database size
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    DELETE FROM pattern_detections 
                    WHERE timestamp < ? AND outcome_7d != 'PENDING'
                ''', (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old pattern records")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
            return 0
    
    def get_tracking_statistics(self) -> Dict:
        """Get tracking system statistics"""
        return {
            'database_path': str(self.db_path),
            'tracking_parameters': self.tracking_params,
            'outcome_periods': ['1 day', '3 days', '7 days'],
            'data_retention': '90 days',
            'ml_features': [
                'Pattern characteristics',
                'Market regime context', 
                'Filter results',
                'Risk/reward ratios',
                'Clustering information',
                'BB analysis context',
                'Success outcomes'
            ],
            'export_formats': ['CSV for ML training', 'JSON for analysis'],
            'performance_monitoring': True
        }