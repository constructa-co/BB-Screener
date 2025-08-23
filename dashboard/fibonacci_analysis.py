"""
Fibonacci Retracement Analysis Dashboard Page
Integrates with existing dashboard structure - ADDITION ONLY
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TradeLogger (same as main dashboard)
from trade_logger import TradeLogger

# Timezone handling
def convert_to_uae_time(dt):
    """Convert datetime to UAE timezone (UTC+4)"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume UTC if no timezone info
        dt = dt.replace(tzinfo=timezone.utc)
    # UAE is UTC+4
    uae_offset = timezone(timedelta(hours=4))
    return dt.astimezone(uae_offset)

def format_datetime_for_display(dt):
    """Format datetime for display in UAE timezone"""
    if dt is None:
        return "N/A"
    uae_dt = convert_to_uae_time(dt)
    return uae_dt.strftime('%Y-%m-%d %H:%M:%S UAE')

# Use the same TradeLogger as the main dashboard
def get_db():
    """Get database connection using TradeLogger - matches main dashboard"""
    return TradeLogger()

def load_fibonacci_signals(hours_back=24, min_confidence=0.6):
    """Load Fibonacci signals from database"""
    logger = get_db()
    if not logger.connection:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                symbol,
                timeframe,
                signal_type,
                fibonacci_level,
                price_level,
                current_price,
                confidence_score,
                volume_confirmation,
                momentum_confirmation,
                swing_high,
                swing_low,
                detected_at,
                quality_score,
                move_percentage,
                stop_loss_price,
                entry_timing_status,
                target_1_price,
                target_1_percentage,
                target_1_risk_reward,
                target_2_price,
                target_2_percentage,
                target_2_risk_reward,
                target_3_price,
                target_3_percentage,
                target_3_risk_reward,
                entry_price,
                risk_percentage,
                setup_stage
            FROM other_scanners.fibonacci_signals
            WHERE detected_at > NOW() - INTERVAL %s
                AND confidence_score >= %s
            ORDER BY detected_at DESC
            LIMIT 500
        """
        logger.cursor.execute(query, (f"{hours_back} hours", min_confidence))
        results = logger.cursor.fetchall()
        if results:
            df = pd.DataFrame(results)
            # Convert detected_at timestamps to UAE timezone
            if 'detected_at' in df.columns:
                df['detected_at'] = pd.to_datetime(df['detected_at']).apply(convert_to_uae_time)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Fibonacci signals: {e}")
        return pd.DataFrame()

def calculate_performance_metrics():
    """Calculate Fibonacci scanner performance metrics"""
    logger = get_db()
    if not logger.connection:
        st.error("Database connection failed")
        return None, pd.DataFrame()
    
    try:
        # Overall metrics
        query = """
            SELECT 
                COUNT(*) as total_signals,
                COALESCE(AVG(confidence_score), 0) as avg_confidence,
                COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END) as high_confidence_count,
                COUNT(DISTINCT symbol) as unique_symbols,
                COALESCE(AVG(quality_score), 0) as avg_quality_score
            FROM other_scanners.fibonacci_signals
            WHERE detected_at > NOW() - INTERVAL '24 hours'
        """
        
        logger.cursor.execute(query)
        metrics_row = logger.cursor.fetchone()
        
        if metrics_row and metrics_row['total_signals'] > 0:  # Check if we have signals
            metrics = {
                'total_signals': int(metrics_row['total_signals']) if metrics_row['total_signals'] else 0,
                'avg_confidence': float(metrics_row['avg_confidence']) if metrics_row['avg_confidence'] else 0.0,
                'high_confidence_count': int(metrics_row['high_confidence_count']) if metrics_row['high_confidence_count'] else 0,
                'unique_symbols': int(metrics_row['unique_symbols']) if metrics_row['unique_symbols'] else 0,
                'avg_quality_score': float(metrics_row['avg_quality_score']) if metrics_row['avg_quality_score'] else 0.0
            }
        else:
            # Return default metrics if no data
            metrics = {
                'total_signals': 0,
                'avg_confidence': 0.0,
                'high_confidence_count': 0,
                'unique_symbols': 0,
                'avg_quality_score': 0.0
            }
        
        # Performance by level
        level_query = """
            SELECT 
                fibonacci_level,
                COUNT(*) as signal_count,
                COALESCE(AVG(confidence_score), 0) as avg_confidence,
                signal_type,
                COUNT(DISTINCT symbol) as symbols,
                COALESCE(AVG(quality_score), 0) as avg_quality
            FROM other_scanners.fibonacci_signals
            WHERE detected_at > NOW() - INTERVAL '24 hours'
            GROUP BY fibonacci_level, signal_type
            ORDER BY signal_count DESC
        """
        
        logger.cursor.execute(level_query)
        level_performance = pd.DataFrame(logger.cursor.fetchall())
        
        return metrics, level_performance
    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")
        # Return default metrics on error
        return {
            'total_signals': 0,
            'avg_confidence': 0.0,
            'high_confidence_count': 0,
            'unique_symbols': 0,
            'avg_quality_score': 0.0
        }, pd.DataFrame()

def create_fibonacci_analysis_page():
    """Create the main Fibonacci analysis page"""
    
    # Main page layout
    st.markdown('<h1 class="main-header">📐 Fibonacci Retracement Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Real-time Fibonacci level detection and signal analysis with complete trading data")
    
    # Sidebar filters
    with st.sidebar:
        st.header("📊 Filters")
        
        hours_back = st.slider(
            "Hours to analyze",
            min_value=1,
            max_value=168,
            value=24,
            step=1
        )
        
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.65,
            step=0.05,
            format="%.2f"
        )
        
        signal_types = st.multiselect(
            "Signal Types",
            options=["SUPPORT", "RESISTANCE", "BREAKOUT", "BOUNCE"],
            default=["SUPPORT", "RESISTANCE", "BREAKOUT", "BOUNCE"]
        )
        
        timeframes = st.multiselect(
            "Timeframes",
            options=["1m", "5m", "15m", "1h", "4h", "1d"],
            default=["5m", "1h"]
        )

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    metrics, level_performance = calculate_performance_metrics()

    if metrics and metrics['total_signals'] > 0:
        with col1:
            st.metric(
                "Total Signals (24h)",
                f"{metrics['total_signals']:,}",
                delta=None
            )

        with col2:
            st.metric(
                "Avg Confidence",
                f"{metrics['avg_confidence']:.1%}",
                delta=f"{(metrics['avg_confidence'] - 0.65):.1%}"
            )

        with col3:
            st.metric(
                "High Confidence",
                f"{metrics['high_confidence_count']:,}",
                delta=f"{(metrics['high_confidence_count']/max(metrics['total_signals'],1)):.1%}"
            )

        with col4:
            st.metric(
                "Avg Quality Score",
                f"{metrics['avg_quality_score']:.0f}/100",
                delta=None
            )
    else:
        st.info("No recent Fibonacci signals in the last 24 hours. The scanner may be running but no signals detected yet.")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Signals", "📊 Performance Analysis", "🎯 Top Opportunities", "📉 Historical Patterns"])

    with tab1:
        st.header("Live Fibonacci Signals")
        
        # Load signals
        signals_df = load_fibonacci_signals(hours_back, min_confidence)
        
        if not signals_df.empty:
            # Filter by selected types and timeframes
            signals_df = signals_df[
                (signals_df['signal_type'].isin(signal_types)) &
                (signals_df['timeframe'].isin(timeframes))
            ]
            
            # Display enhanced signals table with all trading data
            display_columns = [
                'symbol', 'signal_type', 'fibonacci_level', 'confidence_score', 
                'quality_score', 'entry_timing_status', 'setup_stage',
                'target_1_price', 'target_1_percentage', 'target_1_risk_reward',
                'target_2_price', 'target_2_percentage', 'target_2_risk_reward',
                'target_3_price', 'target_3_percentage', 'target_3_risk_reward',
                'stop_loss_price', 'risk_percentage', 'move_percentage', 'detected_at'
            ]
            
            # Color code by confidence
            def highlight_confidence(val):
                if val >= 0.8:
                    return 'background-color: #90EE90'  # Light green
                elif val >= 0.7:
                    return 'background-color: #FFFFE0'  # Light yellow
                else:
                    return ''
            
            # Display signals table
            st.dataframe(
                signals_df[display_columns].style.applymap(
                    highlight_confidence,
                    subset=['confidence_score']
                ),
                use_container_width=True,
                height=400
            )
            
            # Quick stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Signals by Type")
                type_counts = signals_df['signal_type'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Signals by Level")
                level_counts = signals_df['fibonacci_level'].value_counts().head(5)
                fig = px.bar(x=level_counts.index, y=level_counts.values)
                fig.update_layout(xaxis_title="Fibonacci Level", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signals found for selected filters")

    with tab2:
        st.header("Performance Analysis")
        
        if not level_performance.empty:
            # Performance by Fibonacci level
            st.subheader("Performance by Fibonacci Level")
            
            fig = px.scatter(
                level_performance,
                x='fibonacci_level',
                y='avg_confidence',
                size='signal_count',
                color='signal_type',
                hover_data=['symbols', 'avg_quality'],
                title="Fibonacci Level Performance"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed performance table
            st.subheader("Detailed Level Statistics")
            st.dataframe(
                level_performance.sort_values('signal_count', ascending=False),
                use_container_width=True
            )
        
        # Time series analysis
        st.subheader("Signal Generation Over Time")
        
        logger = get_db()
        if logger.connection:
            try:
                logger.cursor.execute("""
                    SELECT 
                        DATE_TRUNC('hour', detected_at) as hour,
                        COUNT(*) as signal_count,
                        AVG(confidence_score) as avg_confidence,
                        AVG(quality_score) as avg_quality
                    FROM other_scanners.fibonacci_signals
                    WHERE detected_at > NOW() - INTERVAL %s
                    GROUP BY DATE_TRUNC('hour', detected_at)
                    ORDER BY hour
                """, (f"{hours_back} hours",))
                
                hourly_signals = pd.DataFrame(logger.cursor.fetchall())
                # Convert hour timestamps to UAE timezone
                if not hourly_signals.empty and 'hour' in hourly_signals.columns:
                    hourly_signals['hour'] = pd.to_datetime(hourly_signals['hour']).apply(convert_to_uae_time)
                
                if not hourly_signals.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hourly_signals['hour'],
                        y=hourly_signals['signal_count'],
                        mode='lines+markers',
                        name='Signal Count',
                        yaxis='y'
                    ))
                    fig.add_trace(go.Scatter(
                        x=hourly_signals['hour'],
                        y=hourly_signals['avg_confidence'],
                        mode='lines+markers',
                        name='Avg Confidence',
                        yaxis='y2'
                    ))
                    fig.add_trace(go.Scatter(
                        x=hourly_signals['hour'],
                        y=hourly_signals['avg_quality'] / 100,  # Normalize to 0-1
                        mode='lines+markers',
                        name='Avg Quality (normalized)',
                        yaxis='y2'
                    ))
                    fig.update_layout(
                        yaxis=dict(title="Signal Count"),
                        yaxis2=dict(title="Confidence/Quality", overlaying='y', side='right'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading time series data: {e}")

    with tab3:
        st.header("Top Trading Opportunities")
        
        # Get high confidence signals with complete trading data
        logger = get_db()
        if logger.connection:
            try:
                logger.cursor.execute("""
                    SELECT 
                        symbol,
                        signal_type,
                        fibonacci_level,
                        price_level,
                        confidence_score,
                        quality_score,
                        volume_confirmation,
                        momentum_confirmation,
                        timeframe,
                        entry_timing_status,
                        target_1_price,
                        target_1_percentage,
                        target_1_risk_reward,
                        target_2_price,
                        target_2_percentage,
                        target_2_risk_reward,
                        target_3_price,
                        target_3_percentage,
                        target_3_risk_reward,
                        stop_loss_price,
                        risk_percentage,
                        move_percentage,
                        detected_at
                    FROM other_scanners.fibonacci_signals
                    WHERE confidence_score >= 0.75
                        AND detected_at > NOW() - INTERVAL '4 hours'
                    ORDER BY confidence_score DESC, quality_score DESC
                    LIMIT 20
                """)
                
                top_signals = pd.DataFrame(logger.cursor.fetchall())
                
                if not top_signals.empty:
                    st.subheader(f"🎯 Top {len(top_signals)} High-Confidence Signals (Last 4 Hours)")
                    
                    # Add trading recommendations
                    top_signals['recommendation'] = top_signals.apply(
                        lambda x: f"{'BUY' if x['signal_type'] in ['SUPPORT', 'BOUNCE'] else 'SELL'} near {x['price_level']:.8f}",
                        axis=1
                    )
                    
                    # Display with enhanced trading data
                    display_cols = [
                        'symbol', 'signal_type', 'fibonacci_level', 'confidence_score', 'quality_score',
                        'entry_timing_status', 'recommendation', 'target_1_risk_reward', 'target_2_risk_reward', 'target_3_risk_reward',
                        'risk_percentage', 'move_percentage', 'timeframe', 'detected_at'
                    ]
                    
                    st.dataframe(
                        top_signals[display_cols],
                        use_container_width=True
                    )
                    
                    # Symbol performance
                    st.subheader("Best Performing Symbols")
                    symbol_stats = top_signals.groupby('symbol').agg({
                        'confidence_score': 'mean',
                        'quality_score': 'mean',
                        'signal_type': 'count'
                    }).rename(columns={'signal_type': 'signal_count'}).sort_values('confidence_score', ascending=False)
                    
                    st.bar_chart(symbol_stats['confidence_score'].head(10))
                else:
                    st.info("No high-confidence signals in the last 4 hours")
            except Exception as e:
                st.error(f"Error loading top signals: {e}")

    with tab4:
        st.header("Historical Pattern Analysis")
        
        # Most common Fibonacci levels with enhanced data
        logger = get_db()
        if logger.connection:
            try:
                logger.cursor.execute("""
                    SELECT 
                        fibonacci_level,
                        signal_type,
                        COUNT(*) as occurrences,
                        AVG(confidence_score) as avg_confidence,
                        AVG(quality_score) as avg_quality,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        AVG(move_percentage) as avg_move_pct,
                        AVG(risk_percentage) as avg_risk_pct
                    FROM other_scanners.fibonacci_signals
                    WHERE detected_at > NOW() - INTERVAL '7 days'
                    GROUP BY fibonacci_level, signal_type
                    HAVING COUNT(*) > 10
                    ORDER BY occurrences DESC
                """)
                
                historical_patterns = pd.DataFrame(logger.cursor.fetchall())
                
                if not historical_patterns.empty:
                    st.subheader("Most Active Fibonacci Levels (7 Days)")
                    
                    # Heatmap of level activity
                    pivot_data = historical_patterns.pivot(
                        index='fibonacci_level',
                        columns='signal_type',
                        values='occurrences'
                    ).fillna(0)
                    
                    fig = px.imshow(
                        pivot_data,
                        labels=dict(x="Signal Type", y="Fibonacci Level", color="Occurrences"),
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Success patterns with enhanced metrics
                    st.subheader("Success Patterns with Trading Data")
                    success_patterns = historical_patterns[historical_patterns['avg_confidence'] >= 0.7]
                    
                    if not success_patterns.empty:
                        st.dataframe(
                            success_patterns.sort_values('avg_confidence', ascending=False),
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error loading historical patterns: {e}")

    # Auto-refresh option
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Add auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

# Main function to be called from dashboard
def show_fibonacci_analysis():
    """Main function to display Fibonacci analysis page"""
    create_fibonacci_analysis_page()
