#!/usr/bin/env python3
"""
ICT Analysis Dashboard
File: dashboard/ict_analysis.py
Purpose: Display ICT scanner signals with actionable trading data across all timeframes
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import os
from decimal import Decimal
import sys
import json

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_ict_data(hours_back=24, min_score=60, timeframe_filter=None, pattern_filter=None):
    """Fetch ICT signals from database with filters"""
    try:
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            st.error("DATABASE_URL not configured")
            return pd.DataFrame()
        
        conn = psycopg2.connect(dsn)
        
        # Build query with filters
        query = """
            SELECT 
                symbol,
                COALESCE(timeframe, '1h') as timeframe,
                side,
                entry_price,
                stop_loss,
                take_profit,
                quantity,
                timestamp,
                probability,
                risk_reward_ratio,
                current_price,
                bb_score,
                rsi,
                mfi,
                stochastic_k,
                volume_surge,
                macd_signal,
                pattern_type,
                pattern_quality,
                confluence_score,
                historical_win_rate,
                category_win_rate,
                similar_setups_count,
                market_cap,
                volume_24h,
                price_change_24h,
                scanner_specific_data,
                trade_taken,
                trade_result,
                actual_exit_price,
                profit_loss_percent,
                gap_high,
                gap_low,
                gap_size_pct,
                swing_high,
                swing_low,
                order_block_high,
                order_block_low,
                fib_618,
                fib_500,
                fib_382,
                liquidity_sweep_level,
                imbalance_high,
                imbalance_low,
                fib_236,
                fib_786
            FROM public.trade_opportunities
            WHERE timestamp > NOW() - INTERVAL '%s hours'
            AND scanner_specific_data::text LIKE '%%ICT%%'
        """
        params = [hours_back]
        
        if timeframe_filter and timeframe_filter != "All":
            query += " AND COALESCE(timeframe, '1h') = %s"
            params.append(timeframe_filter)
        
        if pattern_filter and pattern_filter != "All":
            query += " AND scanner_specific_data->>'pattern_type' LIKE %s"
            params.append(f"%{pattern_filter}%")
        
        query += " ORDER BY timestamp DESC, probability DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert Decimal to float for display
        numeric_columns = ['entry_price', 'stop_loss', 'take_profit', 'current_price', 
                          'probability', 'risk_reward_ratio', 'bb_score', 'rsi', 'mfi', 
                          'stochastic_k', 'volume_surge', 'confluence_score', 
                          'historical_win_rate', 'category_win_rate', 'market_cap', 
                          'volume_24h', 'price_change_24h', 'profit_loss_percent',
                          'gap_high', 'gap_low', 'gap_size_pct', 'swing_high', 'swing_low',
                          'order_block_high', 'order_block_low', 'fib_618', 'fib_500', 
                          'fib_382', 'liquidity_sweep_level', 'imbalance_high', 
                          'imbalance_low', 'fib_236', 'fib_786']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Extract additional data from scanner_specific_data JSONB
        df = extract_ict_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()

def extract_ict_data(df):
    """Extract additional ICT-specific data from scanner_specific_data JSONB field"""
    try:
        # Extract pattern type
        df['pattern_type'] = df['scanner_specific_data'].apply(
            lambda x: x.get('pattern_type', 'Unknown') if isinstance(x, dict) else 'Unknown'
        )
        
        # Extract FVG-specific data
        df['fvg_age'] = df['scanner_specific_data'].apply(
            lambda x: x.get('fvg_age', 0) if isinstance(x, dict) else 0
        )
        
        df['distance_to_entry'] = df['scanner_specific_data'].apply(
            lambda x: x.get('distance_to_entry', 0) if isinstance(x, dict) else 0
        )
        
        df['gap_midpoint'] = df['scanner_specific_data'].apply(
            lambda x: x.get('gap_midpoint', 0) if isinstance(x, dict) else 0
        )
        
        df['swing_range_pct'] = df['scanner_specific_data'].apply(
            lambda x: x.get('swing_range_pct', 0) if isinstance(x, dict) else 0
        )
        
        df['risk_pct'] = df['scanner_specific_data'].apply(
            lambda x: x.get('risk_pct', 0) if isinstance(x, dict) else 0
        )
        
        df['action_required'] = df['scanner_specific_data'].apply(
            lambda x: x.get('action_required', '') if isinstance(x, dict) else ''
        )
        
        df['quality_score'] = df['scanner_specific_data'].apply(
            lambda x: x.get('quality_score', 0) if isinstance(x, dict) else 0
        )
        
        df['fib_quality'] = df['scanner_specific_data'].apply(
            lambda x: x.get('fib_quality', 0) if isinstance(x, dict) else 0
        )
        
        # Extract market conditions
        df['category'] = df['scanner_specific_data'].apply(
            lambda x: x.get('category', '') if isinstance(x, dict) else ''
        )
        
        df['volume_surge_bool'] = df['scanner_specific_data'].apply(
            lambda x: x.get('volume_surge', False) if isinstance(x, dict) else False
        )
        
        # Extract Fibonacci targets from feature_vector
        df['target_t1'] = df['scanner_specific_data'].apply(
            lambda x: x.get('feature_vector', [0]*8)[0] if isinstance(x, dict) and 'feature_vector' in x else 0
        )
        
        df['target_t2'] = df['scanner_specific_data'].apply(
            lambda x: x.get('feature_vector', [0]*8)[1] if isinstance(x, dict) and 'feature_vector' in x else 0
        )
        
        df['target_t3'] = df['scanner_specific_data'].apply(
            lambda x: x.get('feature_vector', [0]*8)[2] if isinstance(x, dict) and 'feature_vector' in x else 0
        )
        
        return df
        
    except Exception as e:
        st.warning(f"Error extracting ICT data: {e}")
        return df

def show_ict_analysis():
    """Main ICT analysis page"""
    st.title("🎯 ICT Scanner Analysis")
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Time range filter
    hours_back = st.sidebar.slider(
        "Hours Back", 
        min_value=1, 
        max_value=168, 
        value=24,
        help="Number of hours to look back for signals"
    )
    
    # Minimum score filter
    min_score = st.sidebar.slider(
        "Minimum Score", 
        min_value=0, 
        max_value=100, 
        value=60,
        help="Minimum probability/quality score"
    )
    
    # Timeframe filter
    timeframe_options = ["All", "1h", "4h", "15m"]
    timeframe_filter = st.sidebar.selectbox(
        "Timeframe",
        timeframe_options,
        help="Filter by specific timeframe"
    )
    
    # Pattern filter
    pattern_options = ["All", "ICT FVG bullish", "ICT FVG bearish"]
    pattern_filter = st.sidebar.selectbox(
        "Pattern Type",
        pattern_options,
        help="Filter by ICT pattern type"
    )
    
    # Fetch data
    df = get_ict_data(hours_back, min_score, timeframe_filter, pattern_filter)
    
    if df.empty:
        st.warning("No ICT signals found with current filters")
        return
    
    # Filter by minimum score
    df = df[df['probability'] >= min_score]
    
    if df.empty:
        st.warning(f"No ICT signals found with probability >= {min_score}")
        return
    
    # Convert timestamp to UAE time (UTC+4)
    df['timestamp_uae'] = pd.to_datetime(df['timestamp']) + timedelta(hours=4)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Signals", 
            len(df),
            help="Total number of ICT signals found"
        )
    
    with col2:
        avg_score = df['probability'].mean()
        st.metric(
            "Avg Score", 
            f"{avg_score:.1f}",
            help="Average probability score"
        )
    
    with col3:
        bullish_count = len(df[df['side'] == 'BUY'])
        st.metric(
            "Bullish Signals", 
            bullish_count,
            help="Number of bullish (BUY) signals"
        )
    
    with col4:
        bearish_count = len(df[df['side'] == 'SELL'])
        st.metric(
            "Bearish Signals", 
            bearish_count,
            help="Number of bearish (SELL) signals"
        )
    
    # Timeframe distribution
    st.subheader("📊 Signal Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeframe distribution
        timeframe_counts = df['timeframe'].value_counts()
        fig_timeframe = px.pie(
            values=timeframe_counts.values,
            names=timeframe_counts.index,
            title="Signals by Timeframe"
        )
        st.plotly_chart(fig_timeframe, use_container_width=True)
    
    with col2:
        # Pattern distribution
        pattern_counts = df['pattern_type'].value_counts()
        fig_pattern = px.pie(
            values=pattern_counts.values,
            names=pattern_counts.index,
            title="Signals by Pattern Type"
        )
        st.plotly_chart(fig_pattern, use_container_width=True)
    
    # Score distribution
    st.subheader("📈 Score Distribution")
    fig_score = px.histogram(
        df, 
        x='probability',
        nbins=20,
        title="Probability Score Distribution",
        labels={'probability': 'Probability Score', 'count': 'Number of Signals'}
    )
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Main signals table
    st.subheader("🎯 ICT Signals Table")
    
    # Prepare table data
    display_df = df.copy()
    
    # Format columns for display
    display_df['Entry Price'] = display_df['entry_price'].apply(lambda x: f"${x:.6f}" if x else "N/A")
    display_df['Stop Loss'] = display_df['stop_loss'].apply(lambda x: f"${x:.6f}" if x else "N/A")
    display_df['Take Profit'] = display_df['take_profit'].apply(lambda x: f"${x:.6f}" if x else "N/A")
    display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:.6f}" if x else "N/A")
    display_df['Score'] = display_df['probability'].apply(lambda x: f"{x:.1f}" if x else "N/A")
    display_df['R:R Ratio'] = display_df['risk_reward_ratio'].apply(lambda x: f"{x:.2f}" if x else "N/A")
    display_df['Gap Size %'] = display_df['gap_size_pct'].apply(lambda x: f"{x:.2f}%" if x else "N/A")
    display_df['FVG Age'] = display_df['fvg_age'].apply(lambda x: f"{x}h" if x else "N/A")
    display_df['Side'] = display_df['side'].apply(lambda x: "🟢 LONG" if x == 'BUY' else "🔴 SHORT")
    display_df['Timeframe'] = display_df['timeframe'].apply(lambda x: x.upper())
    display_df['Detected'] = display_df['timestamp_uae'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else "N/A")
    
    # Select columns for display
    columns_to_show = [
        'symbol', 'Side', 'Timeframe', 'Score', 'Entry Price', 'Stop Loss', 'Take Profit',
        'Current Price', 'R:R Ratio', 'Gap Size %', 'FVG Age', 'pattern_type', 'Detected'
    ]
    
    # Rename columns for display
    column_mapping = {
        'symbol': 'Symbol',
        'pattern_type': 'Pattern',
        'timestamp_uae': 'Detected'
    }
    
    display_df = display_df[columns_to_show].rename(columns=column_mapping)
    
    # Color code the dataframe
    def color_code_row(row):
        colors = []
        for col in row.index:
            if col == 'Side':
                colors.append('background-color: lightgreen' if 'LONG' in str(row[col]) else 'background-color: lightcoral')
            elif col == 'Score':
                score = float(str(row[col]).replace('N/A', '0'))
                if score >= 90:
                    colors.append('background-color: #90EE90; color: black')
                elif score >= 80:
                    colors.append('background-color: #98FB98; color: black')
                elif score >= 70:
                    colors.append('background-color: #F0E68C; color: black')
                else:
                    colors.append('background-color: #FFB6C1; color: black')
            else:
                colors.append('')
        return colors
    
    # Apply styling
    styled_df = display_df.style.apply(color_code_row, axis=1)
    
    # Display the table
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download ICT Signals CSV",
        data=csv,
        file_name=f"ict_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Detailed analysis section
    st.subheader("🔍 Detailed Analysis")
    
    # Top signals by score
    top_signals = df.nlargest(10, 'probability')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏆 Top 10 Signals by Score**")
        for idx, row in top_signals.iterrows():
            score_color = "🟢" if row['probability'] >= 90 else "🟡" if row['probability'] >= 80 else "🟠"
            side_emoji = "📈" if row['side'] == 'BUY' else "📉"
            st.markdown(
                f"{score_color} **{row['symbol']}** {side_emoji} "
                f"Score: {row['probability']:.1f} | "
                f"Entry: ${row['entry_price']:.6f} | "
                f"R:R: {row['risk_reward_ratio']:.2f}"
            )
    
    with col2:
        st.markdown("**📊 Performance Metrics**")
        st.markdown(f"**Average Score:** {df['probability'].mean():.1f}")
        st.markdown(f"**Average R:R Ratio:** {df['risk_reward_ratio'].mean():.2f}")
        st.markdown(f"**Average Gap Size:** {df['gap_size_pct'].mean():.2f}%")
        st.markdown(f"**Signals in Last Hour:** {len(df[df['timestamp'] > datetime.now() - timedelta(hours=1)])}")
    
    # Auto-refresh
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

if __name__ == "__main__":
    show_ict_analysis()
