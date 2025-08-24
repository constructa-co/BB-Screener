#!/usr/bin/env python3
"""
Wyckoff Analysis Dashboard
File: dashboard/wyckoff_analysis.py
Purpose: Display Wyckoff scanner signals with actionable trading data
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

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_wyckoff_data(hours_back=24, min_score=60, phase_filter=None, timeframe_filter=None):
    """Fetch Wyckoff signals from database with filters"""
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
                timeframe,
                phase,
                pattern_type,
                setup_score,
                entry_price,
                stop_loss,
                target_1,
                target_2,
                risk_reward_1,
                risk_reward_2,
                volume_confirmation,
                strength_score,
                entry_signal,
                wait_condition,
                current_price,
                support_level,
                resistance_level,
                range_size_pct,
                spring_detected,
                upthrust_detected,
                computed_at
            FROM other_scanners.wyckoff_signals
            WHERE computed_at > NOW() - INTERVAL '%s hours'
            AND setup_score >= %s
        """
        params = [hours_back, min_score]
        
        if phase_filter and phase_filter != "All":
            query += " AND phase = %s"
            params.append(phase_filter)
        
        if timeframe_filter and timeframe_filter != "All":
            query += " AND timeframe = %s"
            params.append(timeframe_filter)
        
        query += " ORDER BY setup_score DESC, computed_at DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert Decimal to float for display
        numeric_columns = ['entry_price', 'stop_loss', 'target_1', 'target_2', 
                          'risk_reward_1', 'risk_reward_2', 'volume_confirmation', 
                          'strength_score', 'current_price', 'support_level', 
                          'resistance_level', 'range_size_pct']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        return df
        
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calculate key metrics for the dashboard"""
    if df.empty:
        return {
            'total_signals': 0,
            'avg_score': 0,
            'phase_distribution': {},
            'pattern_distribution': {},
            'timeframe_distribution': {}
        }
    
    metrics = {
        'total_signals': len(df),
        'avg_score': round(df['setup_score'].mean(), 1),
        'phase_distribution': df['phase'].value_counts().to_dict(),
        'pattern_distribution': df['pattern_type'].value_counts().to_dict(),
        'timeframe_distribution': df['timeframe'].value_counts().to_dict()
    }
    
    return metrics

def create_phase_chart(df):
    """Create phase distribution chart"""
    if df.empty:
        return None
    
    phase_counts = df['phase'].value_counts()
    
    # Color mapping for phases
    colors = {
        'ACCUMULATION': '#00FF00',  # Green
        'DISTRIBUTION': '#FF0000',  # Red
        'MARKUP': '#0000FF',        # Blue
        'MARKDOWN': '#FFA500'       # Orange
    }
    
    fig = px.pie(
        values=phase_counts.values,
        names=phase_counts.index,
        title="Wyckoff Phase Distribution",
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_score_distribution(df):
    """Create setup score distribution chart"""
    if df.empty:
        return None
    
    fig = px.histogram(
        df, 
        x='setup_score',
        nbins=20,
        title="Setup Score Distribution",
        labels={'setup_score': 'Setup Score', 'count': 'Number of Signals'}
    )
    
    fig.add_vline(x=80, line_dash="dash", line_color="green", 
                  annotation_text="High Quality (80+)")
    fig.add_vline(x=60, line_dash="dash", line_color="orange", 
                  annotation_text="Minimum (60+)")
    
    return fig

def format_signal_table(df):
    """Format the signals table for display"""
    if df.empty:
        return df
    
    # Select and rename columns for display
    display_df = df[[
        'symbol', 'timeframe', 'phase', 'pattern_type', 'setup_score',
        'entry_price', 'stop_loss', 'target_1', 'risk_reward_1',
        'volume_confirmation', 'strength_score', 'computed_at'
    ]].copy()
    
    # Rename columns for better display
    display_df.columns = [
        'Symbol', 'TF', 'Phase', 'Pattern', 'Score',
        'Entry', 'Stop', 'Target1', 'R/R',
        'Vol Ratio', 'Strength', 'Detected'
    ]
    
    # Format numeric columns
    display_df['Entry'] = display_df['Entry'].round(4)
    display_df['Stop'] = display_df['Stop'].round(4)
    display_df['Target1'] = display_df['Target1'].round(4)
    display_df['R/R'] = display_df['R/R'].round(2)
    display_df['Vol Ratio'] = display_df['Vol Ratio'].round(2)
    display_df['Strength'] = display_df['Strength'].round(2)
    
    # Convert to UAE time (UTC+4)
    display_df['Detected'] = pd.to_datetime(display_df['Detected']).dt.tz_convert('Asia/Dubai')
    display_df['Detected'] = display_df['Detected'].dt.strftime('%Y-%m-%d %H:%M')
    
    return display_df

def show_wyckoff_analysis():
    """Main Wyckoff analysis dashboard"""
    st.set_page_config(
        page_title="Wyckoff Analysis",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Wyckoff Pattern Analysis")
    st.markdown("Real-time Wyckoff accumulation and distribution pattern signals")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    hours_back = st.sidebar.slider(
        "Hours Back", 
        min_value=1, 
        max_value=168, 
        value=24,
        help="How many hours back to fetch signals"
    )
    
    min_score = st.sidebar.slider(
        "Minimum Score", 
        min_value=60, 
        max_value=100, 
        value=60,
        help="Minimum setup score to display"
    )
    
    phase_filter = st.sidebar.selectbox(
        "Phase Filter",
        ["All", "ACCUMULATION", "DISTRIBUTION", "MARKUP", "MARKDOWN"],
        help="Filter by Wyckoff phase"
    )
    
    timeframe_filter = st.sidebar.selectbox(
        "Timeframe Filter",
        ["All", "1h", "15m", "4h"],
        help="Filter by timeframe"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Fetch data
    with st.spinner("Fetching Wyckoff signals..."):
        df = get_wyckoff_data(hours_back, min_score, phase_filter, timeframe_filter)
    
    if df.empty:
        st.warning("No Wyckoff signals found with current filters")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", metrics['total_signals'])
    
    with col2:
        st.metric("Average Score", metrics['avg_score'])
    
    with col3:
        top_phase = max(metrics['phase_distribution'].items(), key=lambda x: x[1]) if metrics['phase_distribution'] else ("None", 0)
        st.metric("Top Phase", f"{top_phase[0]} ({top_phase[1]})")
    
    with col4:
        top_pattern = max(metrics['pattern_distribution'].items(), key=lambda x: x[1]) if metrics['pattern_distribution'] else ("None", 0)
        st.metric("Top Pattern", f"{top_pattern[0]} ({top_pattern[1]})")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        phase_chart = create_phase_chart(df)
        if phase_chart:
            st.plotly_chart(phase_chart, use_container_width=True)
    
    with col2:
        score_chart = create_score_distribution(df)
        if score_chart:
            st.plotly_chart(score_chart, use_container_width=True)
    
    # Signals table
    st.subheader("📊 Wyckoff Signals")
    
    display_df = format_signal_table(df)
    
    # Color coding for phases
    def color_phase(val):
        if val == 'ACCUMULATION':
            return 'background-color: #90EE90'  # Light green
        elif val == 'DISTRIBUTION':
            return 'background-color: #FFB6C1'  # Light red
        elif val == 'MARKUP':
            return 'background-color: #87CEEB'  # Light blue
        elif val == 'MARKDOWN':
            return 'background-color: #FFE4B5'  # Light orange
        return ''
    
    # Apply styling
    styled_df = display_df.style.applymap(
        color_phase, 
        subset=['Phase']
    ).format({
        'Score': '{:.0f}',
        'Entry': '{:.4f}',
        'Stop': '{:.4f}',
        'Target1': '{:.4f}',
        'R/R': '{:.2f}',
        'Vol Ratio': '{:.2f}',
        'Strength': '{:.2f}'
    })
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Signals CSV",
        data=csv,
        file_name=f"wyckoff_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Data Source**: Wyckoff Scanner R1 | "
        "**Last Updated**: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " UAE"
    )

if __name__ == "__main__":
    show_wyckoff_analysis()
