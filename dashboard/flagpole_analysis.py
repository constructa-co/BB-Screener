#!/usr/bin/env python3
"""
Flagpole Pattern Analysis Dashboard
Standalone dashboard for flagpole scanner analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import os
import json
import pytz

st.set_page_config(
    page_title="🚩 Flagpole Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_db_connection():
    """Get database connection"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        st.error("DATABASE_URL not set")
        return None
    
    try:
        return psycopg2.connect(database_url)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_flagpole_signals(hours_back=24, min_score=60):
    """Fetch flagpole signals from database"""
    
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                signal_id, symbol, timeframe, detected_at,
                pattern_type, pattern_details, direction,
                current_price, breakout_level, target_price, stop_loss,
                potential_pct, risk_pct, risk_reward,
                pole_pct, vol_decline_pct, slope_pct, age_candles,
                score, quality_raw, quality_indicators,
                is_ready, has_strong_vol, has_fast_pole,
                expires_at
            FROM other_scanners.flagpole_signals
            WHERE detected_at > NOW() - INTERVAL '%s hours'
                AND score >= %s
            ORDER BY detected_at DESC
            LIMIT 500
        """
        
        df = pd.read_sql(query, conn, params=(hours_back, min_score))
        
        if not df.empty:
            # Format timestamps in UTC (already handled by timezone wrapper)
            df['detected_at'] = df['detected_at'].dt.strftime('%Y-%m-%d %H:%M')
            if 'expires_at' in df.columns:
                df['expires_at'] = df['expires_at'].dt.strftime('%Y-%m-%d %H:%M')
        
        return df
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def show_flagpole_analysis():
    """Main dashboard function"""
    
    st.title("🚩 Flagpole & Triangle Pattern Analysis")
    
    # Display current UAE time
    uae_tz = pytz.timezone('Asia/Dubai')
    current_time_uae = datetime.now(uae_tz)
    st.caption(f"Last updated: {current_time_uae.strftime('%Y-%m-%d %H:%M:%S')} UAE")
    
    st.markdown("---")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🚩 Filters")
        
        hours_back = st.slider("Hours Back", 1, 168, 24)
        min_score = st.slider("Min Score", 0, 100, 60)
        
        # Pattern type filter
        pattern_types = st.multiselect(
            "Pattern Types",
            ["Flag", "Pennant", "Triangle"],
            default=["Flag", "Pennant"]
        )
        
        # Direction filter
        directions = st.multiselect(
            "Direction",
            ["Bullish", "Bearish", "Either direction"],
            default=["Bullish", "Bearish"]
        )
        
        # Quality filters
        st.subheader("Quality Indicators")
        show_ready = st.checkbox("Ready Only", value=False)
        show_strong_vol = st.checkbox("Strong Volume Only", value=False)
        show_fast_pole = st.checkbox("Fast Pole Only", value=False)
    
    # Fetch data
    with st.spinner("Fetching flagpole signals..."):
        df = fetch_flagpole_signals(hours_back, min_score)
    
    if df.empty:
        st.warning("No flagpole signals found in the specified time range")
        return
    
    # Apply filters
    if pattern_types:
        df = df[df['pattern_type'].isin(pattern_types)]
    
    if directions:
        df = df[df['direction'].isin(directions)]
    
    if show_ready:
        df = df[df['is_ready'] == True]
    
    if show_strong_vol:
        df = df[df['has_strong_vol'] == True]
    
    if show_fast_pole:
        df = df[df['has_fast_pole'] == True]
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Patterns", len(df))
    
    with col2:
        avg_score = df['score'].mean()
        st.metric("Avg Score", f"{avg_score:.0f}")
    
    with col3:
        ready_count = df['is_ready'].sum()
        st.metric("Ready", ready_count)
    
    with col4:
        avg_rr = df['risk_reward'].mean()
        st.metric("Avg R:R", f"{avg_rr:.2f}")
    
    with col5:
        avg_potential = df['potential_pct'].mean()
        st.metric("Avg Potential", f"{avg_potential:.1f}%")
    
    # Charts
    st.markdown("---")
    st.subheader("🚩 Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pattern distribution
        fig = px.pie(
            df, 
            names='pattern_type', 
            title="Pattern Distribution",
            color_discrete_map={
                'Flag': '#1f77b4',
                'Pennant': '#ff7f0e', 
                'Triangle': '#2ca02c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk/Reward distribution
        fig = px.histogram(
            df,
            x='risk_reward',
            nbins=20,
            title="Risk/Reward Distribution",
            labels={'risk_reward': 'R:R Ratio', 'count': 'Count'}
        )
        fig.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
    
    # Quality indicators chart
    col1, col2 = st.columns(2)
    
    with col1:
        quality_data = pd.DataFrame({
            'Indicator': ['Ready', 'Strong Vol', 'Fast Pole'],
            'Count': [
                df['is_ready'].sum(),
                df['has_strong_vol'].sum(),
                df['has_fast_pole'].sum()
            ]
        })
        
        fig = px.bar(
            quality_data,
            x='Indicator',
            y='Count',
            title="Quality Indicators",
            color='Indicator',
            color_discrete_map={
                'Ready': '#2ca02c',
                'Strong Vol': '#1f77b4',
                'Fast Pole': '#ff7f0e'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score distribution
        fig = px.histogram(
            df,
            x='score',
            nbins=10,
            title="Score Distribution",
            labels={'score': 'Score', 'count': 'Count'}
        )
        fig.update_traces(marker_color='#9467bd')
        st.plotly_chart(fig, use_container_width=True)
    
    # Signals table
    st.markdown("---")
    st.subheader("🚩 Recent Signals")
    
    if not df.empty:
        # Prepare display dataframe
        display_df = pd.DataFrame({
            'Symbol': df['symbol'],
            'Pattern': df['pattern_details'],
            'Direction': df['direction'],
            'Score': df['score'],
            'Current': df['current_price'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A"),
            'Breakout': df['breakout_level'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A"),
            'Target': df['target_price'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A"),
            'Stop': df['stop_loss'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A"),
            'Potential %': df['potential_pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"),
            'Risk %': df['risk_pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"),
            'R:R': df['risk_reward'].apply(lambda x: f"{x:.1f}:1" if pd.notna(x) else "N/A"),
            'Pole %': df['pole_pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"),
            'Vol Decline %': df['vol_decline_pct'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"),
            'Age': df['age_candles'].apply(lambda x: f"{x} candles" if pd.notna(x) else "N/A"),
            'Quality': df['quality_raw'],
            'Detected (UAE)': df['detected_at'].dt.strftime('%m-%d %H:%M')
        })
        
        # Apply color coding based on score
        def highlight_score(row):
            score = df.loc[row.name, 'score']
            if score >= 90:
                return ['background-color: #d4f4dd'] * len(row)
            elif score >= 70:
                return ['background-color: #fff3cd'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_score, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "🚩 Download CSV",
            csv,
            f"flagpole_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            key='download'
        )

if __name__ == "__main__":
    show_flagpole_analysis()
