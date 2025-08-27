#!/usr/bin/env python3
"""
Fair Value Gap Analysis Dashboard
Standalone module for FVG signal analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
import os
import json

# Page configuration
st.set_page_config(
    page_title="FVG Analysis",
    page_icon="🎯",
    layout="wide"
)

# Database connection
def get_db_connection():
    """Get database connection"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        st.error("DATABASE_URL not set")
        return None
    
    try:
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def get_fvg_signals(hours_back=24, limit=1000):
    """Fetch FVG signals from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                symbol, timeframe, detected_at, gap_type, gap_high, gap_low,
                (gap_high + gap_low) / 2.0 AS gap_midpoint,
                COALESCE(gap_size_pct, ((gap_high - gap_low) / ((gap_high + gap_low) / 2.0)) * 100.0) AS gap_width_pct,
                current_price, entry_price, stop_loss, target_1, target_2, target_3,
                risk_reward_1, risk_reward_2, risk_reward_3,
                fib_level, fib_confluence, fib_confluence_score,
                setup_score, volume_confirmation, momentum_confirmation,
                gap_status, fill_percentage, gap_age_minutes,
                entry_timing, current_distance_pct, risk_pct,
                swing_high, swing_low, fib_levels, target_levels,
                expires_at, algorithm_parameters,
                -- Additional computed fields
                CASE 
                    WHEN current_price IS NOT NULL AND entry_price IS NOT NULL 
                    THEN ROUND(((current_price - entry_price) / entry_price) * 100, 2)
                    ELSE NULL 
                END AS price_vs_entry_pct,
                CASE 
                    WHEN target_1 IS NOT NULL AND entry_price IS NOT NULL 
                    THEN ROUND(((target_1 - entry_price) / entry_price) * 100, 2)
                    ELSE NULL 
                END AS tp1_pct,
                CASE 
                    WHEN target_2 IS NOT NULL AND entry_price IS NOT NULL 
                    THEN ROUND(((target_2 - entry_price) / entry_price) * 100, 2)
                    ELSE NULL 
                END AS tp2_pct,
                CASE 
                    WHEN stop_loss IS NOT NULL AND entry_price IS NOT NULL 
                    THEN ROUND(((stop_loss - entry_price) / entry_price) * 100, 2)
                    ELSE NULL 
                END AS stop_pct
            FROM other_scanners.fvg_signals
            WHERE detected_at > NOW() - INTERVAL '%s hours'
            ORDER BY detected_at DESC
            LIMIT %s
        """
        
        df = pd.read_sql(query, conn, params=(hours_back, limit))
        
        if not df.empty:
            # Convert UTC to local time (UTC+4 for UAE)
            df['detected_at'] = pd.to_datetime(df['detected_at']) + pd.Timedelta(hours=4)
            
            # Add entry timing display
            df['entry_status'] = df['entry_timing'].apply(lambda x: {
                'immediate': '⚡ NOW',
                'waiting': '⏳ WAIT', 
                'approaching': '🔜 APPROACHING',
                'close': '🔜 CLOSE'
            }.get(str(x).lower(), str(x).upper()) if x else 'UNKNOWN')
            
            # Add distance display
            df['distance_display'] = df['current_distance_pct'].apply(
                lambda x: f"{float(x):.2f}%" if x and not pd.isna(x) else "N/A"
            )
            
            # Add risk display
            df['risk_display'] = df['risk_pct'].apply(
                lambda x: f"{float(x):.2f}%" if x and not pd.isna(x) else "N/A"
            )
            
            # Add enhanced display fields
            df['current_price_display'] = df['current_price'].apply(
                lambda x: f"${x:.4f}" if x and not pd.isna(x) else "N/A"
            )
            
            df['entry_price_display'] = df['entry_price'].apply(
                lambda x: f"${x:.4f}" if x and not pd.isna(x) else "N/A"
            )
            
            df['tp1_display'] = df.apply(
                lambda x: f"${x['target_1']:.4f} ({x['tp1_pct']:.2f}%)" if x['target_1'] and not pd.isna(x['target_1']) else "N/A", axis=1
            )
            
            df['tp2_display'] = df.apply(
                lambda x: f"${x['target_2']:.4f} ({x['tp2_pct']:.2f}%)" if x['target_2'] and not pd.isna(x['target_2']) else "N/A", axis=1
            )
            
            df['stop_display'] = df.apply(
                lambda x: f"${x['stop_loss']:.4f} ({x['stop_pct']:.2f}%)" if x['stop_loss'] and not pd.isna(x['stop_loss']) else "N/A", axis=1
            )
            
            df['fib_range_display'] = df.apply(
                lambda x: f"${x['swing_low']:.4f} → ${x['swing_high']:.4f}" if x['swing_low'] and x['swing_high'] and not pd.isna(x['swing_low']) and not pd.isna(x['swing_high']) else "N/A", axis=1
            )
            
            df['confluence_display'] = df['fib_confluence_score'].apply(
                lambda x: f"{int(x)}/10" if x and not pd.isna(x) else "N/A"
            )
            
            df['gap_age_display'] = df['gap_age_minutes'].apply(
                lambda x: f"{int(x)}m" if x and not pd.isna(x) else "N/A"
            )
        
        return df
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def main():
    st.title("🎯 Fair Value Gap Analysis")
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    hours_back = st.sidebar.slider("Hours Back", 1, 168, 24, help="How many hours back to analyze")
    
    # Fetch data
    with st.spinner("Fetching FVG signals..."):
        df = get_fvg_signals(hours_back=hours_back)
    
    if df.empty:
        st.warning("No FVG signals found in the specified time range")
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", len(df))
    
    with col2:
        avg_score = df['setup_score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col3:
        bullish_count = len(df[df['gap_type'] == 'BULLISH'])
        st.metric("Bullish Gaps", bullish_count)
    
    with col4:
        bearish_count = len(df[df['gap_type'] == 'BEARISH'])
        st.metric("Bearish Gaps", bearish_count)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Gap type distribution
        fig_gap_type = px.pie(
            df, 
            names='gap_type', 
            title="Gap Type Distribution",
            color_discrete_map={'BULLISH': 'green', 'BEARISH': 'red'}
        )
        st.plotly_chart(fig_gap_type, use_container_width=True)
    
    with col2:
        # Score distribution
        fig_score = px.histogram(
            df, 
            x='setup_score', 
            title="Setup Score Distribution",
            nbins=20
        )
        st.plotly_chart(fig_score, use_container_width=True)
    
    # Gap width vs Score scatter
    fig_scatter = px.scatter(
        df,
        x='gap_width_pct',
        y='setup_score',
        color='gap_type',
        size='fib_confluence_score',
        hover_data=['symbol', 'timeframe'],
        title="Gap Width vs Setup Score"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Signals table
    st.markdown("---")
    st.header("Recent FVG Signals")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_gap_type = st.selectbox("Gap Type", ["All"] + list(df['gap_type'].unique()))
    
    with col2:
        min_score = st.slider("Min Score", 0, 100, 60)
    
    with col3:
        selected_timeframe = st.selectbox("Timeframe", ["All"] + list(df['timeframe'].unique()))
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_gap_type != "All":
        filtered_df = filtered_df[filtered_df['gap_type'] == selected_gap_type]
    
    filtered_df = filtered_df[filtered_df['setup_score'] >= min_score]
    
    if selected_timeframe != "All":
        filtered_df = filtered_df[filtered_df['timeframe'] == selected_timeframe]
    
    # Display filtered results
    if not filtered_df.empty:
        # Format display columns
        display_df = filtered_df.copy()
        # detected_at is already converted to UAE time, just format it
        display_df['detected_at'] = display_df['detected_at'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['gap_range'] = display_df.apply(lambda x: f"${x['gap_low']:.4f} - ${x['gap_high']:.4f}", axis=1)
        display_df['midpoint'] = display_df['gap_midpoint'].apply(lambda x: f"${x:.4f}" if pd.notna(x) else "N/A")
        display_df['width_pct'] = display_df['gap_width_pct'].apply(lambda x: f"{x:.3f}%" if pd.notna(x) else "N/A")
        
        # Select columns to display
        columns_to_show = [
            'symbol', 'timeframe', 'detected_at', 'gap_type', 'gap_range', 
            'current_price_display', 'entry_price_display', 'setup_score', 'entry_status', 
            'tp1_display', 'tp2_display', 'stop_display', 'fib_range_display',
            'confluence_display', 'gap_age_display', 'gap_status'
        ]
        
        st.dataframe(
            display_df[columns_to_show].head(50),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"fvg_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No signals match the selected filters")
    
    # Footer
    st.markdown("---")
    st.markdown("*FVG Analysis Dashboard - Real-time Fair Value Gap monitoring*")

if __name__ == "__main__":
    main()
