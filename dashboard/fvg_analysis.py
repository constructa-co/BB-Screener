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

def format_timestamp(timestamp_series):
    """Format timestamp for display"""
    try:
        # Check if it's a Series with datetime
        if hasattr(timestamp_series, 'dt'):
            return timestamp_series.dt.strftime('%Y-%m-%d %H:%M')
        # If it's already datetime objects
        else:
            return pd.Series(timestamp_series).apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else '')
    except:
        return timestamp_series

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
        conn = psycopg2.connect(database_url, options="-c timezone=UTC")
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
                signal_id,
                symbol, timeframe, detected_at, gap_type, gap_high, gap_low,
                gap_size, gap_size_pct,
                (gap_high + gap_low) / 2.0 AS gap_midpoint,
                COALESCE(gap_size_pct, ((gap_high - gap_low) / ((gap_high + gap_low) / 2.0)) * 100.0) AS gap_width_pct,
                current_price, entry_price, stop_loss, target_1, target_2, target_3,
                risk_reward_1, risk_reward_2, risk_reward_3,
                fib_level, fib_confluence, fib_confluence_score,
                setup_score, volume_at_gap, volume_confirmation, momentum_confirmation,
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
        
        # Use the timezone wrapper to handle inconsistent timestamp types
        from db_helper import query_with_timezone_fix
        df = query_with_timezone_fix(query, conn, params=(hours_back, limit))
        
        if not df.empty:
            # Timestamps from PostgreSQL are already timezone-aware
            # Just convert to UAE timezone (don't localize)
            import pytz
            uae_tz = pytz.timezone('Asia/Dubai')
            
            df['detected_at'] = format_timestamp(df['detected_at'])
            
            if 'expires_at' in df.columns and not df['expires_at'].isna().all():
                df['expires_at'] = format_timestamp(df['expires_at'])
            
            # Convert numeric columns for pandas compatibility
            numeric_columns = ['setup_score', 'gap_size', 'gap_size_pct', 'current_price', 'entry_price', 
                              'stop_loss', 'target_1', 'target_2', 'risk_reward_1', 'risk_reward_2', 
                              'fib_confluence_score', 'fill_percentage', 'gap_age_minutes']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
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
            
            # Format numeric columns for display
            df['fib_confluence_score'] = df['fib_confluence_score'].apply(
                lambda x: f"{int(x)}/10" if x and not pd.isna(x) else "N/A"
            )
            
            df['gap_age_minutes'] = df['gap_age_minutes'].apply(
                lambda x: f"{int(x)}m" if x and not pd.isna(x) else "N/A"
            )
        
        return df
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def display_signals(df, title="FVG Signals"):
    """Display FVG signals in a formatted table"""
    
    st.write(f"DEBUG: display_signals called with {len(df)} rows")
    
    if df.empty:
        st.info("No signals found for selected filters")
        return
    
    st.subheader(f"{title} ({len(df)} signals)")
    
    # Prepare display DataFrame with ALL fields
    display_df = pd.DataFrame()
    
    # Basic fields
    display_df['Symbol'] = df['symbol']
    display_df['TF'] = df['timeframe']
    display_df['Type'] = df['gap_type']
    display_df['Gap Range'] = df.apply(lambda x: f"${x['gap_low']:.6f} - ${x['gap_high']:.6f}" 
                                       if pd.notna(x['gap_low']) else "N/A", axis=1)
    display_df['Gap %'] = df['gap_percentage'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Price fields - these are missing in your current version
    display_df['Current'] = df['current_price'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
    display_df['Entry'] = df['entry_price'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
    display_df['Stop'] = df['stop_loss'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
    
    # Target fields
    display_df['TP1'] = df['target_1'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
    display_df['TP2'] = df['target_2'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
    
    # Risk/Reward
    display_df['R:R'] = df['risk_reward_1'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Fibonacci
    display_df['Fib'] = df['fib_level'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    display_df['Confluence'] = df['fib_confluence'].apply(lambda x: "✓" if x else "")
    
    # Status fields
    display_df['Score'] = df['setup_score']
    display_df['Status'] = df['gap_status']
    display_df['Filled'] = df['fill_percentage'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0%")
    
    # Time fields with UAE time (same as trend_following_analysis.py)
    import pytz
    uae_tz = pytz.timezone('Asia/Dubai')
    display_df['Detected (UTC)'] = format_timestamp(df['detected_at'])
    
    # Calculate age using UAE timezone
    now_uae = datetime.now(uae_tz)
    display_df['Age'] = df['detected_at'].apply(
        lambda x: f"{(pd.Timestamp.now() - pd.to_datetime(x)).total_seconds() / 3600:.1f}h"
    )
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def main():
    st.title("🎯 Fair Value Gap Analysis")
    
    # Display current UAE time at the top
    import pytz
    from datetime import datetime
    uae_tz = pytz.timezone('Asia/Dubai')
    current_time_uae = datetime.now(uae_tz)
    st.caption(f"Last updated: {current_time_uae.strftime('%Y-%m-%d %H:%M:%S')} UAE")
    
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
    # Convert fib_confluence_score to numeric for the scatter plot
    df_scatter = df.copy()
    df_scatter['fib_confluence_score'] = pd.to_numeric(df_scatter['fib_confluence_score'], errors='coerce').fillna(0)
    
    fig_scatter = px.scatter(
        df_scatter,
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
    
    # Display filtered results with enhanced format
    if not filtered_df.empty:
        # Create enhanced display DataFrame
        display_df = pd.DataFrame()
        
        # Basic fields
        display_df['Symbol'] = filtered_df['symbol']
        display_df['TF'] = filtered_df['timeframe']
        display_df['Type'] = filtered_df['gap_type']
        display_df['Gap Range'] = filtered_df.apply(lambda x: f"${x['gap_low']:.6f} - ${x['gap_high']:.6f}" 
                                                   if pd.notna(x['gap_low']) else "N/A", axis=1)
        display_df['Gap %'] = filtered_df['gap_size_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        # Price fields
        display_df['Current'] = filtered_df['current_price'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
        display_df['Entry'] = filtered_df['entry_price'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
        display_df['Stop'] = filtered_df['stop_loss'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
        
        # Target fields
        display_df['TP1'] = filtered_df['target_1'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
        display_df['TP2'] = filtered_df['target_2'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
        
        # Risk/Reward
        display_df['R:R'] = filtered_df['risk_reward_1'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # Fibonacci
        display_df['Fib'] = filtered_df['fib_level'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        display_df['Confluence'] = filtered_df['fib_confluence'].apply(lambda x: "✓" if x else "")
        
        # Status fields
        display_df['Score'] = filtered_df['setup_score']
        display_df['Status'] = filtered_df['gap_status']
        display_df['Filled'] = filtered_df['fill_percentage'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0%")
        
        # Time fields with UAE timezone (FIX: Use safe timezone conversion for both naive and aware timestamps)
        import pytz
        uae_tz = pytz.timezone('Asia/Dubai')
        display_df['Detected (UTC)'] = format_timestamp(filtered_df['detected_at'])
        
        # Calculate age using UAE time (timezone-aware calculation)
        now_uae = datetime.now(uae_tz)
        display_df['Age'] = filtered_df['detected_at'].apply(
            lambda x: f"{(now_uae - x).total_seconds() / 3600:.1f}h"
        )
        
        st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)
    else:
        st.info("No signals match the selected filters")
    
    # Download button
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"fvg_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("*FVG Analysis Dashboard - Real-time Fair Value Gap monitoring*")

if __name__ == "__main__":
    main()
