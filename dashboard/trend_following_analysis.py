#!/usr/bin/env python3
"""
Trend Following Scanner Analysis Dashboard
Comprehensive analysis of trend following signals and performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_trend_following_data():
    """Get trend following data from database"""
    try:
        # Get database URL from environment
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            st.error("❌ DATABASE_URL not found in environment")
            return None
        
        # Connect to database
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Get trend following signals
        query = """
            SELECT 
                symbol,
                signal_type,
                trend_direction,
                trend_strength,
                momentum_score,
                quality_score,
                confidence_score,
                entry_price,
                stop_loss,
                target_1,
                target_2,
                target_3,
                risk_reward_1,
                risk_reward_2,
                risk_reward_3,
                risk_pct,
                detected_at,
                algorithm_parameters
            FROM other_scanners.trend_following_signals 
            WHERE timeframe = '1h'
            ORDER BY detected_at DESC
        """
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        if data:
            df = pd.DataFrame(data, columns=columns)
            df['detected_at'] = pd.to_datetime(df['detected_at'])
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def show_trend_following_analysis():
    """Main trend following analysis page"""
    
    st.markdown('<h1 class="main-header">🎯 Trend Following Scanner Analysis</h1>', unsafe_allow_html=True)
    
    # Get data
    with st.spinner("📊 Loading Trend Following data..."):
        df = get_trend_following_data()
    
    if df is None or df.empty:
        st.warning("⚠️ No Trend Following data available")
        return
    
    # Data overview
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", len(df), f"Last 24h: {len(df[df['detected_at'] > datetime.now() - timedelta(hours=24)])}")
    
    with col2:
        avg_quality = df['quality_score'].mean()
        st.metric("Avg Quality Score", f"{avg_quality:.1f}/100", f"±{df['quality_score'].std():.1f}")
    
    with col3:
        bullish_count = len(df[df['signal_type'] == 'BULLISH'])
        bearish_count = len(df[df['signal_type'] == 'BEARISH'])
        st.metric("Signal Distribution", f"🟢 {bullish_count} | 🔴 {bearish_count}")
    
    with col4:
        recent_signals = len(df[df['detected_at'] > datetime.now() - timedelta(hours=1)])
        st.metric("Recent Activity", f"{recent_signals} signals", "Last hour")
    
    st.markdown("---")
    
    # Signal Quality Analysis
    st.subheader("🎯 Signal Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality score distribution
        fig_quality = px.histogram(
            df, 
            x='quality_score', 
            nbins=20,
            title="Signal Quality Distribution",
            labels={'quality_score': 'Quality Score', 'count': 'Number of Signals'},
            color_discrete_sequence=['#667eea']
        )
        fig_quality.update_layout(showlegend=False)
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        # Trend strength vs Quality
        fig_trend_quality = px.scatter(
            df,
            x='trend_strength',
            y='quality_score',
            color='signal_type',
            title="Trend Strength vs Quality Score",
            labels={'trend_strength': 'Trend Strength', 'quality_score': 'Quality Score'},
            color_discrete_map={'BULLISH': '#00ff00', 'BEARISH': '#ff0000', 'NEUTRAL': '#808080'}
        )
        st.plotly_chart(fig_trend_quality, use_container_width=True)
    
    st.markdown("---")
    
    # Signal Type Analysis
    st.subheader("📈 Signal Type Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Signal type pie chart
        signal_counts = df['signal_type'].value_counts()
        fig_signal_type = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Signal Type Distribution",
            color_discrete_map={'BULLISH': '#00ff00', 'BEARISH': '#ff0000', 'NEUTRAL': '#808080'}
        )
        st.plotly_chart(fig_signal_type, use_container_width=True)
    
    with col2:
        # Trend direction analysis
        trend_counts = df['trend_direction'].value_counts()
        fig_trend = px.bar(
            x=trend_counts.index,
            y=trend_counts.values,
            title="Trend Direction Analysis",
            labels={'x': 'Trend Direction', 'y': 'Number of Signals'},
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # Risk-Reward Analysis
    st.subheader("⚖️ Risk-Reward Analysis")
    
    # Filter out None values for R:R analysis
    rr_df = df[df['risk_reward_1'].notna() & (df['risk_reward_1'] > 0)]
    
    if not rr_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # R:R distribution
            fig_rr = px.histogram(
                rr_df,
                x='risk_reward_1',
                nbins=15,
                title="Risk-Reward Ratio Distribution (Target 1)",
                labels={'risk_reward_1': 'Risk-Reward Ratio', 'count': 'Number of Signals'},
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig_rr, use_container_width=True)
        
        with col2:
            # Quality vs R:R
            fig_quality_rr = px.scatter(
                rr_df,
                x='quality_score',
                y='risk_reward_1',
                color='signal_type',
                title="Quality Score vs Risk-Reward Ratio",
                labels={'quality_score': 'Quality Score', 'risk_reward_1': 'Risk-Reward Ratio'},
                color_discrete_map={'BULLISH': '#00ff00', 'BEARISH': '#ff0000', 'NEUTRAL': '#808080'}
            )
            st.plotly_chart(fig_quality_rr, use_container_width=True)
    else:
        st.info("ℹ️ No risk-reward data available for analysis")
    
    st.markdown("---")
    
    # Recent Signals Table
    st.subheader("🕒 Recent Signals")
    
    # Filter recent signals (last 24 hours)
    recent_df = df[df['detected_at'] > datetime.now() - timedelta(hours=24)].copy()
    
    if not recent_df.empty:
        # Format the data for display
        display_df = recent_df[['symbol', 'signal_type', 'trend_direction', 'quality_score', 
                               'trend_strength', 'risk_reward_1', 'detected_at']].copy()
        
        # Format timestamps
        display_df['detected_at'] = display_df['detected_at'].dt.strftime('%H:%M')
        
        # Add emojis for signal types
        display_df['Signal'] = display_df['signal_type'].map({
            'BULLISH': '🟢 BULLISH',
            'BEARISH': '🔴 BEARISH',
            'NEUTRAL': '⚪ NEUTRAL'
        })
        
        # Reorder columns
        display_df = display_df[['symbol', 'Signal', 'trend_direction', 'quality_score', 
                                'trend_strength', 'risk_reward_1', 'detected_at']]
        
        # Rename columns for display
        display_df.columns = ['Symbol', 'Signal', 'Trend', 'Quality', 'Strength', 'R:R', 'Time']
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("ℹ️ No recent signals in the last 24 hours")
    
    st.markdown("---")
    
    # Top Quality Signals
    st.subheader("🏆 Top Quality Signals")
    
    # Get top 10 signals by quality score
    top_signals = df.nlargest(10, 'quality_score')[['symbol', 'signal_type', 'trend_direction', 
                                                   'quality_score', 'trend_strength', 'risk_reward_1', 'detected_at']]
    
    if not top_signals.empty:
        # Format for display
        top_display = top_signals.copy()
        top_display['detected_at'] = top_display['detected_at'].dt.strftime('%Y-%m-%d %H:%M')
        top_display['Signal'] = top_display['signal_type'].map({
            'BULLISH': '🟢 BULLISH',
            'BEARISH': '🔴 BEARISH',
            'NEUTRAL': '⚪ NEUTRAL'
        })
        
        # Reorder columns
        top_display = top_display[['symbol', 'Signal', 'trend_direction', 'quality_score', 
                                  'trend_strength', 'risk_reward_1', 'detected_at']]
        top_display.columns = ['Symbol', 'Signal', 'Trend', 'Quality', 'Strength', 'R:R', 'Detected']
        
        st.dataframe(top_display, use_container_width=True)
    
    st.markdown("---")
    
    # Performance Metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average trend strength
        avg_strength = df['trend_strength'].mean()
        st.metric("Avg Trend Strength", f"{avg_strength:.1f}/100", 
                 f"±{df['trend_strength'].std():.1f}")
    
    with col2:
        # High quality signals (>80)
        high_quality = len(df[df['quality_score'] >= 80])
        st.metric("High Quality Signals", f"{high_quality}", f"{(high_quality/len(df)*100):.1f}%")
    
    with col3:
        # Average R:R ratio
        if not rr_df.empty:
            avg_rr = rr_df['risk_reward_1'].mean()
            st.metric("Avg Risk-Reward", f"{avg_rr:.2f}:1", 
                     f"±{rr_df['risk_reward_1'].std():.2f}")
        else:
            st.metric("Avg Risk-Reward", "N/A", "No data")
    
    # Footer
    st.markdown("---")
    st.caption("📊 Trend Following Scanner Analysis - Real-time data from automated scanning")
