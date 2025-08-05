#!/usr/bin/env python3
"""
Crypto Scanner Dashboard
Streamlit dashboard for viewing scanner results and database analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Crypto Scanner Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(ttl=300)  # Cache for 5 minutes
def get_database_connection():
    """Get database connection with caching"""
    try:
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            st.error("❌ DATABASE_URL not found in environment variables")
            return None
        
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

@st.cache_data(ttl=300)
def get_scan_results():
    """Get recent scan results"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            scan_type,
            scan_timestamp,
            total_coins_analyzed,
            premium_trades_found,
            execution_time_seconds,
            scanner_version
        FROM scan_results 
        ORDER BY scan_timestamp DESC 
        LIMIT 100
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching scan results: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_trade_opportunities():
    """Get recent trade opportunities"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            t.*,
            s.scan_type,
            s.scan_timestamp as scan_time
        FROM trade_opportunities t
        JOIN scan_results s ON t.scan_id = s.id
        ORDER BY t.timestamp DESC 
        LIMIT 500
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching trade opportunities: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_backtest_results():
    """Get backtest results"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            strategy_name,
            timeframe,
            total_trades,
            winning_trades,
            win_rate,
            avg_profit,
            max_drawdown,
            sharpe_ratio,
            run_timestamp
        FROM backtest_results 
        ORDER BY run_timestamp DESC 
        LIMIT 50
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching backtest results: {e}")
        return pd.DataFrame()

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Crypto Scanner Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📈 Overview", "🔍 Trade Opportunities", "📊 Backtest Results", "⚙️ Scanner Performance"]
    )
    
    # Check database connection
    conn = get_database_connection()
    if not conn:
        st.error("❌ Cannot connect to database. Please check your DATABASE_URL configuration.")
        return
    
    conn.close()
    
    if page == "📈 Overview":
        show_overview()
    elif page == "🔍 Trade Opportunities":
        show_trade_opportunities()
    elif page == "📊 Backtest Results":
        show_backtest_results()
    elif page == "⚙️ Scanner Performance":
        show_scanner_performance()

def show_overview():
    """Show dashboard overview"""
    
    st.header("📈 Dashboard Overview")
    
    # Get data
    scan_df = get_scan_results()
    trade_df = get_trade_opportunities()
    backtest_df = get_backtest_results()
    
    if scan_df.empty and trade_df.empty and backtest_df.empty:
        st.warning("⚠️ No data available. Please run some scans first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not scan_df.empty:
            total_scans = len(scan_df)
            st.metric("Total Scans", total_scans)
        else:
            st.metric("Total Scans", 0)
    
    with col2:
        if not trade_df.empty:
            total_opportunities = len(trade_df)
            st.metric("Trade Opportunities", total_opportunities)
        else:
            st.metric("Trade Opportunities", 0)
    
    with col3:
        if not trade_df.empty:
            avg_probability = trade_df['probability'].mean()
            st.metric("Avg Probability", f"{avg_probability:.1f}%")
        else:
            st.metric("Avg Probability", "0%")
    
    with col4:
        if not backtest_df.empty:
            avg_win_rate = backtest_df['win_rate'].mean()
            st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
        else:
            st.metric("Avg Win Rate", "0%")
    
    # Recent activity
    st.subheader("📊 Recent Activity")
    
    if not scan_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Scans")
            recent_scans = scan_df.head(10)[['scan_type', 'scan_timestamp', 'premium_trades_found', 'execution_time_seconds']]
            st.dataframe(recent_scans, use_container_width=True)
        
        with col2:
            st.subheader("Scan Performance")
            fig = px.line(scan_df, x='scan_timestamp', y='premium_trades_found', 
                         title='Premium Trades Found Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    # High probability trades
    if not trade_df.empty:
        st.subheader("🎯 High Probability Trades")
        high_prob_trades = trade_df[trade_df['probability'] >= 70].head(10)
        if not high_prob_trades.empty:
            st.dataframe(high_prob_trades[['symbol', 'probability', 'risk_reward_ratio', 'pattern_type', 'timestamp']], 
                        use_container_width=True)
        else:
            st.info("No high probability trades found recently.")

def show_trade_opportunities():
    """Show trade opportunities analysis"""
    
    st.header("🔍 Trade Opportunities Analysis")
    
    trade_df = get_trade_opportunities()
    
    if trade_df.empty:
        st.warning("⚠️ No trade opportunities data available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_probability = st.slider("Min Probability (%)", 0, 100, 50)
    
    with col2:
        selected_scanner = st.selectbox("Scanner Type", ["All"] + list(trade_df['scan_type'].unique()))
    
    with col3:
        selected_pattern = st.selectbox("Pattern Type", ["All"] + list(trade_df['pattern_type'].unique()))
    
    # Filter data
    filtered_df = trade_df.copy()
    filtered_df = filtered_df[filtered_df['probability'] >= min_probability]
    
    if selected_scanner != "All":
        filtered_df = filtered_df[filtered_df['scan_type'] == selected_scanner]
    
    if selected_pattern != "All":
        filtered_df = filtered_df[filtered_df['pattern_type'] == selected_pattern]
    
    # Display filtered results
    st.subheader(f"📊 Filtered Results ({len(filtered_df)} opportunities)")
    st.dataframe(filtered_df[['symbol', 'probability', 'risk_reward_ratio', 'pattern_type', 'scan_type', 'timestamp']], 
                use_container_width=True)
    
    # Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Probability Distribution")
        fig = px.histogram(filtered_df, x='probability', nbins=20, 
                          title='Trade Probability Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk/Reward Analysis")
        fig = px.scatter(filtered_df, x='probability', y='risk_reward_ratio', 
                        color='pattern_type', title='Probability vs Risk/Reward')
        st.plotly_chart(fig, use_container_width=True)

def show_backtest_results():
    """Show backtest results analysis"""
    
    st.header("📊 Backtest Results Analysis")
    
    backtest_df = get_backtest_results()
    
    if backtest_df.empty:
        st.warning("⚠️ No backtest results available.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_win_rate = backtest_df['win_rate'].mean()
        st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
    
    with col2:
        avg_profit = backtest_df['avg_profit'].mean()
        st.metric("Average Profit", f"{avg_profit:.2f}%")
    
    with col3:
        avg_drawdown = backtest_df['max_drawdown'].mean()
        st.metric("Average Max Drawdown", f"{avg_drawdown:.2f}%")
    
    with col4:
        avg_sharpe = backtest_df['sharpe_ratio'].mean()
        st.metric("Average Sharpe Ratio", f"{avg_sharpe:.2f}")
    
    # Backtest results table
    st.subheader("📋 Backtest Results")
    st.dataframe(backtest_df, use_container_width=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Win Rate by Strategy")
        fig = px.bar(backtest_df, x='strategy_name', y='win_rate', 
                    title='Win Rate by Strategy')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Profit vs Drawdown")
        fig = px.scatter(backtest_df, x='avg_profit', y='max_drawdown', 
                        color='strategy_name', title='Profit vs Risk')
        st.plotly_chart(fig, use_container_width=True)

def show_scanner_performance():
    """Show scanner performance analysis"""
    
    st.header("⚙️ Scanner Performance Analysis")
    
    scan_df = get_scan_results()
    
    if scan_df.empty:
        st.warning("⚠️ No scanner performance data available.")
        return
    
    # Scanner performance metrics
    scanner_stats = scan_df.groupby('scan_type').agg({
        'total_coins_analyzed': 'sum',
        'premium_trades_found': 'sum',
        'execution_time_seconds': 'mean'
    }).reset_index()
    
    scanner_stats['opportunity_rate'] = (scanner_stats['premium_trades_found'] / 
                                       scanner_stats['total_coins_analyzed'] * 100)
    
    # Display scanner stats
    st.subheader("📊 Scanner Performance Summary")
    st.dataframe(scanner_stats, use_container_width=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Opportunity Rate by Scanner")
        fig = px.bar(scanner_stats, x='scan_type', y='opportunity_rate', 
                    title='Opportunity Rate by Scanner Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Execution Time by Scanner")
        fig = px.bar(scan_df, x='scan_type', y='execution_time_seconds', 
                    title='Average Execution Time')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("📈 Scanner Activity Over Time")
    fig = px.line(scan_df, x='scan_timestamp', y='premium_trades_found', 
                  color='scan_type', title='Premium Trades Found Over Time')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 