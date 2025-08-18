#!/usr/bin/env python3
"""
Crypto Trading Command Center - Fixed Version
Unified dashboard for all crypto scanner analytics and trading management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os
import json
import time
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    }
    .scanner-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .opportunity-row {
        transition: all 0.3s ease;
    }
    .opportunity-row:hover {
        background-color: #f8f9fa;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def get_database_connection():
    """Get database connection with fallback"""
    try:
        import psycopg2
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Try to get database URL
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            st.error("❌ DATABASE_URL not found in environment variables")
            return None
            
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

def get_other_scanners_connection():
    """Get connection to other_scanners database"""
    try:
        import psycopg2
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Try OTHER_SCANNERS_DATABASE_URL first
        DATABASE_URL = os.getenv('OTHER_SCANNERS_DATABASE_URL')
        if not DATABASE_URL:
            # Fallback to main DATABASE_URL with schema
            main_db_url = os.getenv('DATABASE_URL')
            if main_db_url:
                if '?' in main_db_url:
                    DATABASE_URL = main_db_url + '&options=-csearch_path=other_scanners'
                else:
                    DATABASE_URL = main_db_url + '?options=-csearch_path=other_scanners'
            else:
                st.error("❌ No database URL found")
                return None
        
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        st.error(f"❌ Other scanners database connection failed: {e}")
        return None

def load_main_scanner_data():
    """Load data from main scanner"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            symbol,
            timeframe,
            bb_score,
            probability,
            risk_reward_ratio,
            current_price,
            entry_price,
            stop_loss,
            target_1,
            pattern_type,
            pattern_quality,
            timestamp,
            scanner_specific_data
        FROM public.trade_opportunities
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ Error loading main scanner data: {e}")
        conn.close()
        return pd.DataFrame()

def load_other_scanners_data():
    """Load data from other scanners"""
    conn = get_other_scanners_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SET search_path TO other_scanners;
        SELECT 
            id,
            scanner_name,
            scanner_version,
            symbol,
            timeframe,
            side,
            entry_price,
            stop_loss,
            take_profit,
            status,
            created_at,
            technical_indicators,
            scanner_signals,
            market_conditions,
            execution_metadata
        FROM other_scanners_trades
        WHERE created_at > NOW() - INTERVAL '24 hours'
        ORDER BY created_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ Error loading other scanners data: {e}")
        conn.close()
        return pd.DataFrame()

def create_main_dashboard():
    """Create main dashboard page"""
    st.markdown('<h1 class="main-header">🚀 Crypto Trading Command Center</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading main scanner data..."):
        main_data = load_main_scanner_data()
    
    with st.spinner("Loading other scanners data..."):
        other_data = load_other_scanners_data()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Main Scanner</h3>
            <h2>{}</h2>
            <p>Trades (24h)</p>
        </div>
        """.format(len(main_data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Other Scanners</h3>
            <h2>{}</h2>
            <p>Trades (24h)</p>
        </div>
        """.format(len(other_data)), unsafe_allow_html=True)
    
    with col3:
        if not main_data.empty:
            avg_prob = main_data['probability'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Probability</h3>
                <h2>{:.1f}%</h2>
                <p>Main Scanner</p>
            </div>
            """.format(avg_prob), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Probability</h3>
                <h2>N/A</h2>
                <p>No Data</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if not other_data.empty:
            unique_scanners = other_data['scanner_name'].nunique()
            st.markdown("""
            <div class="metric-card">
                <h3>Active Scanners</h3>
                <h2>{}</h2>
                <p>Other Scanners</p>
            </div>
            """.format(unique_scanners), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>Active Scanners</h3>
                <h2>0</h2>
                <p>No Data</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main scanner opportunities
    if not main_data.empty:
        st.subheader("📊 Main Scanner Opportunities (Last 24h)")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            min_prob = st.slider("Min Probability", 0, 100, 70, key="main_prob")
        with col2:
            min_rr = st.slider("Min Risk/Reward", 0.0, 5.0, 1.0, key="main_rr")
        with col3:
            selected_patterns = st.multiselect(
                "Pattern Types",
                options=main_data['pattern_type'].unique(),
                default=main_data['pattern_type'].unique()[:5]
            )
        
        # Filter data
        filtered_data = main_data[
            (main_data['probability'] >= min_prob) &
            (main_data['risk_reward_ratio'] >= min_rr) &
            (main_data['pattern_type'].isin(selected_patterns))
        ]
        
        if not filtered_data.empty:
            # Display opportunities
            for _, row in filtered_data.head(10).iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{row['symbol']}** ({row['timeframe']})")
                        st.markdown(f"Pattern: {row['pattern_type']} - {row['pattern_quality']}")
                    
                    with col2:
                        st.metric("Probability", f"{row['probability']:.1f}%")
                    
                    with col3:
                        st.metric("R/R Ratio", f"{row['risk_reward_ratio']:.2f}")
                    
                    with col4:
                        st.metric("Entry Price", f"${row['entry_price']:.4f}")
                    
                    st.divider()
        else:
            st.info("No opportunities match the current filters")
    
    # Other scanners summary
    if not other_data.empty:
        st.subheader("🔧 Other Scanners Summary")
        
        # Scanner activity
        scanner_summary = other_data.groupby('scanner_name').agg({
            'id': 'count',
            'created_at': 'max',
            'symbol': 'nunique'
        }).rename(columns={'id': 'trades', 'created_at': 'last_run', 'symbol': 'unique_symbols'})
        
        st.dataframe(scanner_summary, use_container_width=True)
        
        # Recent trades
        st.subheader("📈 Recent Other Scanner Trades")
        recent_trades = other_data[['scanner_name', 'symbol', 'side', 'entry_price', 'created_at']].head(10)
        st.dataframe(recent_trades, use_container_width=True)

def create_analytics_page():
    """Create analytics page"""
    st.header("📊 Analytics Dashboard")
    
    # Load data
    main_data = load_main_scanner_data()
    other_data = load_other_scanners_data()
    
    if main_data.empty and other_data.empty:
        st.warning("No data available for analytics")
        return
    
    # Time series analysis
    if not main_data.empty:
        st.subheader("Main Scanner Activity Over Time")
        
        # Group by hour
        main_data['hour'] = pd.to_datetime(main_data['timestamp']).dt.floor('H')
        hourly_data = main_data.groupby('hour').size().reset_index(name='trades')
        
        fig = px.line(hourly_data, x='hour', y='trades', title='Trades per Hour')
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability distribution
        st.subheader("Probability Distribution")
        fig = px.histogram(main_data, x='probability', nbins=20, title='Probability Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    if not other_data.empty:
        st.subheader("Other Scanners Activity")
        
        # Scanner comparison
        scanner_counts = other_data['scanner_name'].value_counts()
        fig = px.bar(x=scanner_counts.index, y=scanner_counts.values, title='Trades by Scanner')
        st.plotly_chart(fig, use_container_width=True)
        
        # Side distribution
        side_counts = other_data['side'].value_counts()
        fig = px.pie(values=side_counts.values, names=side_counts.index, title='BUY vs SELL Distribution')
        st.plotly_chart(fig, use_container_width=True)

def create_settings_page():
    """Create settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("Database Configuration")
    
    # Show current database status
    main_conn = get_database_connection()
    other_conn = get_other_scanners_connection()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if main_conn:
            st.success("✅ Main Scanner Database: Connected")
            main_conn.close()
        else:
            st.error("❌ Main Scanner Database: Disconnected")
    
    with col2:
        if other_conn:
            st.success("✅ Other Scanners Database: Connected")
            other_conn.close()
        else:
            st.error("❌ Other Scanners Database: Disconnected")
    
    st.subheader("System Information")
    
    # Show system info
    info_data = {
        'Component': ['Streamlit Version', 'Pandas Version', 'Python Version', 'Current Time'],
        'Version': [
            st.__version__,
            pd.__version__,
            sys.version.split()[0],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    }
    
    st.dataframe(pd.DataFrame(info_data), use_container_width=True)

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Main Dashboard", "Analytics", "Settings"]
    )
    
    # Page routing
    if page == "Main Dashboard":
        create_main_dashboard()
    elif page == "Analytics":
        create_analytics_page()
    elif page == "Settings":
        create_settings_page()

if __name__ == "__main__":
    main()
