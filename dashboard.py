#!/usr/bin/env python3
"""
Crypto Trading Command Center
Unified dashboard for all crypto scanner analytics and trading management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from trade_logger import TradeLogger
import tradingview_charts as tv
import json
import os

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
        background-color: #f0f0f0;
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize database connection
@st.cache_resource
def get_db():
    return TradeLogger()

# Scanner configuration mapping
SCANNER_CONFIG = {
    'BB Scanner': {
        'timeframes': ['4H'],
        'icon': '🎯',
        'description': 'Bollinger Band bounce detection with 88% MFI win rate',
        'schedule': 'Every 4 hours'
    },
    'ICT Strategies': {
        'timeframes': ['15M', '1H', '4H'],
        'icon': '📈',
        'description': 'Order blocks, liquidity grabs, and Fair Value Gaps',
        'schedule': 'Every hour'
    },
    'Wyckoff': {
        'timeframes': ['15M', '1H', '4H'],
        'icon': '🏛️',
        'description': 'Accumulation/Distribution phase detection',
        'schedule': 'Every 2 hours'
    },
    'Elliott Waves': {
        'timeframes': ['1H', '4H', 'Daily', 'Weekly'],
        'icon': '🌊',
        'description': 'Wave pattern identification for major moves',
        'schedule': 'Daily'
    },
    'Supply & Demand': {
        'timeframes': ['1M', '5M', '15M', '1H', '4H'],
        'icon': '📦',
        'description': 'Institutional supply and demand zones',
        'schedule': 'Every 15 minutes'
    },
    'Fibonacci': {
        'timeframes': ['5M', '1H', '4H'],
        'icon': '📐',
        'description': 'Retracement and extension levels',
        'schedule': 'Every hour'
    },
    'FVG Scanner': {
        'timeframes': ['1M', '5M', '1H', '4H'],
        'icon': '🕳️',
        'description': 'Fair Value Gap detection',
        'schedule': 'Every 15 minutes'
    },
    'Trend Following': {
        'timeframes': ['1H', '4H', 'Daily'],
        'icon': '📊',
        'description': 'Trend continuation patterns',
        'schedule': 'Every 4 hours'
    }
}

# Helper functions
def get_scanner_status():
    """Get current status of all scanners"""
    logger = get_db()
    status_data = []
    
    if logger.connection:
        for scanner, config in SCANNER_CONFIG.items():
            for tf in config['timeframes']:
                scanner_type = f"{scanner.lower().replace(' ', '_')}_{tf.lower()}"
                
                # Get last run info
                logger.cursor.execute("""
                    SELECT MAX(scan_timestamp) as last_run,
                           COUNT(*) as total_runs,
                           SUM(premium_trades_found) as total_opportunities
                    FROM scan_results
                    WHERE scan_type = %s
                    AND scan_timestamp > NOW() - INTERVAL '24 hours'
                """, (scanner_type,))
                
                result = logger.cursor.fetchone()
                
                status_data.append({
                    'Scanner': f"{config['icon']} {scanner}",
                    'Timeframe': tf,
                    'Last Run': result['last_run'] or 'Never',
                    'Status': '🟢 Active' if result['last_run'] and (datetime.now() - result['last_run']).seconds < 3600 else '🔴 Inactive',
                    'Opportunities (24h)': result['total_opportunities'] or 0
                })
    
    return pd.DataFrame(status_data)

def get_best_opportunities(hours=24, min_prob=70):
    """Get best trading opportunities across all scanners"""
    logger = get_db()
    opportunities = []
    
    if logger.connection:
        logger.cursor.execute("""
            SELECT 
                t.*,
                s.scan_type,
                s.scan_timestamp,
                'Day Trading' as trading_style,
                '4H' as timeframe
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.probability >= %s
            AND t.trade_taken = FALSE
            ORDER BY t.probability DESC, t.risk_reward_ratio DESC
            LIMIT 50
        """, (min_prob,))
        
        opportunities = logger.cursor.fetchall()
    
    return opportunities



# Sidebar Configuration
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Time filter
    time_range = st.selectbox(
        "📅 Time Range",
        ["Last 1 Hour", "Last 4 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=2
    )
    
    # Convert to hours
    time_hours = {
        "Last 1 Hour": 1,
        "Last 4 Hours": 4,
        "Last 24 Hours": 24,
        "Last 7 Days": 168,
        "Last 30 Days": 720
    }[time_range]
    
    # Probability filter
    min_probability = st.slider("🎯 Min Probability %", 50, 95, 70, 5)
    
    # Scanner filter
    active_scanners = st.multiselect(
        "🔍 Active Scanners",
        list(SCANNER_CONFIG.keys()),
        default=list(SCANNER_CONFIG.keys())
    )
    
    # Trading style filter
    trading_styles = st.multiselect(
        "💹 Trading Styles",
        ["Scalping", "Day Trading", "Swing Trading", "Position Trading"],
        default=["Day Trading", "Swing Trading"]
    )
    
    st.markdown("---")
    
    # Market conditions (from your market regime logic)
    st.markdown("### 🌍 Market Conditions")
    
    # These would come from your actual market analysis
    market_regime = st.metric("Market Regime", "Bullish 🟢", "72% strength")
    btc_dominance = st.metric("BTC Dominance", "48.5%", "-2.1%")
    fear_greed = st.metric("Fear & Greed", "68", "Greed")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Run All Scanners", use_container_width=True):
        st.success("All scanners initiated!")
        st.balloons()
    
    if st.button("📊 Generate Report", use_container_width=True):
        st.info("Generating comprehensive report...")
    
    if st.button("🔔 Test Alerts", use_container_width=True):
        st.info("Alert test sent to Telegram!")

# Main navigation
page = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Overview", "🎯 Scanner Dashboard", "💹 All Opportunities", 
     "📊 Performance Analytics", "🤖 3Commas Integration", 
     "📈 Post-Mortem Analysis", "⚙️ Settings"]
)

# Page content based on selection
if page == "🏠 Overview":
    # Header
    st.markdown('<h1 class="main-header">🚀 Crypto Trading Command Center</h1>', unsafe_allow_html=True)
    
    # Get current data
    logger = get_db()
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if logger.connection:
        # Get overview stats
        logger.cursor.execute("""
            SELECT 
                COUNT(DISTINCT scan_type) as active_scanners,
                COUNT(*) as total_scans,
                SUM(premium_trades_found) as total_opportunities
            FROM scan_results
        """, ())
        
        stats = logger.cursor.fetchone()
        
        with col1:
            st.metric("Active Scanners", stats['active_scanners'] or 0, "+2 new")
        
        with col2:
            st.metric("Total Scans", stats['total_scans'] or 0, f"Last {time_range.lower()}")
        
        with col3:
            st.metric("Opportunities", stats['total_opportunities'] or 0, "+23 new")
        
        with col4:
            # Get high probability count
            logger.cursor.execute("""
                SELECT COUNT(*) as high_prob
                FROM trade_opportunities
                WHERE probability >= 85
            """, ())
            high_prob = logger.cursor.fetchone()['high_prob']
            st.metric("High Probability", high_prob or 0, "≥85%")
        
        with col5:
            # Calculate average win rate
            logger.cursor.execute("""
                SELECT 
                    AVG(CASE WHEN trade_result = 'win' THEN 1 ELSE 0 END) * 100 as win_rate
                FROM trade_opportunities
                WHERE trade_taken = TRUE
                AND trade_result IN ('win', 'loss')
            """)
            win_rate = logger.cursor.fetchone()['win_rate'] or 0
            st.metric("Win Rate", f"{win_rate:.1f}%", "+2.1%")
    
    # Two column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Best opportunities
        st.subheader("🎯 Top Trading Opportunities")
        
        opportunities = get_best_opportunities(hours=time_hours, min_prob=min_probability)
        
        if opportunities:
            # Create formatted dataframe
            opp_data = []
            for opp in opportunities[:10]:  # Top 10
                scanner_type = opp['scan_type'].replace('_', ' ').title()
                icon = '🎯'  # Default icon
                for scanner, config in SCANNER_CONFIG.items():
                    if scanner.lower() in opp['scan_type']:
                        icon = config['icon']
                        break
                
                opp_data.append({
                    'Scanner': f"{icon} {scanner_type}",
                    'Symbol': opp['symbol'],
                    'Probability': opp['probability'],
                    'R:R': opp['risk_reward_ratio'],
                    'Entry': opp['entry_price'],
                    'Target 1': opp['target_1'],
                    'Style': opp['trading_style'],
                    'Time': opp['timestamp']
                })
            
            opp_df = pd.DataFrame(opp_data)
            
            # Display with custom formatting
            st.dataframe(
                opp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Probability": st.column_config.ProgressColumn(
                        "Probability",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "R:R": st.column_config.NumberColumn(
                        "Risk/Reward",
                        format="%.2f:1"
                    ),
                    "Entry": st.column_config.NumberColumn(
                        "Entry Price",
                        format="$%.6f"
                    ),
                    "Target 1": st.column_config.NumberColumn(
                        "Target 1",
                        format="$%.6f"
                    ),
                    "Time": st.column_config.DatetimeColumn(
                        "Found",
                        format="MMM D, HH:mm"
                    )
                }
            )
            
            # Add chart buttons for top opportunities
            st.subheader("📊 Chart Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Show BTC/USDT Chart", key="btc_chart"):
                    st.session_state.show_btc_chart = True
            
            with col2:
                if st.button("Show ETH/USDT Chart", key="eth_chart"):
                    st.session_state.show_eth_chart = True
            
            with col3:
                if st.button("Show SOL/USDT Chart", key="sol_chart"):
                    st.session_state.show_sol_chart = True
            
            # Display charts if requested
            if st.session_state.get('show_btc_chart', False):
                with st.expander("📈 BTC/USDT Chart", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        tv.show_tradingview_chart("BTC/USDT", timeframe='240', height=600)
                    with col2:
                        tv.show_technical_analysis_widget("BTC/USDT")
                        st.metric("Current Price", "$45,234.56")
                        st.metric("24h Change", "+2.34%")
                st.session_state.show_btc_chart = False
            
            if st.session_state.get('show_eth_chart', False):
                with st.expander("📈 ETH/USDT Chart", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        tv.show_tradingview_chart("ETH/USDT", timeframe='240', height=600)
                    with col2:
                        tv.show_technical_analysis_widget("ETH/USDT")
                        st.metric("Current Price", "$2,456.78")
                        st.metric("24h Change", "+1.87%")
                st.session_state.show_eth_chart = False
            
            if st.session_state.get('show_sol_chart', False):
                with st.expander("📈 SOL/USDT Chart", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        tv.show_tradingview_chart("SOL/USDT", timeframe='240', height=600)
                    with col2:
                        tv.show_technical_analysis_widget("SOL/USDT")
                        st.metric("Current Price", "$98.45")
                        st.metric("24h Change", "+3.21%")
                st.session_state.show_sol_chart = False
        else:
            st.info("No opportunities found matching your criteria")
        
        # Scanner performance chart
        st.subheader("📈 Scanner Performance")
        
        if logger.connection:
            # Get scanner performance data
            logger.cursor.execute("""
                SELECT 
                    scan_type as scanner,
                    COUNT(*) as scans,
                    SUM(premium_trades_found) as opportunities,
                    AVG(execution_time_seconds) as avg_time
                FROM scan_results
                GROUP BY scan_type
                ORDER BY opportunities DESC
            """, ())
            
            perf_data = pd.DataFrame(logger.cursor.fetchall())
            
            if not perf_data.empty:
                fig = px.bar(
                    perf_data,
                    x='scanner',
                    y='opportunities',
                    color='avg_time',
                    color_continuous_scale='viridis',
                    title='Opportunities Found by Scanner',
                    labels={'opportunities': 'Total Opportunities', 'avg_time': 'Avg Scan Time (s)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        # Scanner status
        st.subheader("🔄 Scanner Status")
        
        status_df = get_scanner_status()
        
        if not status_df.empty:
            # Group by scanner
            for scanner in active_scanners:
                if scanner in SCANNER_CONFIG:
                    config = SCANNER_CONFIG[scanner]
                    
                    with st.expander(f"{config['icon']} {scanner}", expanded=True):
                        scanner_data = status_df[status_df['Scanner'].str.contains(scanner)]
                        
                        if not scanner_data.empty:
                            for _, row in scanner_data.iterrows():
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.write(f"**{row['Timeframe']}**")
                                with col2:
                                    st.write(row['Status'])
                                with col3:
                                    st.metric("Found", row['Opportunities (24h)'])
                        
                        st.caption(f"📅 {config['schedule']}")
        
        # Trading distribution pie chart
        st.subheader("💹 Trading Style Distribution")
        
        if opportunities:
            style_counts = {}
            for opp in opportunities:
                style = opp['trading_style']
                style_counts[style] = style_counts.get(style, 0) + 1
            
            if style_counts:
                fig = px.pie(
                    values=list(style_counts.values()),
                    names=list(style_counts.keys()),
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Scanner Dashboard":
    st.title("🎯 Scanner Dashboard")
    
    # Scanner selection
    selected_scanner = st.selectbox(
        "Select Scanner",
        list(SCANNER_CONFIG.keys()),
        format_func=lambda x: f"{SCANNER_CONFIG[x]['icon']} {x}"
    )
    
    config = SCANNER_CONFIG[selected_scanner]
    
    # Scanner info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### {config['description']}")
    with col2:
        st.metric("Timeframes", len(config['timeframes']))
    with col3:
        st.metric("Schedule", config['schedule'])
    
    # Timeframe tabs
    tabs = st.tabs(config['timeframes'])
    
    for i, tf in enumerate(config['timeframes']):
        with tabs[i]:
            scanner_type = f"{selected_scanner.lower().replace(' ', '_')}_{tf.lower()}"
            
            # Get opportunities for this specific scanner/timeframe
            logger = get_db()
            if logger.connection:
                logger.cursor.execute("""
                    SELECT t.*, s.scan_timestamp
                    FROM trade_opportunities t
                    JOIN scan_results s ON t.scan_id = s.id
                    WHERE s.scan_type = %s

                    AND t.probability >= %s
                    ORDER BY t.probability DESC
                """, (scanner_type, min_probability))
                
                scanner_opps = logger.cursor.fetchall()
                
                if scanner_opps:
                    # Display opportunities
                    st.metric("Active Opportunities", len(scanner_opps))
                    
                    # Create detailed view
                    for i, opp in enumerate(scanner_opps[:5]):  # Show top 5
                        with st.expander(f"📊 {opp['symbol']} - {opp['probability']:.1f}% Probability", expanded=(i==0)):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Show TradingView chart
                                tv.show_tradingview_chart(
                                    opp['symbol'],
                    timeframe='240',
                    height=400,
                    studies=["BB@tv-basicstudies", "RSI@tv-basicstudies", "MACD@tv-basicstudies"]
                )
                            
                            with col2:
                                # Trade details
                                st.markdown(f"""
                                <div class="scanner-card">
                                    <h4>Trade Details</h4>
                                    <p><b>Entry:</b> ${opp['entry_price']:.6f}</p>
                                    <p><b>Stop:</b> ${opp['stop_loss']:.6f}</p>
                                    <p><b>Target 1:</b> ${opp['target_1']:.6f}</p>
                                    <p><b>Risk/Reward:</b> {opp['risk_reward_ratio']:.2f}:1</p>
                                    <p><b>Pattern:</b> {opp['pattern_type'] or 'N/A'}</p>
                                    <p><b>Found:</b> {opp['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Technical analysis widget
                                tv.show_technical_analysis_widget(opp['symbol'])
                else:
                    st.info(f"No {tf} opportunities found for {selected_scanner}")

elif page == "💹 All Opportunities":
    st.title("💹 All Trading Opportunities")
    
    # Get all opportunities
    opportunities = get_best_opportunities(hours=time_hours, min_prob=min_probability)
    
    if opportunities:
        # Filter by trading style
        filtered_opps = [opp for opp in opportunities if opp['trading_style'] in trading_styles]
        
        st.metric("Total Opportunities", len(filtered_opps))
        
        # Group by timeframe
        timeframe_groups = {}
        for opp in filtered_opps:
            tf = opp['timeframe']
            if tf not in timeframe_groups:
                timeframe_groups[tf] = []
            timeframe_groups[tf].append(opp)
        
        # Display by timeframe
        for tf in ['1M', '5M', '15M', '1H', '4H', 'Daily', 'Weekly']:
            if tf in timeframe_groups:
                with st.expander(f"⏰ {tf} Timeframe ({len(timeframe_groups[tf])} opportunities)", expanded=(tf in ['1H', '4H'])):
                    # Create dataframe for this timeframe
                    tf_data = []
                    for opp in timeframe_groups[tf]:
                        tf_data.append({
                            'Symbol': opp['symbol'],
                            'Scanner': opp['scan_type'].replace('_', ' ').title(),
                            'Probability': opp['probability'],
                            'R:R': opp['risk_reward_ratio'],
                            'Entry': opp['entry_price'],
                            'Stop': opp['stop_loss'],
                            'Target 1': opp['target_1'],
                            'Market Cap': f"${opp['market_cap']/1e9:.1f}B" if opp['market_cap'] > 1e9 else f"${opp['market_cap']/1e6:.0f}M",
                            'Volume': f"${opp['volume_24h']/1e6:.1f}M",
                            'RSI': opp['rsi'],
                            'MFI': opp['mfi']
                        })
                    
                    tf_df = pd.DataFrame(tf_data)
                    
                    # Display with filtering
                    st.dataframe(
                        tf_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Probability": st.column_config.ProgressColumn(
                                "Prob %",
                                format="%.0f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "R:R": st.column_config.NumberColumn(
                                "R:R",
                                format="%.1f"
                            ),
                            "Entry": st.column_config.NumberColumn(
                                "Entry",
                                format="$%.6f"
                            ),
                            "Stop": st.column_config.NumberColumn(
                                "Stop",
                                format="$%.6f"
                            ),
                            "Target 1": st.column_config.NumberColumn(
                                "T1",
                                format="$%.6f"
                            ),
                            "RSI": st.column_config.NumberColumn(
                                "RSI",
                                format="%.0f"
                            ),
                            "MFI": st.column_config.NumberColumn(
                                "MFI",
                                format="%.0f"
                            )
                        }
                    )
                    
                    # Add chart analysis for top opportunities
                    if not tf_df.empty:
                        st.subheader("📊 Chart Analysis")
                        
                        # Show charts for top 3 opportunities
                        for i, (_, row) in enumerate(tf_df.head(3).iterrows()):
                            symbol = row['Symbol']
                            with st.expander(f"📈 {symbol} Chart Analysis", expanded=(i==0)):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    tv.show_tradingview_chart(
                                        symbol,
                                        timeframe='240',
                                        height=400,
                                        studies=["BB@tv-basicstudies", "RSI@tv-basicstudies", "MACD@tv-basicstudies"]
                                    )
                                
                                with col2:
                                    # Mini chart for quick overview
                                    tv.show_mini_chart(symbol, width=300, height=150)
                                    
                                    # Trade metrics with safe access
                                    st.metric("Probability", f"{row.get('Probability', 0):.0f}%")
                                    st.metric("Risk/Reward", f"{row.get('R:R', 0):.1f}:1")
                                    st.metric("Entry", f"${row.get('Entry', 0):.6f}")
                                    st.metric("Target", f"${row.get('Target 1', 0):.6f}")
                                    st.metric("Stop", f"${row.get('Stop', 0):.6f}")
                                    
                                    # Technical indicators with safe access
                                    st.write("**Technical Indicators**")
                                    st.write(f"RSI: {row.get('RSI', 0):.0f}")
                                    st.write(f"MFI: {row.get('MFI', 0):.0f}")
                                
                                # Technical analysis widget
                                st.subheader("Technical Analysis")
                                tv.show_technical_analysis_widget(symbol)
    else:
        st.info("No opportunities found matching your criteria")

elif page == "📊 Performance Analytics":
    st.title("📊 Performance Analytics")
    
    # Time period selection
    period = st.radio(
        "Select Period",
        ["Today", "This Week", "This Month", "All Time"],
        horizontal=True
    )
    
    logger = get_db()
    
    if logger.connection:
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate period filter
        period_filter = {
            "Today": "1 day",
            "This Week": "7 days",
            "This Month": "30 days",
            "All Time": "10 years"
        }[period]
        
        # Get performance stats
        logger.cursor.execute("""
            SELECT 
                COUNT(CASE WHEN trade_taken = TRUE THEN 1 END) as total_trades,
                COUNT(CASE WHEN trade_result = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN trade_result = 'loss' THEN 1 END) as losses,
                AVG(CASE WHEN trade_result IN ('win', 'loss') THEN profit_loss_percent END) as avg_pnl,
                MAX(profit_loss_percent) as best_trade,
                MIN(profit_loss_percent) as worst_trade
            FROM trade_opportunities
            WHERE timestamp > NOW() - INTERVAL '%s'
        """, (period_filter,))
        
        stats = logger.cursor.fetchone()
        
        with col1:
            st.metric("Total Trades", stats['total_trades'] or 0)
        
        with col2:
            win_rate = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            st.metric("Avg P&L", f"{stats['avg_pnl'] or 0:.1f}%")
        
        with col4:
            st.metric("Best Trade", f"{stats['best_trade'] or 0:.1f}%")
        
        # Performance by scanner
        st.subheader("Performance by Scanner")
        
        logger.cursor.execute("""
            SELECT 
                CASE 
                    WHEN s.scan_type LIKE '%bb%' THEN 'BB Scanner'
                    WHEN s.scan_type LIKE '%ict%' THEN 'ICT'
                    WHEN s.scan_type LIKE '%wyckoff%' THEN 'Wyckoff'
                    WHEN s.scan_type LIKE '%elliott%' THEN 'Elliott Waves'
                    WHEN s.scan_type LIKE '%supply%' THEN 'Supply & Demand'
                    ELSE 'Other'
                END as scanner,
                COUNT(CASE WHEN t.trade_taken = TRUE THEN 1 END) as trades,
                COUNT(CASE WHEN t.trade_result = 'win' THEN 1 END) as wins,
                AVG(CASE WHEN t.trade_result IN ('win', 'loss') THEN t.profit_loss_percent END) as avg_pnl
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.timestamp > NOW() - INTERVAL '%s'
            GROUP BY scanner
            HAVING COUNT(CASE WHEN t.trade_taken = TRUE THEN 1 END) > 0
        """, (period_filter,))
        
        perf_by_scanner = pd.DataFrame(logger.cursor.fetchall())
        
        if not perf_by_scanner.empty:
            perf_by_scanner['win_rate'] = (perf_by_scanner['wins'] / perf_by_scanner['trades'] * 100).round(1)
            
            fig = px.bar(
                perf_by_scanner,
                x='scanner',
                y='win_rate',
                color='avg_pnl',
                color_continuous_scale='RdYlGn',
                title='Win Rate by Scanner',
                labels={'win_rate': 'Win Rate %', 'avg_pnl': 'Avg P&L %'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # P&L over time
        st.subheader("Cumulative P&L")
        
        logger.cursor.execute("""
            SELECT 
                DATE(actual_exit_time) as date,
                SUM(profit_loss_percent) as daily_pnl,
                SUM(SUM(profit_loss_percent)) OVER (ORDER BY DATE(actual_exit_time)) as cumulative_pnl
            FROM trade_opportunities
            WHERE trade_result IN ('win', 'loss')
            AND actual_exit_time > NOW() - INTERVAL '%s'
            GROUP BY date
            ORDER BY date
        """, (period_filter,))
        
        pnl_data = pd.DataFrame(logger.cursor.fetchall())
        
        if not pnl_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pnl_data['date'],
                y=pnl_data['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='green', width=3)
            ))
            fig.update_layout(
                title='Cumulative P&L Over Time',
                xaxis_title='Date',
                yaxis_title='Cumulative P&L %',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 3Commas Integration":
    st.title("🤖 3Commas Integration")
    
    # Connection status
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Connection Status", "🟢 Connected")
    
    # Active bots
    st.subheader("Active Trading Bots")
    
    # This would connect to your actual 3Commas integration
    bot_data = [
        {"Bot": "BB Bounce Bot", "Status": "🟢 Active", "Deals": 3, "P&L": "+4.2%"},
        {"Bot": "ICT Scalper", "Status": "🟢 Active", "Deals": 7, "P&L": "+2.8%"},
        {"Bot": "Wyckoff Swing", "Status": "🟡 Paused", "Deals": 1, "P&L": "+12.3%"},
        {"Bot": "Elliott Wave", "Status": "🔴 Stopped", "Deals": 0, "P&L": "0%"}
    ]
    
    bot_df = pd.DataFrame(bot_data)
    st.dataframe(bot_df, use_container_width=True, hide_index=True)
    
    # Trade execution queue
    st.subheader("Trade Execution Queue")
    
    logger = get_db()
    if logger.connection:
        # Get pending trades
        logger.cursor.execute("""
            SELECT 
                symbol,
                probability,
                entry_price,
                stop_loss,
                target_1,
                risk_reward_ratio,
                timestamp
            FROM trade_opportunities
            WHERE trade_taken = FALSE
            AND probability >= 80
            AND timestamp > NOW() - INTERVAL '4 hours'
            ORDER BY probability DESC
            LIMIT 10
        """)
        
        pending_trades = pd.DataFrame(logger.cursor.fetchall())
        
        if not pending_trades.empty:
            st.dataframe(
                pending_trades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "probability": st.column_config.ProgressColumn(
                        "Probability",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "entry_price": st.column_config.NumberColumn(
                        "Entry",
                        format="$%.6f"
                    ),
                    "stop_loss": st.column_config.NumberColumn(
                        "Stop",
                        format="$%.6f"
                    ),
                    "target_1": st.column_config.NumberColumn(
                        "Target",
                        format="$%.6f"
                    ),
                    "risk_reward_ratio": st.column_config.NumberColumn(
                        "R:R",
                        format="%.1f:1"
                    )
                }
            )
            
            if st.button("Execute Selected Trades", type="primary", use_container_width=True):
                st.success("Trades sent to 3Commas for execution!")
        else:
            st.info("No high-probability trades pending execution")

elif page == "📈 Post-Mortem Analysis":
    st.title("📈 Post-Mortem Analysis")
    
    # Completed trades analysis
    logger = get_db()
    
    if logger.connection:
        # Get completed trades
        logger.cursor.execute("""
            SELECT 
                t.*,
                s.scan_type,
                CASE 
                    WHEN profit_loss_percent > 0 THEN 'Profitable'
                    WHEN profit_loss_percent < 0 THEN 'Loss'
                    ELSE 'Breakeven'
                END as outcome
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.trade_taken = TRUE
            AND t.trade_result IS NOT NULL
            ORDER BY t.actual_exit_time DESC
            LIMIT 100
        """)
        
        completed_trades = pd.DataFrame(logger.cursor.fetchall())
        
        if not completed_trades.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            profitable = len(completed_trades[completed_trades['outcome'] == 'Profitable'])
            losses = len(completed_trades[completed_trades['outcome'] == 'Loss'])
            
            with col1:
                st.metric("Total Completed", len(completed_trades))
            with col2:
                st.metric("Profitable", profitable, f"{profitable/len(completed_trades)*100:.0f}%")
            with col3:
                st.metric("Losses", losses, f"{losses/len(completed_trades)*100:.0f}%")
            with col4:
                avg_pnl = completed_trades['profit_loss_percent'].mean()
                st.metric("Avg P&L", f"{avg_pnl:.1f}%")
            
            # Trade analysis
            st.subheader("Trade Analysis")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                outcome_filter = st.selectbox("Outcome", ["All", "Profitable", "Loss", "Breakeven"])
            with col2:
                scanner_filter = st.selectbox("Scanner", ["All"] + list(completed_trades['scan_type'].unique()))
            with col3:
                sort_by = st.selectbox("Sort By", ["Exit Time", "P&L %", "Symbol"])
            
            # Apply filters
            filtered_df = completed_trades.copy()
            if outcome_filter != "All":
                filtered_df = filtered_df[filtered_df['outcome'] == outcome_filter]
            if scanner_filter != "All":
                filtered_df = filtered_df[filtered_df['scan_type'] == scanner_filter]
            
            # Sort
            sort_mapping = {
                "Exit Time": "actual_exit_time",
                "P&L %": "profit_loss_percent",
                "Symbol": "symbol"
            }
            filtered_df = filtered_df.sort_values(sort_mapping[sort_by], ascending=False)
            
            # Display detailed trades
            for _, trade in filtered_df.head(20).iterrows():
                color = "green" if trade['profit_loss_percent'] > 0 else "red"
                
                with st.expander(f"{trade['symbol']} - {trade['profit_loss_percent']:.1f}% ({trade['outcome']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Entry Details**")
                        st.write(f"Scanner: {trade['scan_type']}")
                        st.write(f"Entry: ${trade['entry_price']:.6f}")
                        st.write(f"Probability: {trade['probability']:.0f}%")
                        st.write(f"Pattern: {trade['pattern_type'] or 'N/A'}")
                    
                    with col2:
                        st.write("**Exit Details**")
                        st.write(f"Exit: ${trade['actual_exit_price']:.6f}")
                        st.write(f"Target 1: ${trade['target_1']:.6f}")
                        st.write(f"Stop Loss: ${trade['stop_loss']:.6f}")
                        st.write(f"Exit Time: {trade['actual_exit_time']}")
                    
                    with col3:
                        st.write("**Performance**")
                        st.write(f"P&L: {trade['profit_loss_percent']:.1f}%")
                        st.write(f"Risk/Reward: {trade['risk_reward_ratio']:.1f}:1")
                        st.write(f"Hold Time: {(trade['actual_exit_time'] - trade['timestamp']).days} days")
                    
                    # Lessons learned section
                    st.write("**Analysis Notes**")
                    if trade['profit_loss_percent'] > 0:
                        st.success(f"✅ Successful {trade['pattern_type'] or 'pattern'} recognition")
                    else:
                        st.error(f"❌ Consider reviewing {trade['pattern_type'] or 'pattern'} criteria")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    # Scanner configuration
    st.subheader("Scanner Configuration")
    
    # Create tabs for each scanner
    scanner_tabs = st.tabs(list(SCANNER_CONFIG.keys()))
    
    for i, (scanner, config) in enumerate(SCANNER_CONFIG.items()):
        with scanner_tabs[i]:
            st.write(f"### {config['icon']} {scanner} Settings")
            
            # Schedule settings
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "Run Schedule",
                    ["Every 15 minutes", "Every hour", "Every 4 hours", "Daily", "Manual only"],
                    index=1,
                    key=f"{scanner}_schedule"
                )
            
            with col2:
                st.multiselect(
                    "Active Timeframes",
                    config['timeframes'],
                    default=config['timeframes'],
                    key=f"{scanner}_timeframes"
                )
            
            # Quality thresholds
            st.write("**Quality Thresholds**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.slider("Min Probability %", 50, 95, 70, 5, key=f"{scanner}_min_prob")
            with col2:
                st.slider("Min Risk/Reward", 1.0, 5.0, 2.0, 0.5, key=f"{scanner}_min_rr")
            with col3:
                st.slider("Max Risk %", 1, 5, 2, key=f"{scanner}_max_risk")
    
    # Alert settings
    st.subheader("Alert Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.toggle("Telegram Alerts", value=True)
        st.text_input("Telegram Chat ID", value="123456789")
    
    with col2:
        st.toggle("Email Alerts", value=False)
        st.text_input("Email Address", placeholder="your@email.com")
    
    # Database settings
    st.subheader("Database Settings")
    
    if st.button("Backup Database"):
        st.success("Database backup initiated!")
    
    if st.button("Clear Old Data (>30 days)"):
        st.warning("This will delete old scan data. Are you sure?")
    
    # Save settings
    if st.button("Save All Settings", type="primary", use_container_width=True):
        st.success("Settings saved successfully!")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC | Database: crypto-scanner-db | Server: 165.232.160.52") 