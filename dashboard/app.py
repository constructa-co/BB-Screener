import streamlit as st
import pandas as pd
from database import get_scanner_data
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="BB-Screener Dashboard", layout="wide")

st.title("🎯 BB-Screener Trading Signals Dashboard")

# Sidebar for scanner selection
scanner = st.sidebar.selectbox(
    "Select Scanner",
    ["FVG", "Flagpole", "Elliott Wave", "Fibonacci"]
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "1h", "4h", "1d"]
)

# Fetch and display data
st.header(f"{scanner} Signals - {timeframe}")

df = get_scanner_data(scanner, timeframe)

if not df.empty:
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals (24h)", len(df))
    with col2:
        if 'detected_at' in df.columns:
            latest = df['detected_at'].iloc[0] if len(df) > 0 else 'N/A'
            st.metric("Latest Signal", latest)
    with col3:
        unique_symbols = df['symbol'].nunique() if 'symbol' in df.columns else 0
        st.metric("Unique Symbols", unique_symbols)
    
    # Display data table
    st.subheader("Recent Signals")
    st.dataframe(df, use_container_width=True)
    
    # Signal distribution chart
    if 'symbol' in df.columns:
        st.subheader("Signal Distribution by Symbol")
        symbol_counts = df['symbol'].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(x=symbol_counts.index, y=symbol_counts.values)
        ])
        fig.update_layout(
            title="Top 10 Most Active Symbols",
            xaxis_title="Symbol",
            yaxis_title="Signal Count"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"No {scanner} signals found in the last 24 hours")

# Add refresh button
if st.button("🔄 Refresh Data"):
    st.rerun()
