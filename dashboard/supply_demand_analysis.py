#!/usr/bin/env python3
"""
Supply & Demand Analysis Dashboard
Streamlit page for analyzing Supply & Demand scanner data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import time
import pytz

# Import TradeLogger for database connection
from trade_logger import TradeLogger

def get_db():
    """Get database connection using TradeLogger"""
    return TradeLogger()

def load_supply_demand_data():
    """Load Supply & Demand data from database"""
    try:
        logger = get_db()
        
        # Query for zones data
        zones_query = """
        SELECT 
            symbol,
            zone_type,
            zone_top,
            zone_bottom,
            zone_strength,
            touch_count,
            current_price,
            distance_to_zone,
            distance_percentage,
            price_position,
            quality_score,
            formation_type,
            freshness_score,
            reliability_score,
            zone_volume,
            average_volume,
            volume_ratio,
            volume_confirmation,
            formation_candles,
            formation_start,
            formation_end,
            algorithm_parameters,
            detected_at,
            expires_at,
            validation_status
        FROM other_scanners.supply_demand_zones
        WHERE detected_at > NOW() - INTERVAL '7 days'
        ORDER BY detected_at DESC
        """
        
        logger.cursor.execute(zones_query)
        zones_data = pd.DataFrame(logger.cursor.fetchall(), columns=[
            'symbol', 'zone_type', 'zone_top', 'zone_bottom', 'zone_strength',
            'touch_count', 'current_price', 'distance_to_zone', 'distance_percentage',
            'price_position', 'quality_score', 'formation_type', 'freshness_score',
            'reliability_score', 'zone_volume', 'average_volume', 'volume_ratio',
            'volume_confirmation', 'formation_candles', 'formation_start', 'formation_end',
            'algorithm_parameters', 'detected_at', 'expires_at', 'validation_status'
        ])
        
        # Convert timestamp columns and convert to UAE time (UTC+4)
        zones_data['detected_at'] = pd.to_datetime(zones_data['detected_at']).dt.tz_localize('UTC').dt.tz_convert('Asia/Dubai')
        zones_data['expires_at'] = pd.to_datetime(zones_data['expires_at']).dt.tz_localize('UTC').dt.tz_convert('Asia/Dubai')
        zones_data['formation_start'] = pd.to_datetime(zones_data['formation_start']).dt.tz_localize('UTC').dt.tz_convert('Asia/Dubai')
        zones_data['formation_end'] = pd.to_datetime(zones_data['formation_end']).dt.tz_localize('UTC').dt.tz_convert('Asia/Dubai')
        
        return zones_data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_performance_metrics(data):
    """Calculate performance metrics for S&D zones"""
    if data.empty:
        return None
    
    try:
        # Total zones in last 7 days
        total_zones = len(data)
        
        # Active zones (not expired)
        active_zones = len(data[data['validation_status'] == 'ACTIVE'])
        
        # Average quality score
        avg_quality = data['quality_score'].mean()
        
        # Zone type distribution
        zone_type_counts = data['zone_type'].value_counts()
        
        # Formation type distribution
        formation_counts = data['formation_type'].value_counts()
        
        # Average zone strength
        avg_strength = data['zone_strength'].mean()
        
        # Volume confirmation rate
        volume_confirmed = data['volume_confirmation'].sum()
        volume_confirmation_rate = (volume_confirmed / len(data)) * 100 if len(data) > 0 else 0
        
        # Price position distribution
        price_position_counts = data['price_position'].value_counts()
        
        return {
            'total_zones': total_zones,
            'active_zones': active_zones,
            'avg_quality': avg_quality,
            'zone_type_counts': zone_type_counts,
            'formation_counts': formation_counts,
            'avg_strength': avg_strength,
            'volume_confirmation_rate': volume_confirmation_rate,
            'price_position_counts': price_position_counts
        }
        
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return None

def show_supply_demand_analysis():
    """Main Supply & Demand analysis page"""
    st.title("📊 Supply & Demand Analysis")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading Supply & Demand data..."):
        data = load_supply_demand_data()
    
    if data.empty:
        st.warning("No Supply & Demand data available. Please ensure the scanners are running and data is being logged.")
        return
    
    # Display summary metrics
    metrics = calculate_performance_metrics(data)
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Zones (7d)", metrics['total_zones'])
        
        with col2:
            st.metric("Active Zones", metrics['active_zones'])
        
        with col3:
            st.metric("Avg Quality Score", f"{metrics['avg_quality']:.1f}")
        
        with col4:
            st.metric("Volume Confirmation", f"{metrics['volume_confirmation_rate']:.1f}%")
    
    st.markdown("---")
    
    # Data overview
    st.subheader("📈 Data Overview")
    
    # Recent zones table
    st.write("**Recent Zones (Last 24 hours):**")
    # Convert UTC to UAE time for display only
    uae_offset = timedelta(hours=4)
    
    # Fix timezone comparison by making both sides timezone-aware
    from datetime import timezone
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=1)
    recent_data = data[data['detected_at'] > cutoff_time]
    
    if not recent_data.empty:
        # Create display data with UAE time conversion and actionable trading data
        display_data = recent_data[['symbol', 'zone_type', 'quality_score', 'zone_strength', 
                                   'price_position', 'formation_type', 'current_price', 'zone_top', 'zone_bottom', 'detected_at']].copy()
        
        # Fix timezone display - convert to UAE timezone properly
        import pytz
        uae_tz = pytz.timezone('Asia/Dubai')
        display_data['detected_at'] = display_data['detected_at'].dt.tz_convert(uae_tz).dt.strftime('%H:%M')
        
        # Add actionable trading data
        display_data['Entry'] = display_data['zone_top'].round(6)  # Use zone top as entry for DEMAND, zone bottom for SUPPLY
        display_data['Stop Loss'] = display_data['zone_bottom'].round(6)  # Use zone bottom as stop for DEMAND, zone top for SUPPLY
        
        # Calculate targets based on zone strength (simplified)
        display_data['TP1'] = (display_data['current_price'] * (1 + display_data['zone_strength'] * 0.01)).round(6)
        display_data['TP2'] = (display_data['current_price'] * (1 + display_data['zone_strength'] * 0.02)).round(6)
        
        # Add trade action recommendation
        display_data['Action'] = display_data['quality_score'].apply(lambda x: '🎯 ENTER NOW' if x >= 80 else '⏳ WAIT' if x >= 60 else '❌ AVOID')
        
        # Add trade direction
        display_data['Direction'] = display_data['zone_type'].map({'DEMAND': '🟢 LONG', 'SUPPLY': '🔴 SHORT'})
        
        # Reorder columns for better display
        display_data = display_data[['symbol', 'Direction', 'zone_type', 'quality_score', 'zone_strength', 
                                    'Entry', 'Stop Loss', 'TP1', 'TP2', 'Action', 'detected_at']]
        
        # Rename columns for display
        display_data.columns = ['Symbol', 'Direction', 'Zone Type', 'Quality', 'Strength', 'Entry', 'Stop Loss', 'TP1', 'TP2', 'Action', 'Time']
        
        st.dataframe(display_data.head(20), use_container_width=True)
    else:
        st.info("No zones detected in the last 24 hours.")
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Zone Analysis Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Zone type distribution
        if not data.empty and 'zone_type' in data.columns:
            zone_type_fig = px.pie(
                data, 
                names='zone_type', 
                title="Zone Type Distribution",
                color_discrete_map={'SUPPLY': '#ff6b6b', 'DEMAND': '#4ecdc4'}
            )
            st.plotly_chart(zone_type_fig, use_container_width=True)
    
    with col2:
        # Quality score distribution
        if not data.empty and 'quality_score' in data.columns:
            quality_fig = px.histogram(
                data, 
                x='quality_score', 
                title="Quality Score Distribution",
                nbins=20,
                color='zone_type',
                color_discrete_map={'SUPPLY': '#ff6b6b', 'DEMAND': '#4ecdc4'}
            )
            st.plotly_chart(quality_fig, use_container_width=True)
    
    # Formation type analysis
    st.subheader("🏗️ Formation Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not data.empty and 'formation_type' in data.columns:
            formation_counts = data['formation_type'].value_counts().reset_index()
            formation_counts.columns = ['formation_type', 'count']
            formation_fig = px.bar(
                formation_counts,
                x='formation_type',
                y='count',
                title="Formation Type Distribution",
                labels={'formation_type': 'Formation Type', 'count': 'Count'}
            )
            st.plotly_chart(formation_fig, use_container_width=True)
    
    with col2:
        if not data.empty and 'price_position' in data.columns:
            position_fig = px.pie(
                data, 
                names='price_position', 
                title="Price Position vs Zones",
                color_discrete_map={'ABOVE': '#ff6b6b', 'BELOW': '#4ecdc4', 'INSIDE': '#45b7d1'}
            )
            st.plotly_chart(position_fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("⏰ Time Series Analysis")
    
    if not data.empty and 'detected_at' in data.columns:
        # Daily zone detection - fix timezone handling
        try:
            # Convert to UAE timezone for proper date grouping
            uae_tz = pytz.timezone('Asia/Dubai')
            data_copy = data.copy()
            data_copy['detected_at'] = data_copy['detected_at'].dt.tz_convert(uae_tz)
            
            # Group by date in UAE timezone
            daily_zones = data_copy.groupby(data_copy['detected_at'].dt.date).size().reset_index()
            daily_zones.columns = ['date', 'zone_count']
            
            if not daily_zones.empty:
                time_fig = px.line(
                    daily_zones,
                    x='date',
                    y='zone_count',
                    title="Daily Zone Detection",
                    labels={'date': 'Date', 'zone_count': 'Zones Detected'}
                )
                st.plotly_chart(time_fig, use_container_width=True)
            else:
                st.info("No time series data available")
        except Exception as e:
            st.error(f"Error creating time series: {e}")
            st.info("Time series chart unavailable")
    
    # Symbol performance
    st.subheader("🏆 Top Performing Symbols")
    
    if not data.empty and 'symbol' in data.columns:
        symbol_performance = data.groupby('symbol').agg({
            'quality_score': 'mean',
            'zone_strength': 'mean',
            'zone_type': 'count'
        }).reset_index()
        symbol_performance.columns = ['Symbol', 'Avg Quality', 'Avg Strength', 'Zone Count']
        symbol_performance = symbol_performance.sort_values('Avg Quality', ascending=False)
        
        st.dataframe(symbol_performance.head(15), use_container_width=True)
    
    # Advanced filters
    st.markdown("---")
    st.subheader("🔍 Advanced Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_quality = st.slider("Minimum Quality Score", 0, 100, 50)
    
    with col2:
        zone_types = st.multiselect(
            "Zone Types",
            options=['SUPPLY', 'DEMAND'],
            default=['SUPPLY', 'DEMAND']
        )
    
    with col3:
        price_positions = st.multiselect(
            "Price Positions",
            options=['ABOVE', 'BELOW', 'INSIDE'],
            default=['ABOVE', 'BELOW', 'INSIDE']
        )
    
    # Apply filters
    filtered_data = data[
        (data['quality_score'] >= min_quality) &
        (data['zone_type'].isin(zone_types)) &
        (data['price_position'].isin(price_positions))
    ]
    
    st.write(f"**Filtered Results: {len(filtered_data)} zones**")
    
    if not filtered_data.empty:
        # Define display columns for filtered data
        filtered_display_cols = ['symbol', 'zone_type', 'quality_score', 'zone_strength', 'price_position', 'formation_type', 'current_price', 'detected_at']
        available_cols = [col for col in filtered_display_cols if col in filtered_data.columns]
        
        if available_cols:
            st.dataframe(filtered_data[available_cols], use_container_width=True)
        else:
            st.dataframe(filtered_data.head(20), use_container_width=True)
    
    # Export functionality
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    if st.button("Export Filtered Data to CSV"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"supply_demand_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    show_supply_demand_analysis()
