#!/usr/bin/env python3
"""
UTC/UAE Clock Display Module
Shows current time in both UTC and UAE timezone for reference
"""

import streamlit as st
from datetime import datetime
import pytz

def display_time_header():
    """Display UTC and UAE time at top of dashboard"""
    try:
        utc_now = datetime.now(pytz.UTC)
        uae_now = utc_now.astimezone(pytz.timezone('Asia/Dubai'))
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.empty()  # Spacer
        with col2:
            st.metric("🌍 UTC Time", utc_now.strftime('%H:%M:%S'))
            st.caption(utc_now.strftime('%Y-%m-%d'))
        with col3:
            st.metric("🇦🇪 UAE Time (+4h)", uae_now.strftime('%H:%M:%S'))
            st.caption(uae_now.strftime('%Y-%m-%d'))
            
    except Exception as e:
        # Fallback if timezone fails
        st.warning(f"⚠️ Time display error: {e}")
        st.info("All times shown in UTC format")
