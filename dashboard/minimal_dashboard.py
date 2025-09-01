import streamlit as st
import pandas as pd
from datetime import datetime
import pytz

# Page configuration
st.set_page_config(
    page_title="BB Screener - Minimal Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Display time header
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
        st.warning(f"⚠️ Time display error: {e}")
        st.info("All times shown in UTC format")

# Main dashboard
def main():
    st.markdown('<h1 class="main-header">🚀 BB Screener - Minimal Working Dashboard</h1>', unsafe_allow_html=True)
    display_time_header()
    
    st.warning("⚠️ This is a minimal dashboard to restore basic functionality while we resolve the underlying database timestamp issues.")
    
    # Overview section
    st.header("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dashboard", "🟢 Running")
        st.caption("Minimal version active")
    
    with col2:
        st.metric("Database", "🟡 Partial")
        st.caption("Timestamp issues identified")
    
    with col3:
        st.metric("Scanners", "🟡 Partial")
        st.caption("Some working, some with issues")
    
    # Scanner status
    st.header("🎯 Scanner Status")
    
    scanners = [
        ("FVG Analysis", "🟡 Partial - Timestamp issues"),
        ("Supply & Demand", "🟡 Partial - Timestamp issues"),
        ("Trend Following", "🟢 Working"),
        ("Flagpole", "🟢 Working"),
        ("ICT", "🟢 Working"),
        ("Wyckoff", "🟢 Working"),
        ("Elliott Wave", "🟡 Partial - Timestamp issues"),
        ("Fibonacci", "🔴 Disabled - Causing crashes")
    ]
    
    for name, status in scanners:
        st.info(f"{name}: {status}")
    
    # Current issue explanation
    st.header("🔍 Current Issue Analysis")
    
    st.markdown("""
    **Root Cause Identified:**
    - Database has mixed timestamp column types
    - Some tables use `timestamp with time zone`
    - Others use `timestamp without time zone`
    - This causes pandas to crash when processing data
    
    **What We've Fixed:**
    - ✅ Removed all .dt accessors from Python code
    - ✅ Standardized main table timestamps
    - ✅ Fixed problematic database views
    - ✅ Removed Fibonacci components causing crashes
    
    **What Still Needs Fixing:**
    - 16+ remaining timestamp columns in database
    - Systemic database schema inconsistencies
    - Need complete database audit or rebuild
    """)
    
    # Next steps
    st.header("🚀 Next Steps")
    
    st.markdown("""
    **Immediate Actions:**
    1. **Verify this minimal dashboard works** ✅ (You're here!)
    2. **Test basic functionality** - confirm no more crashes
    3. **Plan database schema standardization**
    
    **Long-term Solution Options:**
    - **Option A:** Fix all remaining database timestamp columns systematically
    - **Option B:** Rebuild database with consistent timestamp schema
    - **Option C:** Create new database views that handle timestamp conversion
    
    **Current Status:** Ready for systematic database fix once we confirm this dashboard is stable.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Minimal BB Screener Dashboard - Temporary solution while resolving database issues*")

if __name__ == "__main__":
    main()
