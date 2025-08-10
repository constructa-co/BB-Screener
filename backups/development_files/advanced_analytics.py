"""
Advanced Analytics Module for Trading Dashboard
Add this as a new file: advanced_analytics.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from datetime import datetime, timedelta

def create_correlation_heatmap(df):
    """
    Create correlation heatmap of indicators
    """
    st.subheader("🔥 Indicator Correlation Analysis")
    
    # Select numerical columns
    numerical_cols = ['rsi', 'mfi', 'stochastic_k', 'bb_score', 
                      'volume_surge', 'probability', 'risk_reward_ratio',
                      'profit_loss_percent']
    
    # Filter available columns
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(available_cols) > 2:
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Indicator", y="Indicator", color="Correlation"),
            x=available_cols,
            y=available_cols,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="Indicator Correlation Matrix"
        )
        
        # Add correlation values
        fig.update_traces(text=corr_matrix.round(2), texttemplate='%{text}')
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.info("""
        **📊 How to Read This:**
        - **Dark Blue**: Strong positive correlation (indicators move together)
        - **Dark Red**: Strong negative correlation (indicators move opposite)
        - **White**: No correlation (indicators are independent)
        
        **💡 Trading Insights:**
        - Look for indicators that DON'T correlate (white) for confluence
        - Avoid using highly correlated indicators together
        - Negative correlations can confirm reversal signals
        """)

def predict_trade_success(trade_data, historical_df):
    """
    ML prediction for trade success probability
    """
    st.subheader("🤖 AI Trade Success Prediction")
    
    # Prepare features
    feature_cols = ['rsi', 'mfi', 'stochastic_k', 'bb_score', 
                    'volume_surge', 'risk_reward_ratio', 'probability']
    
    # Filter available features
    available_features = [col for col in feature_cols if col in historical_df.columns]
    
    if len(available_features) >= 3 and len(historical_df) >= 100:
        # Prepare training data
        X = historical_df[available_features].fillna(0)
        y = (historical_df['profit_loss_percent'] > 0).astype(int)  # 1 for profit, 0 for loss
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        with st.spinner("Training AI model..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Model accuracy
            accuracy = accuracy_score(y_test, model.predict(X_test))
            st.metric("Model Accuracy", f"{accuracy:.1%}")
        
        # Predict current trade
        if trade_data:
            trade_features = pd.DataFrame([trade_data])[available_features].fillna(0)
            
            # Get prediction and probability
            prediction = model.predict(trade_features)[0]
            probability = model.predict_proba(trade_features)[0][1] * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "AI Prediction",
                    "WIN" if prediction == 1 else "LOSS",
                    f"{probability:.1f}% confidence"
                )
            
            with col2:
                # Feature importance
                importance_df = pd.DataFrame({
                    'Feature': available_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Similar historical trades
                st.write("**Similar Historical Trades:**")
                similar = find_similar_trades(trade_data, historical_df, available_features)
                if not similar.empty:
                    win_rate = (similar['profit_loss_percent'] > 0).mean() * 100
                    avg_pnl = similar['profit_loss_percent'].mean()
                    
                    st.metric("Historical Win Rate", f"{win_rate:.1f}%")
                    st.metric("Avg P&L", f"{avg_pnl:.1f}%")

def create_pattern_success_analysis(df):
    """
    Analyze success rates by pattern type
    """
    st.subheader("📊 Pattern Success Analysis")
    
    if 'pattern_type' in df.columns and 'profit_loss_percent' in df.columns:
        # Group by pattern
        pattern_stats = df.groupby('pattern_type').agg({
            'profit_loss_percent': ['count', 'mean', lambda x: (x > 0).mean() * 100]
        }).round(2)
        
        pattern_stats.columns = ['Total Trades', 'Avg P&L %', 'Win Rate %']
        pattern_stats = pattern_stats.sort_values('Win Rate %', ascending=False)
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Win Rate by Pattern', 'Average P&L by Pattern')
        )
        
        # Win rate bar chart
        fig.add_trace(
            go.Bar(
                x=pattern_stats.index,
                y=pattern_stats['Win Rate %'],
                name='Win Rate %',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Avg P&L bar chart
        colors = ['green' if x > 0 else 'red' for x in pattern_stats['Avg P&L %']]
        fig.add_trace(
            go.Bar(
                x=pattern_stats.index,
                y=pattern_stats['Avg P&L %'],
                name='Avg P&L %',
                marker_color=colors
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern details table
        st.dataframe(
            pattern_stats.style.background_gradient(subset=['Win Rate %'], cmap='RdYlGn'),
            use_container_width=True
        )

def create_timeframe_analysis(df):
    """
    Analyze performance by timeframe
    """
    st.subheader("⏰ Timeframe Performance Analysis")
    
    if 'timeframe' in df.columns:
        # Group by timeframe
        tf_stats = df.groupby('timeframe').agg({
            'profit_loss_percent': ['count', 'mean', lambda x: (x > 0).mean() * 100],
            'risk_reward_ratio': 'mean'
        }).round(2)
        
        tf_stats.columns = ['Total Trades', 'Avg P&L %', 'Win Rate %', 'Avg R:R']
        
        # Create radar chart
        categories = tf_stats.index.tolist()
        
        fig = go.Figure()
        
        # Normalize values for radar chart
        normalized_stats = tf_stats.copy()
        for col in normalized_stats.columns:
            max_val = normalized_stats[col].max()
            if max_val > 0:
                normalized_stats[col] = normalized_stats[col] / max_val * 100
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_stats['Win Rate %'],
            theta=categories,
            fill='toself',
            name='Win Rate %'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_stats['Avg R:R'] * 20,  # Scale for visibility
            theta=categories,
            fill='toself',
            name='Risk/Reward'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Timeframe Performance Radar"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed stats
        st.dataframe(
            tf_stats.style.background_gradient(subset=['Win Rate %'], cmap='RdYlGn'),
            use_container_width=True
        )

def create_ml_backtesting_simulator(df):
    """
    ML-based backtesting with different strategies
    """
    st.subheader("🧪 ML Strategy Backtesting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_prob = st.slider("Min Probability %", 50, 95, 70)
    with col2:
        min_rr = st.slider("Min Risk/Reward", 1.0, 5.0, 2.0)
    with col3:
        max_risk = st.slider("Max Risk %", 1, 5, 2)
    
    # Filter trades based on criteria
    filtered_df = df[
        (df['probability'] >= min_prob) &
        (df['risk_reward_ratio'] >= min_rr)
    ]
    
    if not filtered_df.empty:
        # Calculate performance
        total_trades = len(filtered_df)
        winning_trades = len(filtered_df[filtered_df['profit_loss_percent'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = filtered_df[filtered_df['profit_loss_percent'] > 0]['profit_loss_percent'].mean()
        avg_loss = filtered_df[filtered_df['profit_loss_percent'] < 0]['profit_loss_percent'].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        with col4:
            expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
            st.metric("Expectancy", f"{expectancy:.2f}%")
        
        # Equity curve
        filtered_df = filtered_df.sort_values('timestamp')
        filtered_df['cumulative_pnl'] = filtered_df['profit_loss_percent'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df['cumulative_pnl'],
            mode='lines',
            name='Equity Curve',
            line=dict(color='green', width=2)
        ))
        
        # Add drawdown shading
        running_max = filtered_df['cumulative_pnl'].expanding().max()
        drawdown = filtered_df['cumulative_pnl'] - running_max
        
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        fig.update_layout(
            title="Strategy Equity Curve",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L %",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def find_similar_trades(current_trade, historical_df, features):
    """
    Find similar historical trades using ML
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Prepare current trade features
    current_features = pd.DataFrame([current_trade])[features].fillna(0)
    
    # Calculate similarity
    historical_features = historical_df[features].fillna(0)
    similarities = cosine_similarity(current_features, historical_features)[0]
    
    # Get top 10 similar trades
    similar_indices = similarities.argsort()[-10:][::-1]
    similar_trades = historical_df.iloc[similar_indices]
    
    return similar_trades

def get_historical_trades():
    """
    Get historical trades from database
    """
    from trade_logger import TradeLogger
    
    logger = TradeLogger()
    trades = []
    
    if logger.connection:
        logger.cursor.execute("""
            SELECT 
                t.*,
                s.scan_type,
                s.scan_timestamp as timestamp,
                COALESCE(t.profit_loss_percent, 0) as profit_loss_percent,
                COALESCE(t.rsi, 0) as rsi,
                COALESCE(t.mfi, 0) as mfi,
                COALESCE(t.stochastic_k, 0) as stochastic_k,
                COALESCE(t.bb_score, 0) as bb_score,
                COALESCE(t.volume_surge, 0) as volume_surge,
                COALESCE(t.risk_reward_ratio, 0) as risk_reward_ratio,
                COALESCE(t.probability, 0) as probability,
                'BB Bounce' as pattern_type,
                '4H' as timeframe
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.trade_taken = TRUE
            AND t.timestamp > NOW() - INTERVAL '30 days'
            ORDER BY t.timestamp DESC
        """)
        
        trades = logger.cursor.fetchall()
    
    return pd.DataFrame(trades) if trades else pd.DataFrame() 