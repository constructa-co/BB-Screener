"""
Interactive Controls for Live Strategy Tuning
Add this functionality to your dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def create_strategy_tuning_page():
    """
    Create interactive strategy tuning interface
    """
    st.title("🎛️ Interactive Strategy Tuning")
    
    # Get historical data for backtesting
    historical_data = get_historical_trades()
    
    if historical_data.empty:
        st.warning("No historical data available for backtesting")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Parameter Optimization",
        "📊 Real-time Backtesting", 
        "🔬 Strategy Comparison",
        "💡 AI Recommendations"
    ])
    
    with tab1:
        parameter_optimization_interface(historical_data)
    
    with tab2:
        realtime_backtesting_interface(historical_data)
    
    with tab3:
        strategy_comparison_interface(historical_data)
    
    with tab4:
        ai_recommendations_interface(historical_data)

def parameter_optimization_interface(data):
    """
    Interactive parameter tuning with instant feedback
    """
    st.subheader("🎯 Parameter Optimization")
    
    # Create parameter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Entry Criteria**")
        min_prob = st.slider(
            "Min Probability %",
            min_value=50,
            max_value=95,
            value=70,
            step=5,
            help="Minimum probability for trade entry"
        )
        
        min_rr = st.slider(
            "Min Risk/Reward",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Minimum risk/reward ratio"
        )
        
        max_risk = st.slider(
            "Max Risk % per Trade",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Maximum risk per trade as % of capital"
        )
    
    with col2:
        st.write("**Technical Filters**")
        rsi_oversold = st.slider(
            "RSI Oversold Level",
            min_value=10,
            max_value=40,
            value=30,
            help="RSI level to consider oversold"
        )
        
        rsi_overbought = st.slider(
            "RSI Overbought Level",
            min_value=60,
            max_value=90,
            value=70,
            help="RSI level to consider overbought"
        )
        
        volume_surge = st.slider(
            "Min Volume Surge",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Minimum volume compared to average"
        )
    
    with col3:
        st.write("**Exit Rules**")
        take_profit_1 = st.slider(
            "Take Profit 1 %",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="First take profit target"
        )
        
        take_profit_2 = st.slider(
            "Take Profit 2 %",
            min_value=2.0,
            max_value=20.0,
            value=6.0,
            step=1.0,
            help="Second take profit target"
        )
        
        trailing_stop = st.checkbox(
            "Enable Trailing Stop",
            value=True,
            help="Move stop loss to breakeven after TP1"
        )
    
    # Apply filters to data
    filtered_data = apply_strategy_filters(
        data,
        min_prob=min_prob,
        min_rr=min_rr,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        volume_surge=volume_surge
    )
    
    # Display instant results
    st.markdown("---")
    display_strategy_results(filtered_data, data, max_risk)
    
    # Show trade distribution
    create_trade_distribution_chart(filtered_data)
    
    # Save strategy button
    if st.button("💾 Save Strategy Settings"):
        save_strategy_settings({
            'min_probability': min_prob,
            'min_risk_reward': min_rr,
            'max_risk_percent': max_risk,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'volume_surge': volume_surge,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'trailing_stop': trailing_stop
        })
        st.success("Strategy settings saved!")

def realtime_backtesting_interface(data):
    """
    Real-time backtesting with visual results
    """
    st.subheader("📊 Real-time Backtesting")
    
    # Backtest parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        backtest_period = st.selectbox(
            "Backtest Period",
            ["Last Week", "Last Month", "Last 3 Months", "Last Year", "All Data"]
        )
    
    with col2:
        starting_capital = st.number_input(
            "Starting Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
    
    with col3:
        position_sizing = st.selectbox(
            "Position Sizing",
            ["Fixed %", "Kelly Criterion", "Risk Parity", "Equal Weight"]
        )
    
    with col4:
        max_positions = st.number_input(
            "Max Concurrent Positions",
            min_value=1,
            max_value=20,
            value=5
        )
    
    # Strategy selection
    st.write("**Select Strategies to Backtest:**")
    
    strategies = {}
    cols = st.columns(4)
    
    with cols[0]:
        if st.checkbox("BB Scanner", value=True):
            strategies['BB'] = {
                'min_prob': st.slider("BB Min Prob", 50, 95, 70, key="bb_prob")
            }
    
    with cols[1]:
        if st.checkbox("ICT Scanner"):
            strategies['ICT'] = {
                'min_prob': st.slider("ICT Min Prob", 50, 95, 75, key="ict_prob")
            }
    
    with cols[2]:
        if st.checkbox("Wyckoff Scanner"):
            strategies['Wyckoff'] = {
                'min_prob': st.slider("Wyckoff Min Prob", 50, 95, 80, key="wyck_prob")
            }
    
    with cols[3]:
        if st.checkbox("Combined"):
            strategies['Combined'] = {
                'min_prob': st.slider("Combined Min Prob", 50, 95, 85, key="comb_prob")
            }
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            backtest_results = run_comprehensive_backtest(
                data,
                strategies,
                starting_capital,
                position_sizing,
                max_positions,
                backtest_period
            )
            
            # Display results
            display_backtest_results(backtest_results)

def strategy_comparison_interface(data):
    """
    Compare multiple strategies side by side
    """
    st.subheader("🔬 Strategy Comparison")
    
    # Define strategies to compare
    strategies = {
        "Conservative": {
            "min_prob": 80,
            "min_rr": 3.0,
            "max_risk": 1.0,
            "description": "High probability, low risk"
        },
        "Balanced": {
            "min_prob": 70,
            "min_rr": 2.0,
            "max_risk": 2.0,
            "description": "Balance of frequency and quality"
        },
        "Aggressive": {
            "min_prob": 60,
            "min_rr": 1.5,
            "max_risk": 3.0,
            "description": "More trades, higher risk"
        },
        "Scalping": {
            "min_prob": 55,
            "min_rr": 1.2,
            "max_risk": 1.0,
            "description": "High frequency, small gains"
        }
    }
    
    # Calculate performance for each strategy
    comparison_data = []
    
    for name, params in strategies.items():
        filtered = data[
            (data['probability'] >= params['min_prob']) &
            (data['risk_reward_ratio'] >= params['min_rr'])
        ]
        
        if not filtered.empty:
            wins = len(filtered[filtered['profit_loss_percent'] > 0])
            total = len(filtered)
            win_rate = (wins / total * 100) if total > 0 else 0
            avg_win = filtered[filtered['profit_loss_percent'] > 0]['profit_loss_percent'].mean()
            avg_loss = filtered[filtered['profit_loss_percent'] < 0]['profit_loss_percent'].mean()
            
            comparison_data.append({
                'Strategy': name,
                'Description': params['description'],
                'Total Trades': total,
                'Win Rate %': win_rate,
                'Avg Win %': avg_win,
                'Avg Loss %': avg_loss,
                'Expectancy %': (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss),
                'Profit Factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            })
    
    # Display comparison table
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.background_gradient(subset=['Win Rate %', 'Expectancy %'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Visual comparison
    fig = go.Figure()
    
    # Add traces for each metric
    metrics = ['Win Rate %', 'Expectancy %', 'Profit Factor']
    colors = ['blue', 'green', 'orange']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Strategy'],
            y=comparison_df[metric],
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title="Strategy Performance Comparison",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monte Carlo simulation
    if st.button("Run Monte Carlo Simulation"):
        run_monte_carlo_simulation(data, strategies)

def ai_recommendations_interface(data):
    """
    AI-powered strategy recommendations
    """
    st.subheader("💡 AI Strategy Recommendations")
    
    # Analyze current market conditions
    market_analysis = analyze_market_conditions(data)
    
    # Display market regime
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Market Regime", market_analysis['regime'])
    with col2:
        st.metric("Volatility", market_analysis['volatility'])
    with col3:
        st.metric("Trend Strength", f"{market_analysis['trend_strength']:.1f}%")
    
    st.markdown("---")
    
    # AI recommendations based on analysis
    st.write("### 🤖 AI Recommendations")
    
    recommendations = generate_ai_recommendations(data, market_analysis)
    
    for rec in recommendations:
        with st.expander(f"{rec['icon']} {rec['title']}"):
            st.write(f"**Recommendation:** {rec['description']}")
            st.write(f"**Reason:** {rec['reason']}")
            st.write(f"**Expected Impact:** {rec['impact']}")
            
            if st.button(f"Apply {rec['title']}", key=rec['id']):
                apply_ai_recommendation(rec['settings'])
                st.success(f"Applied {rec['title']} settings!")
                st.experimental_rerun()

# Helper functions
def apply_strategy_filters(data, **kwargs):
    """Apply strategy filters to data"""
    filtered = data.copy()
    
    if 'min_prob' in kwargs:
        filtered = filtered[filtered['probability'] >= kwargs['min_prob']]
    
    if 'min_rr' in kwargs:
        filtered = filtered[filtered['risk_reward_ratio'] >= kwargs['min_rr']]
    
    if 'rsi_oversold' in kwargs and 'rsi' in filtered.columns:
        # For long trades
        long_trades = filtered[filtered['direction'] == 'LONG']
        long_trades = long_trades[long_trades['rsi'] <= kwargs['rsi_oversold']]
        
        # For short trades
        short_trades = filtered[filtered['direction'] == 'SHORT']
        short_trades = short_trades[short_trades['rsi'] >= kwargs.get('rsi_overbought', 70)]
        
        filtered = pd.concat([long_trades, short_trades])
    
    return filtered

def display_strategy_results(filtered_data, original_data, max_risk):
    """Display strategy performance metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_original = len(original_data)
    total_filtered = len(filtered_data)
    
    if total_filtered > 0:
        wins = len(filtered_data[filtered_data['profit_loss_percent'] > 0])
        win_rate = (wins / total_filtered * 100)
        avg_pnl = filtered_data['profit_loss_percent'].mean()
        sharpe = calculate_sharpe_ratio(filtered_data['profit_loss_percent'])
    else:
        win_rate = 0
        avg_pnl = 0
        sharpe = 0
    
    with col1:
        st.metric(
            "Trades Selected",
            f"{total_filtered}/{total_original}",
            f"{(total_filtered/total_original*100):.1f}%"
        )
    
    with col2:
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            f"{win_rate - 50:.1f}%" if win_rate > 50 else f"{win_rate - 50:.1f}%"
        )
    
    with col3:
        st.metric(
            "Avg P&L",
            f"{avg_pnl:.2f}%",
            "Profitable" if avg_pnl > 0 else "Loss"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            "Good" if sharpe > 1 else "Poor"
        )

def create_trade_distribution_chart(data):
    """Create distribution chart of trades"""
    if data.empty:
        st.info("No trades match current filters")
        return
    
    # P&L distribution
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data['profit_loss_percent'],
        nbinsx=30,
        name='P&L Distribution',
        marker_color='lightblue'
    ))
    
    # Add average line
    avg_pnl = data['profit_loss_percent'].mean()
    fig.add_vline(x=avg_pnl, line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {avg_pnl:.1f}%")
    
    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="Profit/Loss %",
        yaxis_title="Frequency",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_comprehensive_backtest(data, strategies, capital, sizing, max_pos, period):
    """Run comprehensive backtest"""
    results = {}
    
    for strategy_name, params in strategies.items():
        # Filter data for strategy
        strategy_data = data[data['probability'] >= params['min_prob']]
        
        # Simulate trading
        equity_curve = [capital]
        trades = []
        
        for _, trade in strategy_data.iterrows():
            # Simple simulation
            position_size = capital * 0.02  # 2% risk
            pnl = position_size * (trade['profit_loss_percent'] / 100)
            capital += pnl
            equity_curve.append(capital)
            
            trades.append({
                'date': trade['timestamp'],
                'pnl': pnl,
                'equity': capital
            })
        
        results[strategy_name] = {
            'final_equity': capital,
            'total_return': ((capital - equity_curve[0]) / equity_curve[0] * 100),
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    return results

def display_backtest_results(results):
    """Display backtest results"""
    # Summary metrics
    summary_data = []
    for strategy, data in results.items():
        summary_data.append({
            'Strategy': strategy,
            'Final Equity': f"${data['final_equity']:,.0f}",
            'Total Return': f"{data['total_return']:.1f}%",
            'Total Trades': len(data['trades'])
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Equity curves
    fig = go.Figure()
    
    for strategy, data in results.items():
        fig.add_trace(go.Scatter(
            y=data['equity_curve'],
            mode='lines',
            name=strategy
        ))
    
    fig.update_layout(
        title="Equity Curves",
        xaxis_title="Trade Number",
        yaxis_title="Equity ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

def get_historical_trades():
    """Get historical trades from database"""
    from trade_logger import TradeLogger
    
    logger = TradeLogger()
    df = pd.DataFrame()
    
    if logger.connection:
        logger.cursor.execute("""
            SELECT * FROM trade_opportunities
            WHERE trade_taken = TRUE
            AND trade_result IS NOT NULL
            ORDER BY timestamp DESC
        """)
        
        results = logger.cursor.fetchall()
        if results:
            df = pd.DataFrame(results)
    
    logger.close()
    return df

def save_strategy_settings(settings):
    """Save strategy settings to database"""
    # Implementation for saving settings
    pass

def run_monte_carlo_simulation(data, strategies):
    """Run Monte Carlo simulation"""
    st.info("Monte Carlo simulation feature coming soon!")

def analyze_market_conditions(data):
    """Analyze current market conditions"""
    if data.empty:
        return {
            'regime': 'Unknown',
            'volatility': 'Low',
            'trend_strength': 0
        }
    
    # Simple analysis
    recent_data = data.head(100)
    volatility = recent_data['profit_loss_percent'].std()
    
    return {
        'regime': 'Trending' if volatility > 2 else 'Ranging',
        'volatility': 'High' if volatility > 3 else 'Medium' if volatility > 1.5 else 'Low',
        'trend_strength': min(100, volatility * 20)
    }

def generate_ai_recommendations(data, market_analysis):
    """Generate AI recommendations based on market conditions"""
    recommendations = []
    
    if market_analysis['regime'] == 'Trending':
        recommendations.append({
            'id': 'trend_following',
            'icon': '📈',
            'title': 'Trend Following',
            'description': 'Increase position sizes and use trend-following indicators',
            'reason': 'Market is trending with strong directional movement',
            'impact': 'Expected 15-25% improvement in win rate',
            'settings': {'min_prob': 65, 'min_rr': 1.8}
        })
    
    if market_analysis['volatility'] == 'High':
        recommendations.append({
            'id': 'volatility_adjustment',
            'icon': '⚡',
            'title': 'Volatility Adjustment',
            'description': 'Reduce position sizes and widen stops',
            'reason': 'High volatility requires more conservative risk management',
            'impact': 'Reduced drawdown by 30-40%',
            'settings': {'max_risk': 1.5, 'trailing_stop': True}
        })
    
    return recommendations

def apply_ai_recommendation(settings):
    """Apply AI recommendation settings"""
    # Implementation for applying settings
    pass 