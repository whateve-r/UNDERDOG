"""
UNDERDOG Streamlit Dashboard - Real Backtesting Integration
=============================================================

Functional dashboard using bt_engine.py with real backtest results.

USAGE:
------
    poetry run streamlit run scripts/streamlit_dashboard.py

Features:
---------
✅ Real backtesting with Backtrader
✅ PropFirmRiskManager integration
✅ Monte Carlo validation
✅ Multiple strategy support
✅ Live equity curve
✅ Trade history table
✅ Performance metrics

Author: UNDERDOG Development Team
Date: October 2025
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from underdog.backtesting.bt_engine import run_backtest, run_parameter_sweep

# Page configuration
st.set_page_config(
    page_title="UNDERDOG Backtesting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================  SIDEBAR ====================

st.sidebar.title("⚙️ Configuration")

# EA Selection
strategy_name = st.sidebar.selectbox(
    "Select Strategy",
    ["ATRBreakout", "SuperTrendRSI", "BollingerCCI"],
    help="Select which strategy to backtest"
)

# Data Source
st.sidebar.subheader("📊 Data Source")
use_real_data = st.sidebar.checkbox(
    "Use HuggingFace Data",
    value=False,
    help="Check to use real historical data from HuggingFace. Requires authentication (run scripts/setup_hf_token.py)"
)

if use_real_data:
    st.sidebar.info("🌐 Using real HuggingFace data (elthariel/histdata_fx_1m)")
else:
    st.sidebar.warning("⚠️ Using synthetic data for testing")

# Symbol & Date Range
symbol = st.sidebar.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY"])

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start", value=datetime(2023, 1, 1))
with col2:
    end_date = st.date_input("End", value=datetime(2024, 12, 31))

# Strategy Parameters
st.sidebar.subheader(f"📐 {strategy_name} Parameters")

params = {}

if strategy_name == "ATRBreakout":
    params['atr_period'] = st.sidebar.slider("ATR Period", 10, 30, 14)
    params['atr_multiplier_entry'] = st.sidebar.slider("Entry Multiplier", 1.0, 3.0, 1.5, 0.1)
    params['atr_multiplier_sl'] = st.sidebar.slider("SL Multiplier", 1.0, 3.0, 1.5, 0.1)
    params['rsi_period'] = st.sidebar.slider("RSI Period", 10, 20, 14)
elif strategy_name == "SuperTrendRSI":
    params['atr_period'] = st.sidebar.slider("ATR Period", 10, 30, 14)
    params['atr_multiplier'] = st.sidebar.slider("ATR Multiplier", 1.0, 4.0, 2.0, 0.5)
    params['rsi_period'] = st.sidebar.slider("RSI Period", 10, 20, 14)
    params['rsi_overbought'] = st.sidebar.slider("RSI Overbought", 60, 80, 65)
    params['rsi_oversold'] = st.sidebar.slider("RSI Oversold", 20, 40, 35)
elif strategy_name == "BollingerCCI":
    params['bb_period'] = st.sidebar.slider("BB Period", 10, 50, 20)
    params['bb_stddev'] = st.sidebar.slider("BB StdDev", 1.0, 3.0, 2.0, 0.5)
    params['cci_period'] = st.sidebar.slider("CCI Period", 10, 30, 20)

# Risk Management
st.sidebar.subheader("💰 Risk Management")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 1000000, 10000, 1000)
risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
commission = st.sidebar.slider("Commission (%)", 0.0, 0.1, 0.01, 0.01) / 100

# Validation
st.sidebar.subheader("✅ Validation")
enable_mc = st.sidebar.checkbox("Monte Carlo Validation", value=True)
mc_iterations = st.sidebar.slider("MC Iterations", 100, 10000, 1000, 100)

# Run Button
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# ==================== MAIN AREA ====================

st.markdown('<div class="main-header">📊 UNDERDOG Backtesting Dashboard</div>', unsafe_allow_html=True)

# Status Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Strategy", strategy_name)
with col2:
    st.metric("Symbol", symbol)
with col3:
    days = (end_date - start_date).days
    st.metric("Period", f"{days} days")
with col4:
    st.metric("Capital", f"${initial_capital:,}")

st.markdown("---")

# ==================== RUN BACKTEST ====================

if run_button:
    with st.spinner(f"🚀 Running backtest: {strategy_name} on {symbol}..."):
        try:
            # Add risk management params
            params['risk_per_trade'] = risk_per_trade
            
            # Run backtest
            results = run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                strategy_params=params,
                initial_capital=initial_capital,
                commission=commission,
                validate_monte_carlo=enable_mc,
                mc_iterations=mc_iterations,
                use_hf_data=use_real_data  # ✨ NEW: Use real data if checked
            )
            
            st.session_state.backtest_results = results
            st.success(f"✅ Backtest complete! {results['metrics']['num_trades']} trades executed.")
            
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ==================== DISPLAY RESULTS ====================

if st.session_state.backtest_results is not None:
    results = st.session_state.backtest_results
    metrics = results['metrics']
    equity_df = results['equity_curve']
    trades_df = results['trades']
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Equity Curve",
        "📊 Performance Metrics",
        "📋 Trade History",
        "✅ Validation"
    ])
    
    # ==================== TAB 1: EQUITY CURVE ====================
    
    with tab1:
        st.subheader("📈 Equity Curve & Drawdown")
        
        # Create figure
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=equity_df['datetime'],
            y=equity_df['equity'],
            name='Equity',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='%{x}<br>Equity: $%{y:,.2f}<extra></extra>'
        ))
        
        # Drawdown (secondary axis)
        fig.add_trace(go.Scatter(
            x=equity_df['datetime'],
            y=equity_df['drawdown_pct'],
            name='Drawdown',
            line=dict(color='#d62728', width=1, dash='dot'),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.1)',
            yaxis='y2',
            hovertemplate='%{x}<br>DD: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            yaxis2=dict(
                title="Drawdown (%)",
                overlaying='y',
                side='right',
                range=[min(equity_df['drawdown_pct']) * 1.2, 0]
            ),
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = f"+{metrics['total_return_pct']:.2f}%" if metrics['total_return_pct'] > 0 else f"{metrics['total_return_pct']:.2f}%"
            st.metric("Final Equity", f"${metrics['final_capital']:,.2f}", delta)
        with col2:
            st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%", delta_color="inverse")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")
    
    # ==================== TAB 2: PERFORMANCE METRICS ====================
    
    with tab2:
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Returns")
            st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Avg P&L", f"${metrics['avg_pnl']:.2f}")
        
        with col2:
            st.markdown("### 📉 Risk")
            st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
            st.metric("Total Trades", f"{metrics['num_trades']}")
            st.metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")
        
        with col3:
            st.markdown("### 💵 Trade Stats")
            st.metric("Avg Win", f"+${metrics['avg_win']:.2f}")
            st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    # ==================== TAB 3: TRADE HISTORY ====================
    
    with tab3:
        st.subheader("📋 Trade History")
        
        if len(trades_df) > 0:
            # Format DataFrame
            df_display = trades_df.copy()
            df_display['entry_date'] = pd.to_datetime(df_display['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
            df_display['exit_date'] = pd.to_datetime(df_display['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
            df_display['pnl'] = df_display['pnl'].round(2)
            df_display['entry_price'] = df_display['entry_price'].round(5)
            df_display['exit_price'] = df_display['exit_price'].round(5)
            
            # Select columns
            cols = ['entry_date', 'exit_date', 'side', 'entry_price', 'exit_price', 'pnl', 'bars_held']
            df_display = df_display[cols]
            
            # Color-code P&L
            def color_pnl(val):
                color = 'green' if val > 0 else 'red'
                return f'color: {color}'
            
            st.dataframe(
                df_display.style.applymap(color_pnl, subset=['pnl']),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade History (CSV)",
                data=csv,
                file_name=f'trades_{strategy_name}_{symbol}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        else:
            st.warning("No trades executed during backtest period")
    
    # ==================== TAB 4: VALIDATION ====================
    
    with tab4:
        st.subheader("✅ Monte Carlo Validation")
        
        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            
            if mc['is_robust']:
                st.success(f"✅ **ROBUST STRATEGY** - Passed Monte Carlo validation ({mc['iterations']:,} iterations)")
                st.info("📊 Strategy performance is NOT due to lucky trade sequence")
            else:
                st.error(f"❌ **NOT ROBUST** - Failed Monte Carlo validation ({mc['iterations']:,} iterations)")
                st.warning("⚠️ Do NOT trade this strategy - results likely due to luck")
            
            st.markdown("---")
            
            st.markdown("### What is Monte Carlo Validation?")
            st.markdown("""
            Monte Carlo validation shuffles the order of trades thousands of times and recalculates P&L.
            
            **Robust Strategy**: Original P&L ranks above 5th percentile (consistent across shuffles)
            
            **Lucky Strategy**: Original P&L ranks below 5th percentile (depends on trade order)
            
            This test helps detect overfitting and ensures strategy robustness.
            """)
        else:
            st.info("Monte Carlo validation not enabled. Enable in sidebar to validate results.")

else:
    # Placeholder when no results
    st.info("👆 Configure parameters in sidebar and click 'Run Backtest' to see results")
    
    st.markdown("### Features")
    st.markdown("""
    - ✅ **Real Backtesting**: Uses Backtrader engine with realistic commission/slippage
    - ✅ **Risk Management**: PropFirmRiskManager with Kelly Criterion and DD limits
    - ✅ **Validation**: Monte Carlo shuffling to detect lucky backtests
    - ✅ **Multiple Strategies**: ATRBreakout, SuperTrendRSI, BollingerCCI
    - ✅ **Synthetic Data**: Realistic OHLCV generation for testing
    """)

# Footer
st.markdown("---")
st.caption("📊 UNDERDOG Algorithmic Trading System | Powered by Backtrader + Streamlit")
