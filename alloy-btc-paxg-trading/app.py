import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import os
import sys

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Alloy modules
from alloy.data_manager import DataManager
from alloy.strategy import AlloyPortfolio
from alloy.backtester import BacktestEngine
from alloy.optimizer import StrategyOptimizer
from alloy.reporting import SignalGenerator, PerformanceReporter

# Set page configuration
st.set_page_config(
    page_title="Alloy BTC-PAXG Trading Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page header
st.title("Alloy BTC-PAXG Trading Assistant")

st.markdown("""
This application helps you analyze and implement the Alloy BTC-PAXG strategy,
a dynamic asset allocation approach designed to preserve value by balancing between 
Bitcoin (BTC) and Pax Gold (PAXG) based on market conditions.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Backtesting", "Trading Signals", "Strategy Optimizer", "About"])

# Initialize DataManager
data_manager = DataManager()

if page == "Backtesting":
    st.header("Strategy Backtesting")
    
    # Get date ranges for backtesting
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime(2020, 1, 1),
            min_value=datetime(2018, 1, 1),
            max_value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now(),
            min_value=start_date + timedelta(days=30),
            max_value=datetime.now()
        )
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
        momentum_window = st.slider("Momentum Window (days)", min_value=10, max_value=60, value=30, step=5)
        volatility_window = st.slider("Volatility Window (days)", min_value=20, max_value=100, value=60, step=10)
        rebalance_frequency = st.slider("Rebalance Frequency (days)", min_value=1, max_value=30, value=3, step=1)
    
    with col2:
        momentum_threshold_bull = st.slider("Momentum Threshold Bull (%)", min_value=0, max_value=20, value=10, step=1)
        momentum_threshold_bear = st.slider("Momentum Threshold Bear (%)", min_value=-20, max_value=0, value=-5, step=1)
        max_btc_allocation = st.slider("Maximum BTC Allocation (%)", min_value=50, max_value=100, value=90, step=5) / 100
        min_btc_allocation = st.slider("Minimum BTC Allocation (%)", min_value=0, max_value=50, value=20, step=5) / 100
    
    # Transaction costs
    transaction_cost = st.slider("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100
    
    # Run backtest button
    if st.button("Run Backtest"):
        with st.spinner("Loading data and running backtest..."):
            # Load historical data
            historical_data = data_manager.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if historical_data is not None:
                # Initialize strategy
                strategy = AlloyPortfolio(
                    initial_capital=initial_capital,
                    momentum_window=momentum_window,
                    volatility_window=volatility_window,
                    momentum_threshold_bull=momentum_threshold_bull,
                    momentum_threshold_bear=momentum_threshold_bear,
                    max_btc_allocation=max_btc_allocation,
                    min_btc_allocation=min_btc_allocation,
                    rebalance_frequency=rebalance_frequency,
                    transaction_cost=transaction_cost
                )
                
                # Run backtest
                backtest_engine = BacktestEngine(strategy)
                results = backtest_engine.run(historical_data)
                
                # Calculate metrics
                metrics = backtest_engine.calculate_metrics(results, historical_data)
                
                # Display results
                st.subheader("Backtest Results")
                
                # Key metrics
                metrics_cols = st.columns(4)
                metrics_cols[0].metric("Total Return", f"{metrics['rendement_total_alloy']:.2f}%")
                metrics_cols[1].metric("Annualized Return", f"{metrics['rendement_annualise_alloy']:.2f}%")
                metrics_cols[2].metric("Sharpe Ratio", f"{metrics['ratio_sharpe_alloy']:.2f}")
                metrics_cols[3].metric("Max Drawdown", f"{metrics['drawdown_maximum_alloy']:.2f}%")
                
                # Additional metrics
                st.subheader("Detailed Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("Alloy Strategy")
                    st.write(f"Total Return: {metrics['rendement_total_alloy']:.2f}%")
                    st.write(f"Annualized Return: {metrics['rendement_annualise_alloy']:.2f}%")
                    st.write(f"Annualized Volatility: {metrics['volatilite_annualisee_alloy']:.2f}%")
                    st.write(f"Sharpe Ratio: {metrics['ratio_sharpe_alloy']:.2f}")
                    st.write(f"Max Drawdown: {metrics['drawdown_maximum_alloy']:.2f}%")
                    st.write(f"Average Drawdown: {metrics['drawdown_moyen_alloy']:.2f}%")
                    st.write(f"Total Trades: {metrics['total_trades']}")
                    st.write(f"Total Fees: ${metrics['total_fees']:.2f}")
                
                with col2:
                    st.write("Buy & Hold Strategy")
                    st.write(f"Total Return: {metrics['rendement_total_buy_hold']:.2f}%")
                    st.write(f"Annualized Return: {metrics['rendement_annualise_buy_hold']:.2f}%")
                    st.write(f"Annualized Volatility: {metrics['volatilite_annualisee_buy_hold']:.2f}%")
                    st.write(f"Sharpe Ratio: {metrics['ratio_sharpe_buy_hold']:.2f}")
                    st.write(f"Max Drawdown: {metrics['drawdown_maximum_buy_hold']:.2f}%")
                
                with col3:
                    st.write("DCA Strategy")
                    st.write(f"Total Return: {metrics['rendement_total_dca']:.2f}%")
                    st.write(f"Annualized Return: {metrics['rendement_annualise_dca']:.2f}%")
                    st.write(f"Annualized Volatility: {metrics['volatilite_annualisee_dca']:.2f}%")
                    st.write(f"Sharpe Ratio: {metrics['ratio_sharpe_dca']:.2f}")
                    st.write(f"Max Drawdown: {metrics['drawdown_maximum_dca']:.2f}%")
                
                # Performance Charts
                st.subheader("Performance Charts")
                
                # Create normalized performance chart
                fig = go.Figure()
                
                # Normalized performance
                normalized_alloy = results['portfolio_value'] / results['portfolio_value'].iloc[0] * 100
                normalized_buy_hold = results['buy_hold_value'] / results['buy_hold_value'].iloc[0] * 100
                normalized_dca = results['dca_value'] / results['dca_value'].iloc[0] * 100
                normalized_btc = historical_data['BTC'] / historical_data['BTC'].iloc[0] * 100
                normalized_paxg = historical_data['PAXG'] / historical_data['PAXG'].iloc[0] * 100
                
                fig.add_trace(go.Scatter(x=normalized_alloy.index, y=normalized_alloy, 
                                         mode='lines', name='Alloy Strategy', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=normalized_buy_hold.index, y=normalized_buy_hold, 
                                         mode='lines', name='Buy & Hold', line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=normalized_dca.index, y=normalized_dca, 
                                         mode='lines', name='DCA', line=dict(color='green', dash='dot')))
                fig.add_trace(go.Scatter(x=normalized_btc.index, y=normalized_btc, 
                                         mode='lines', name='BTC', line=dict(color='red', opacity=0.5)))
                fig.add_trace(go.Scatter(x=normalized_paxg.index, y=normalized_paxg, 
                                         mode='lines', name='PAXG', line=dict(color='purple', opacity=0.5)))
                
                fig.update_layout(
                    title='Comparative Performance (Base 100)',
                    xaxis_title='Date',
                    yaxis_title='Performance (%)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Allocation chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=results.index, y=results['btc_allocation'] * 100,
                                         mode='lines', name='BTC Allocation', line=dict(color='orange')))
                fig2.add_trace(go.Scatter(x=results.index, y=results['paxg_allocation'] * 100,
                                         mode='lines', name='PAXG Allocation', line=dict(color='purple')))
                
                fig2.update_layout(
                    title='Dynamic Asset Allocation',
                    xaxis_title='Date',
                    yaxis_title='Allocation (%)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Drawdown chart
                fig3 = go.Figure()
                
                drawdown_alloy = (results['portfolio_value'] - results['portfolio_value'].expanding(min_periods=1).max()) / results['portfolio_value'].expanding(min_periods=1).max() * 100
                drawdown_btc = (historical_data['BTC'] - historical_data['BTC'].expanding(min_periods=1).max()) / historical_data['BTC'].expanding(min_periods=1).max() * 100
                drawdown_paxg = (historical_data['PAXG'] - historical_data['PAXG'].expanding(min_periods=1).max()) / historical_data['PAXG'].expanding(min_periods=1).max() * 100
                
                fig3.add_trace(go.Scatter(x=drawdown_alloy.index, y=drawdown_alloy,
                                         mode='lines', name='Alloy Strategy', line=dict(color='blue')))
                fig3.add_trace(go.Scatter(x=drawdown_btc.index, y=drawdown_btc,
                                         mode='lines', name='BTC', line=dict(color='red', opacity=0.5)))
                fig3.add_trace(go.Scatter(x=drawdown_paxg.index, y=drawdown_paxg,
                                         mode='lines', name='PAXG', line=dict(color='purple', opacity=0.5)))
                
                fig3.update_layout(
                    title='Drawdowns (%)',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Download results button
                csv = results.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Backtest Results as CSV",
                    data=csv,
                    file_name=f'alloy_backtest_{start_date}_{end_date}.csv',
                    mime='text/csv',
                )
            else:
                st.error("Failed to load historical data. Please try again with a different date range.")

elif page == "Trading Signals":
    st.header("Trading Signals")
    
    # Date input for signals
    signal_date = st.date_input(
        "Date for Signals",
        datetime.now() - timedelta(days=1),
        max_value=datetime.now()
    )
    
    # Parameters for signal generation
    st.subheader("Signal Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        lookback_days = st.slider("Lookback Period (days)", min_value=30, max_value=365, value=180, step=30)
        momentum_window = st.slider("Momentum Window (days)", min_value=10, max_value=60, value=30, step=5)
        
    with col2:
        momentum_threshold_bull = st.slider("Momentum Threshold Bull (%)", min_value=0, max_value=20, value=10, step=1)
        momentum_threshold_bear = st.slider("Momentum Threshold Bear (%)", min_value=-20, max_value=0, value=-5, step=1)
    
    # Generate signals button
    if st.button("Generate Trading Signals"):
        with st.spinner("Analyzing market data and generating signals..."):
            # Calculate start date for data
            start_date = (signal_date - timedelta(days=lookback_days + momentum_window)).strftime('%Y-%m-%d')
            end_date = signal_date.strftime('%Y-%m-%d')
            
            # Load historical data
            historical_data = data_manager.load_data(start_date, end_date)
            
            if historical_data is not None:
                # Generate signals
                signal_generator = SignalGenerator(
                    momentum_window=momentum_window,
                    momentum_threshold_bull=momentum_threshold_bull,
                    momentum_threshold_bear=momentum_threshold_bear
                )
                
                signal = signal_generator.generate_signal(historical_data)
                
                # Display signal
                st.subheader("Trading Signal")
                
                # Signal card
                signal_card = st.container()
                with signal_card:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if signal['decision_type'] == 'bullish':
                            st.markdown("## 📈 BULLISH")
                            signal_color = "green"
                        elif signal['decision_type'] == 'bearish':
                            st.markdown("## 📉 BEARISH")
                            signal_color = "red"
                        else:
                            st.markdown("## ↔️ NEUTRAL")
                            signal_color = "orange"
                    
                    with col2:
                        st.markdown(f"### Signal for {signal_date.strftime('%Y-%m-%d')}")
                        st.markdown(f"**Decision:** {signal['decision_reason']}")
                        
                # Market context
                st.subheader("Market Context")
                context_cols = st.columns(3)
                context_cols[0].metric("BTC Price", f"${signal['market_context']['btc_price']:,.2f}")
                context_cols[1].metric("BTC 30-Day Performance", f"{signal['market_context']['btc_30d_perf']:.2f}%")
                context_cols[2].metric("BTC Momentum", f"{signal['market_context']['btc_momentum']:.2f}%")
                
                # Recommended actions
                st.subheader("Recommended Actions")
                
                for trade in signal['trades']:
                    st.markdown(f"""
                    **{trade['type']} {trade['asset']}**: {trade['size']:.6f} units @ ${trade['price']:,.2f}
                    
                    *New allocation for {trade['asset']}: {trade['new_allocation']:.1f}%*
                    """)
                
                # Display charts
                st.subheader("Market Charts")
                
                # Create price chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(x=historical_data.index[-90:], y=historical_data['BTC'][-90:], 
                                         mode='lines', name='BTC Price', line=dict(color='orange')))
                                         
                fig.add_trace(go.Scatter(x=historical_data.index[-90:], y=historical_data['PAXG'][-90:], 
                                         mode='lines', name='PAXG Price', yaxis="y2", line=dict(color='purple')))
                
                fig.update_layout(
                    title='BTC and PAXG Price (Last 90 Days)',
                    xaxis_title='Date',
                    yaxis_title='BTC Price ($)',
                    yaxis2=dict(
                        title='PAXG Price ($)',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Momentum chart
                momentum = signal_generator.calculate_momentum(historical_data['BTC'])
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(x=momentum.index[-90:], y=momentum[-90:], 
                                         mode='lines', name='BTC Momentum', line=dict(color='blue')))
                
                fig2.add_shape(
                    type="line",
                    x0=momentum.index[-90],
                    y0=momentum_threshold_bull,
                    x1=momentum.index[-1],
                    y1=momentum_threshold_bull,
                    line=dict(color="green", width=2, dash="dash"),
                )
                
                fig2.add_shape(
                    type="line",
                    x0=momentum.index[-90],
                    y0=momentum_threshold_bear,
                    x1=momentum.index[-1],
                    y1=momentum_threshold_bear,
                    line=dict(color="red", width=2, dash="dash"),
                )
                
                fig2.add_annotation(
                    x=momentum.index[-2],
                    y=momentum_threshold_bull,
                    text="Bull Threshold",
                    showarrow=False,
                    yshift=10
                )
                
                fig2.add_annotation(
                    x=momentum.index[-2],
                    y=momentum_threshold_bear,
                    text="Bear Threshold",
                    showarrow=False,
                    yshift=-10
                )
                
                fig2.update_layout(
                    title='BTC Momentum (Last 90 Days)',
                    xaxis_title='Date',
                    yaxis_title='Momentum (%)'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
            else:
                st.error("Failed to load historical data. Please try again with a different date.")

elif page == "Strategy Optimizer":
    st.header("Strategy Parameter Optimizer")
    
    st.markdown("""
    This tool helps you find the optimal parameters for the Alloy BTC-PAXG strategy 
    based on historical data. It uses Optuna, a hyperparameter optimization framework,
    to search for the best combination of parameters.
    
    Note: Optimization can take several minutes to complete depending on the date range
    and number of trials.
    """)
    
    # Date range for optimization
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime(2020, 1, 1),
            min_value=datetime(2018, 1, 1),
            max_value=datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime(2023, 1, 1),
            min_value=start_date + timedelta(days=365),
            max_value=datetime.now()
        )
    
    # Optimization settings
    st.subheader("Optimization Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
        n_trials = st.slider("Number of Trials", min_value=10, max_value=300, value=50, step=10)
    
    with col2:
        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["Sharpe Ratio", "Total Return", "Risk-Adjusted Return", "Drawdown-Adjusted Return"]
        )
        transaction_cost = st.slider("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100
    
    # Run optimizer button
    if st.button("Run Optimizer"):
        with st.spinner("Running optimization... This may take several minutes."):
            # Load historical data
            historical_data = data_manager.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if historical_data is not None:
                # Create optimizer
                optimizer = StrategyOptimizer(
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    optimization_metric=optimization_metric.lower().replace(" ", "_"),
                    n_trials=n_trials
                )
                
                # Run optimization
                best_params, best_metrics, all_results = optimizer.optimize(historical_data)
                
                # Display results
                st.subheader("Optimization Results")
                
                # Best parameters
                st.write("Best Parameters:")
                param_cols = st.columns(4)
                param_cols[0].metric("Momentum Window", f"{best_params['momentum_window']} days")
                param_cols[1].metric("Volatility Window", f"{best_params['volatility_window']} days")
                param_cols[2].metric("Rebalance Frequency", f"{best_params['rebalance_frequency']} days")
                param_cols[3].metric("Max BTC Allocation", f"{best_params['max_btc_allocation'] * 100:.0f}%")
                
                param_cols2 = st.columns(4)
                param_cols2[0].metric("Min BTC Allocation", f"{best_params['min_btc_allocation'] * 100:.0f}%")
                param_cols2[1].metric("Momentum Bull Threshold", f"{best_params['momentum_threshold_bull']}%")
                param_cols2[2].metric("Momentum Bear Threshold", f"{best_params['momentum_threshold_bear']}%")
                
                # Best metrics
                st.write("Performance with Best Parameters:")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Total Return", f"{best_metrics['rendement_total_alloy']:.2f}%")
                metric_cols[1].metric("Annualized Return", f"{best_metrics['rendement_annualise_alloy']:.2f}%")
                metric_cols[2].metric("Sharpe Ratio", f"{best_metrics['ratio_sharpe_alloy']:.2f}")
                metric_cols[3].metric("Max Drawdown", f"{best_metrics['drawdown_maximum_alloy']:.2f}%")
                
                # Create a dataframe for top results
                top_results = []
                for i, result in enumerate(all_results[:5]):
                    param_dict = result['params']
                    metrics_dict = result['metrics']
                    
                    top_results.append({
                        'Rank': i + 1,
                        'Momentum Window': param_dict['momentum_window'],
                        'Volatility Window': param_dict['volatility_window'],
                        'Rebalance Frequency': param_dict['rebalance_frequency'],
                        'Max BTC (%)': param_dict['max_btc_allocation'] * 100,
                        'Min BTC (%)': param_dict['min_btc_allocation'] * 100,
                        'Bull Threshold (%)': param_dict['momentum_threshold_bull'],
                        'Bear Threshold (%)': param_dict['momentum_threshold_bear'],
                        'Total Return (%)': metrics_dict['rendement_total_alloy'],
                        'Annualized Return (%)': metrics_dict['rendement_annualise_alloy'],
                        'Sharpe Ratio': metrics_dict['ratio_sharpe_alloy'],
                        'Max Drawdown (%)': metrics_dict['drawdown_maximum_alloy'],
                        'Total Trades': metrics_dict['total_trades'],
                        'Total Fees ($)': metrics_dict['total_fees']
                    })
                
                # Display top 5 results table
                st.subheader("Top 5 Parameter Combinations")
                top_results_df = pd.DataFrame(top_results)
                st.dataframe(top_results_df)
                
                # Option to run backtest with optimal parameters
                st.subheader("Backtest with Optimal Parameters")
                if st.button("Run Backtest with Optimal Parameters"):
                    with st.spinner("Running backtest with optimal parameters..."):
                        # Initialize strategy with optimal parameters
                        strategy = AlloyPortfolio(
                            initial_capital=initial_capital,
                            **best_params,
                            transaction_cost=transaction_cost
                        )
                        
                        # Run backtest
                        backtest_engine = BacktestEngine(strategy)
                        results = backtest_engine.run(historical_data)
                        
                        # Display charts and metrics
                        # ... (same as in the Backtesting page)
                        
                        # Performance chart
                        fig = go.Figure()
                        
                        normalized_alloy = results['portfolio_value'] / results['portfolio_value'].iloc[0] * 100
                        normalized_buy_hold = results['buy_hold_value'] / results['buy_hold_value'].iloc[0] * 100
                        normalized_dca = results['dca_value'] / results['dca_value'].iloc[0] * 100
                        normalized_btc = historical_data['BTC'] / historical_data['BTC'].iloc[0] * 100
                        normalized_paxg = historical_data['PAXG'] / historical_data['PAXG'].iloc[0] * 100
                        
                        fig.add_trace(go.Scatter(x=normalized_alloy.index, y=normalized_alloy, 
                                               mode='lines', name='Alloy Strategy', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=normalized_buy_hold.index, y=normalized_buy_hold, 
                                               mode='lines', name='Buy & Hold', line=dict(color='orange', dash='dash')))
                        fig.add_trace(go.Scatter(x=normalized_dca.index, y=normalized_dca, 
                                               mode='lines', name='DCA', line=dict(color='green', dash='dot')))
                        fig.add_trace(go.Scatter(x=normalized_btc.index, y=normalized_btc, 
                                               mode='lines', name='BTC', line=dict(color='red', opacity=0.5)))
                        fig.add_trace(go.Scatter(x=normalized_paxg.index, y=normalized_paxg, 
                                               mode='lines', name='PAXG', line=dict(color='purple', opacity=0.5)))
                        
                        fig.update_layout(
                            title='Comparative Performance with Optimal Parameters (Base 100)',
                            xaxis_title='Date',
                            yaxis_title='Performance (%)',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download optimized parameters as JSON
                        import json
                        params_json = json.dumps(best_params, indent=4)
                        st.download_button(
                            label="Download Optimal Parameters as JSON",
                            data=params_json,
                            file_name="alloy_optimal_params.json",
                            mime="application/json"
                        )
            else:
                st.error("Failed to load historical data. Please try again with a different date range.")

else:  # About page
    st.header("About the Alloy BTC-PAXG Strategy")
    
    st.markdown("""
    ## Strategy Overview
    
    The Alloy BTC-PAXG strategy is a dynamic asset allocation approach designed to 
    preserve value by intelligently balancing between Bitcoin (BTC) and Pax Gold (PAXG) 
    based on market conditions.
    
    The strategy uses momentum and volatility signals to determine optimal allocations:
    
    - **During strong BTC bull markets**: Increase BTC allocation to capture upside
    - **During bear markets or high volatility**: Shift to PAXG for capital preservation
    - **During transitional or neutral periods**: Balance based on relative volatility
    
    ## Key Strategy Components
    
    1. **Momentum Measurement**: The strategy calculates BTC momentum over a specified window 
    to identify market trends.
    
    2. **Volatility Analysis**: Both BTC and PAXG volatility are measured to assess 
    market conditions and relative risk.
    
    3. **Dynamic Allocation**: Allocation between assets is adjusted based on market context.
    
    4. **Periodic Rebalancing**: The portfolio is rebalanced at specified intervals 
    to maintain the target allocation.
    
    ## Strategy Benefits
    
    - **Lower Correlation to BTC**: The strategy maintains a low correlation (0.17) to pure BTC holdings
    - **Protection Against Drawdowns**: Reduced drawdowns compared to BTC-only holdings
    - **Upside Participation**: Captures a portion of BTC bull markets
    - **Better Risk-Adjusted Returns**: Higher Sharpe ratio than Buy & Hold BTC
    
    ## Risk Management
    
    The strategy includes several risk management mechanisms:
    
    - **Minimum and Maximum Allocations**: Keeps allocations within defined boundaries
    - **Volatility-Based Positioning**: Reduces exposure during high volatility periods
    - **Transaction Cost Management**: Optimizes trading frequency to reduce costs
    - **Trend Confirmation**: Uses momentum to confirm market direction
    
    ## Disclaimer
    
    This strategy is provided for educational and research purposes only. It is not 
    financial advice. Cryptocurrency investments are volatile and involve significant risk. 
    Always do your own research and consider seeking advice from a qualified financial 
    advisor before making investment decisions.
    """)
    
    st.subheader("Source Code and Documentation")
    
    st.markdown("""
    The complete source code for this application is available on GitHub:
    
    [https://github.com/yourusername/alloy-btc-paxg-trading](https://github.com/yourusername/alloy-btc-paxg-trading)
    
    The repository includes:
    
    - Full implementation of the Alloy strategy
    - Backtesting engine
    - Parameter optimization tools
    - Trading signal generator
    - Detailed documentation
    
    You can customize the strategy parameters, add new assets, or extend the functionality
    to suit your specific needs.
    """)
    
    st.subheader("Community and Discussion")
    
    st.markdown("""
    Join the discussion about the Alloy BTC-PAXG strategy on Reddit:
    
    [r/PAXG - Alloy BTC-PAXG: A Dynamic Value Preservation Strategy](https://www.reddit.com/r/PAXG/)
    
    Share your results, suggest improvements, or discuss adaptations of the strategy
    with the community.
    """)
