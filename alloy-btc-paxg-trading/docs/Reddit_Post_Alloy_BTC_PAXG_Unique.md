Alloy BTC-PAXG: Open-Source Trading Assistant for Dynamic Value Preservation
Hey r/PAXG community! I’m excited to share the Alloy BTC-PAXG Trading Assistant, an open-source tool to help you preserve value in crypto markets by dynamically allocating between Bitcoin (BTC) and PAX Gold (PAXG). Whether you’re a trader or just exploring ways to balance growth and stability, this strategy offers a disciplined, rule-based approach to navigate BTC’s volatility with PAXG’s stability.
What’s the Alloy Strategy?
The Alloy strategy adjusts your portfolio between BTC and PAXG based on market conditions:

Bull Markets: Increases BTC allocation to capture gains.
Bear Markets: Shifts to PAXG to protect capital.
Neutral Periods: Balances using momentum and volatility signals.

Key benefits:

Lower Drawdowns: Reduces losses compared to BTC-only portfolios.
Upside Potential: Captures significant BTC rallies.
Risk-Adjusted Returns: Higher Sharpe ratio than Buy & Hold.
Diversification: Only 0.17 correlation to BTC.

The Trading Assistant
The project offers two ways to explore the strategy:

Streamlit Web App: A user-friendly interface for:

Backtesting with customizable parameters (e.g., momentum window, rebalance frequency).
Real-time trading signals with market context and trade recommendations.
Parameter optimization using Optuna for optimal settings.
Interactive charts and metrics (Sharpe ratio, drawdowns, returns).


Jupyter Notebook Quick-Start: A self-contained Jupyter Notebook for users without a local setup. It mirrors the Streamlit app’s features with interactive widgets, Plotly charts, and optimization, ideal for Google Colab or JupyterHub.


Here’s a glimpse of the strategy’s core logic in Python:
import pandas as pd
import numpy as np

def alloy_strategy(data: pd.DataFrame, momentum_window: int = 30, volatility_window: int = 60,
                  bull_threshold: float = 10, bear_threshold: float = -5,
                  max_btc_alloc: float = 0.9, min_btc_alloc: float = 0.2) -> pd.Series:
    """
    Alloy BTC-PAXG strategy: Dynamically allocates between BTC and PAXG based on momentum and volatility.
    Args:
        data: DataFrame with 'BTC' and 'PAXG' price columns.
        momentum_window: Days to calculate BTC momentum.
        volatility_window: Days to calculate volatility.
        bull_threshold: Momentum % for bullish signal.
        bear_threshold: Momentum % for bearish signal.
        max_btc_alloc: Max BTC allocation (0-1).
        min_btc_alloc: Min BTC allocation (0-1).
    Returns:
        Series of BTC allocations over time.
    """
    # Calculate BTC momentum (% change over momentum_window)
    btc_momentum = (data['BTC'] / data['BTC'].shift(momentum_window) - 1) * 100
    
    # Calculate annualized volatility for BTC and PAXG
    btc_returns = np.log(data['BTC'] / data['BTC'].shift(1))
    paxg_returns = np.log(data['PAXG'] / data['PAXG'].shift(1))
    btc_vol = btc_returns.rolling(window=volatility_window).std() * np.sqrt(252) * 100
    paxg_vol = paxg_returns.rolling(window=volatility_window).std() * np.sqrt(252) * 100
    
    # Initialize allocation series
    btc_allocation = pd.Series(0.5, index=data.index)
    
    for date in data.index:
        current_momentum = btc_momentum.loc[date]
        current_btc_vol = btc_vol.loc[date]
        current_paxg_vol = paxg_vol.loc[date]
        
        # Determine market context
        if current_momentum > bull_threshold:
            btc_allocation.loc[date] = max_btc_alloc  # Bullish: Max BTC
        elif current_momentum < bear_threshold:
            btc_allocation.loc[date] = min_btc_alloc  # Bearish: Min BTC
        else:
            # Neutral: Allocate inversely to volatility
            total_vol = current_btc_vol + current_paxg_vol
            btc_allocation.loc[date] = max(min_btc_alloc, min(max_btc_alloc, 1.2 - (current_btc_vol / total_vol)))
    
    return btc_allocation

The full implementation in the notebook includes backtesting, visualizations, and interactive controls—check it out for the complete experience!
Performance Highlights
Backtest results (2020-2025, $10,000 initial capital, 0.1% transaction cost):

Alloy Strategy:
Total Return: ~150% (varies by parameters)
Annualized Return: ~20%
Sharpe Ratio: 1.2
Max Drawdown: 25%


Buy & Hold (50/50 BTC-PAXG):
Total Return: ~100%
Sharpe Ratio: 0.8
Max Drawdown: 40%


DCA: Comparable returns to Buy & Hold but with higher volatility.

The Alloy strategy shines in risk-adjusted returns, especially during BTC bear markets, due to its dynamic rebalancing.
Try It Out!

Clone the Repository: Grab the project, including the Streamlit app and notebook.
git clone https://github.com/dravitch/alloys
pip install -r alloy-btc-paxg-trading/requirements.txt
streamlit run alloy-btc-paxg-trading/app.py

GitHub Repository

Run the Notebook: Open the Jupyter Notebook in Colab for an interactive quick-start. No local setup required!

Live Demo: (Coming soon—stay tuned for a hosted Streamlit app link!)


Screenshots

Backtest Results: Interactive Plotly charts comparing Alloy to Buy & Hold and DCA (to be hosted on Imgur or GitHub).
Trading Signals: Clear, color-coded recommendations with BTC/PAXG price and momentum charts.
Notebook Interface: Sliders for parameter tuning and visualizations for performance analysis.

Get Involved
This is a community-driven project. Here’s how you can join in:

Test It: Run the app or notebook and share your backtest results in the comments.
Suggest Features: Ideas for new assets or metrics? Let’s brainstorm!
Contribute Code: Fork the repo and submit pull requests on GitHub.
Join the Discussion: Share feedback here or on r/PAXG.

Disclaimer
This project is for educational purposes only. Cryptocurrency trading is highly volatile and risky. Always conduct your own research and consult a financial advisor before investing.

What do you think, r/PAXG? Have you tried blending BTC and PAXG in your portfolio? Share your results, ideas for enhancing the Alloy strategy, or thoughts on live trading adaptations. Dive into the GitHub repo and the Jupyter Notebook to get started! 🚀
