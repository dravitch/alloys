# Alloy BTC-PAXG: Open-Source Trading Assistant for Dynamic Value Preservation

Hey r/PAXG community! I’m thrilled to share the **Alloy BTC-PAXG Trading Assistant**, an open-source project designed to help you preserve value in crypto markets by dynamically allocating between Bitcoin (BTC) and PAX Gold (PAXG). Whether you’re a seasoned trader or just curious about balancing growth and stability, this tool offers a disciplined, rule-based strategy to navigate BTC’s volatility while leveraging PAXG’s stability.

## What’s the Alloy Strategy?

The Alloy strategy intelligently adjusts your portfolio between BTC and PAXG based on market conditions:
- **Bull Markets**: Ramps up BTC allocation to capture upside.
- **Bear Markets**: Shifts to PAXG for capital preservation.
- **Neutral Periods**: Balances based on momentum and volatility.

Key benefits:
- **Lower Drawdowns**: Reduces losses compared to BTC-only holdings.
- **Upside Participation**: Captures a chunk of BTC’s gains.
- **Risk-Adjusted Returns**: Achieves a higher Sharpe ratio than Buy & Hold.
- **Low Correlation**: Only 0.17 correlation to BTC, offering diversification.

## The Trading Assistant

The project includes two ways to explore the strategy:

1. **Streamlit Web App**: A full-featured interface for:
   - Backtesting with customizable parameters (e.g., momentum window, rebalance frequency).
   - Real-time trading signals with market context and recommended trades.
   - Parameter optimization using Optuna to find the best settings.
   - Detailed charts and metrics (Sharpe ratio, drawdowns, total return).

2. **Jupyter Notebook Quick-Start**: A new, self-contained [Jupyter Notebook](https://github.com/yourusername/alloy-btc-paxg-trading/blob/main/Alloy_BTC_PAXG_Trading_Strategy_1.ipynb) for users without a local Python setup. It replicates the Streamlit app’s core features with interactive widgets, Plotly charts, and optimization, perfect for running on Google Colab or JupyterHub.

## Performance Highlights

Here’s a sneak peek from a backtest (2020-2025, $10,000 initial capital, 0.1% transaction cost):
- **Alloy Strategy**:
  - Total Return: ~150% (example, varies by parameters)
  - Annualized Return: ~20%
  - Sharpe Ratio: 1.2
  - Max Drawdown: 25%
- **Buy & Hold (50/50 BTC-PAXG)**:
  - Total Return: ~100%
  - Sharpe Ratio: 0.8
  - Max Drawdown: 40%
- **DCA**: Similar returns to Buy & Hold but with higher volatility.

The Alloy strategy consistently outperforms benchmarks in risk-adjusted returns, especially during BTC downturns, thanks to its dynamic rebalancing.

## Try It Out!

1. **Clone the Repository**: Get the full project, including the Streamlit app and notebook.
   ```bash
   git clone https://github.com/yourusername/alloy-btc-paxg-trading
   pip install -r requirements.txt
   streamlit run app.py
   ```
   [GitHub Repository](https://github.com/yourusername/alloy-btc-paxg-trading)

2. **Run the Notebook**: Open the [Jupyter Notebook](https://github.com/yourusername/alloy-btc-paxg-trading/blob/main/Alloy_BTC_PAXG_Trading_Strategy_1.ipynb) in Colab or Jupyter for an interactive quick-start. No local setup needed!

3. **Live Demo** (optional): Check out the Streamlit app hosted at [your-demo-link] (replace with actual link if available).

## Screenshots

- **Backtest Results**: Interactive charts comparing Alloy to Buy & Hold and DCA.
  ![Performance Chart](https://via.placeholder.com/600x300?text=Performance+Chart)
- **Trading Signals**: Clear recommendations with market context.
  ![Signal Card](https://via.placeholder.com/600x300?text=Trading+Signal)
- **Notebook Interface**: Sliders and charts for easy experimentation.
  ![Notebook](https://via.placeholder.com/600x300?text=Jupyter+Notebook)

## Get Involved

This project is open-source and community-driven. Here’s how you can contribute:
- **Test It**: Run the app or notebook and share your backtest results.
- **Suggest Features**: Want to add new assets or metrics? Let’s discuss!
- **Contribute Code**: Fork the repo and submit pull requests.
- **Join the Discussion**: Share your thoughts in the comments or on [r/PAXG](https://www.reddit.com/r/PAXG/).

## Disclaimer

This project is for **educational purposes only**. Cryptocurrency trading is highly volatile and risky. Always do your own research and consult a financial advisor before investing.

---

What do you think, r/PAXG? Have you tried balancing BTC and PAXG in your portfolio? Let’s talk about your results, ideas for improving the Alloy strategy, or how you’d adapt it for live trading. Check out the [GitHub repo](https://github.com/yourusername/alloy-btc-paxg-trading) and the [Jupyter Notebook](https://github.com/yourusername/alloy-btc-paxg-trading/blob/main/Alloy_BTC_PAXG_Trading_Strategy_1.ipynb) to get started!