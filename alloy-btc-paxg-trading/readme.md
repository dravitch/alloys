# Alloy BTC-PAXG: Dynamic Value Preservation Strategy

![Alloy Strategy](https://img.shields.io/badge/Strategy-Anti--Fragile-blue)
![Crypto](https://img.shields.io/badge/Assets-BTC%20%7C%20PAXG-orange)
![Version](https://img.shields.io/badge/Version-1.0.0-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/alloy-btc-paxg-trading/blob/main/notebooks/alloy_btc_paxg_quickstart.ipynb)

The Alloy BTC-PAXG strategy is a dynamic asset allocation approach designed to preserve value by intelligently balancing between Bitcoin (BTC) and Pax Gold (PAXG) based on market conditions.

## 📊 Strategy Overview

The Alloy strategy aims to:

- Protect against extreme market volatility
- Capture upside during bull markets
- Minimize drawdowns during bear markets
- Maintain low correlation with pure BTC holdings
- Provide better risk-adjusted returns than simple Buy & Hold

The strategy uses momentum and volatility signals to dynamically adjust allocations between BTC and PAXG, providing an anti-fragile portfolio that can weather various market conditions.

## 🚀 Quick Start Options

### Option 1: Run in Google Colab (No Installation Required)

Get started immediately without installing anything:

1. Click the "Open in Colab" button at the top of this README
2. Run the cells in sequence by clicking the play button or pressing Shift+Enter
3. Use the interactive widgets to adjust parameters and explore the strategy
4. Generate trading signals and analyze performance metrics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/alloy-btc-paxg-trading/blob/main/notebooks/alloy_btc_paxg_quickstart.ipynb)

### Option 2: Run the Jupyter Notebook Locally

If you prefer to run the notebook on your own machine:

```bash
# Clone the repository
git clone https://github.com/yourusername/alloy-btc-paxg-trading.git
cd alloy-btc-paxg-trading

# Install required packages
pip install pandas numpy matplotlib plotly yfinance optuna ipywidgets

# Launch Jupyter
jupyter notebook notebooks/Alloy_BTC_PAXG_QuickStart.ipynb
```

### Option 3: Run the Full Streamlit Application

For the complete experience with all features:

```bash
# Clone the repository
git clone https://github.com/yourusername/alloy-btc-paxg-trading.git
cd alloy-btc-paxg-trading

# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app.py
```

## 💻 Features

- **Backtesting Engine**: Test the strategy against historical data
- **Performance Metrics**: Comprehensive performance analysis
- **Trading Signals**: Get actionable buy/sell recommendations
- **Parameter Optimization**: Find optimal settings for your risk profile
- **Benchmark Comparison**: Compare with Buy & Hold and DCA strategies
- **Interactive Visualization**: Explore performance through interactive charts

## ⚙️ Configuration

You can customize the strategy by adjusting the following parameters:

- `momentum_window`: Window size for momentum calculation (default: 30 days)
- `volatility_window`: Window size for volatility calculation (default: 60 days)
- `momentum_threshold_bull`: Threshold for bullish momentum (default: 10%)
- `momentum_threshold_bear`: Threshold for bearish momentum (default: -5%)
- `max_btc_allocation`: Maximum BTC allocation (default: 90%)
- `min_btc_allocation`: Minimum BTC allocation (default: 20%)
- `rebalance_frequency`: Days between rebalancing (default: 3 days)

## 📈 Sample Results

Based on historical data from 2020-01-01 to 2024-01-05:

- **Annualized Return**: 44.11%
- **Annualized Volatility**: 32.70%
- **Sharpe Ratio**: 1.35
- **Maximum Drawdown**: 43.78%
- **BTC Correlation**: 0.1691

Compared to benchmarks:
- **BTC Buy & Hold Return**: 513.60%
- **PAXG Buy & Hold Return**: 32.37%

## 📚 Documentation

Detailed documentation is available in the `docs` folder:

- [Strategy Guide](docs/strategy_guide.md): In-depth explanation of the strategy
- [Trading Signals Guide](docs/trading_signals.md): How to interpret and use trading signals

## 🛠️ Project Structure

```
alloy-btc-paxg-trading/
├── app.py                 # Main Streamlit web application
├── notebooks/             # Jupyter notebooks
│   └── alloy_btc_paxg_quickstart.ipynb  # Interactive quick start
├── alloy/                 # Core package
│   ├── data_manager.py    # Data handling module
│   ├── strategy.py        # Strategy implementation
│   ├── backtester.py      # Backtesting engine
│   ├── optimizer.py       # Parameter optimization
│   └── reporting.py       # Reporting and signals
├── docs/                  # Documentation
│   ├── strategy_guide.md  # Strategy description
│   └── trading_signals.md # Signal interpretation guide
└── tests/                 # Test suite
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, suggest improvements, or create pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is not financial advice. Cryptocurrency investments are volatile and risky. Always do your own research and consider seeking advice from a qualified financial advisor before making investment decisions.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
