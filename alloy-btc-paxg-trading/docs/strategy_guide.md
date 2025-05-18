# Alloy BTC-PAXG Strategy Guide

## Strategy Overview

The Alloy BTC-PAXG strategy is a dynamic asset allocation approach designed to preserve value by intelligently balancing between Bitcoin (BTC) and Pax Gold (PAXG) based on market conditions. It aims to:

- Protect against extreme market volatility
- Capture upside during bull markets
- Minimize drawdowns during bear markets
- Maintain low correlation with pure BTC holdings
- Provide better risk-adjusted returns than simple buy & hold strategies

## Core Strategy Mechanics

### 1. Momentum-Based Allocation

The strategy uses price momentum as a primary indicator to determine market direction:

- **Strong Bullish Momentum** (BTC momentum > threshold_bull): Increase BTC allocation to capture upside
- **Bearish Momentum** (BTC momentum < threshold_bear): Decrease BTC allocation to preserve capital
- **Neutral Momentum**: Balance based on relative volatility

The momentum calculation is:
```
Momentum = (Current Price / Price N days ago - 1) × 100
```

Where N is the momentum window (default: 30 days).

### 2. Volatility-Based Fine-Tuning

During neutral momentum periods, the strategy adjusts allocations based on relative volatility:

- Higher BTC volatility → Lower BTC allocation
- Lower BTC volatility → Higher BTC allocation

This approach allocates more capital to the less volatile asset, promoting stability during uncertain periods.

### 3. Bounded Allocations

To manage risk, the strategy enforces minimum and maximum allocation constraints:

- Maximum BTC allocation: 90% (default)
- Minimum BTC allocation: 20% (default)

These bounds ensure diversification and prevent extreme portfolio concentrations.

### 4. Periodic Rebalancing

The portfolio is rebalanced at specified intervals (default: every 3 days) if the desired allocation has changed significantly. This helps:

- Maintain target risk exposure
- Capture profits during momentum shifts
- Reduce trading frequency and costs

## Parameter Guide

The strategy can be customized with the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum_window` | 30 | Days to calculate momentum |
| `volatility_window` | 60 | Days to calculate volatility |
| `momentum_threshold_bull` | 10% | Threshold for bullish momentum |
| `momentum_threshold_bear` | -5% | Threshold for bearish momentum |
| `max_btc_allocation` | 90% | Maximum allocation to BTC |
| `min_btc_allocation` | 20% | Minimum allocation to BTC |
| `rebalance_frequency` | 3 | Days between rebalancing checks |
| `transaction_cost` | 0.1% | Cost of transactions |
| `rebalance_threshold` | 5% | Minimum allocation change to trigger rebalance |

## Market Regime Behavior

The strategy behaves differently in various market regimes:

### Bull Markets
- **Behavior**: Allocates maximum to BTC (90% by default)
- **Strength**: Captures substantial upside
- **Weakness**: Still underperforms pure BTC holdings
- **Expected outcome**: Good upside capture with reduced volatility

### Bear Markets
- **Behavior**: Shifts to minimum BTC allocation (20% by default)
- **Strength**: Significantly reduces drawdowns
- **Weakness**: May be slow to detect trend reversals
- **Expected outcome**: Substantial capital preservation

### Transition Phases
- **Behavior**: Gradually shifts allocations based on momentum changes
- **Strength**: Adaptive to changing conditions
- **Weakness**: May experience whipsaws during choppy markets
- **Expected outcome**: Smooth transitions between regimes

## Interpretation of Signals

### Bullish Signal
- **Indicator**: BTC momentum > threshold_bull (e.g., 10%)
- **Allocation**: Maximum BTC (e.g., 90%)
- **Rationale**: Strong uptrend detected, maximize upside exposure
- **Action required**: Rebalance to target allocation

### Bearish Signal
- **Indicator**: BTC momentum < threshold_bear (e.g., -5%)
- **Allocation**: Minimum BTC (e.g., 20%)
- **Rationale**: Downtrend detected, protect capital
- **Action required**: Rebalance to target allocation

### Neutral Signal
- **Indicator**: BTC momentum between thresholds
- **Allocation**: Determined by relative volatility (typically 40-60% BTC)
- **Rationale**: No clear trend, balance risk between assets
- **Action required**: Rebalance to target allocation

## Backtesting Results

Based on historical data from 2020-01-01 to 2024-01-05:

- **Annualized Return**: 44.11%
- **Annualized Volatility**: 32.70%
- **Sharpe Ratio**: 1.35
- **Maximum Drawdown**: 43.78%
- **BTC Correlation**: 0.1691

Compared to benchmarks:
- **BTC Buy & Hold Return**: 513.60%
- **PAXG Buy & Hold Return**: 32.37%
- **BTC Average Drawdown**: 36.72%
- **PAXG Average Drawdown**: 9.58%

## Optimization Guidance

When optimizing the strategy for your specific needs, consider the following:

1. **Risk Tolerance**:
   - Higher risk tolerance → Increase max_btc_allocation
   - Lower risk tolerance → Decrease max_btc_allocation and increase min_btc_allocation

2. **Trading Frequency**:
   - Higher trading costs → Increase rebalance_frequency and rebalance_threshold
   - Lower trading costs → Decrease rebalance_frequency for more responsive adjustments

3. **Market Responsiveness**:
   - More responsive → Decrease momentum_window
   - More stable → Increase momentum_window

4. **Bull/Bear Balance**:
   - More bullish bias → Decrease momentum_threshold_bull
   - More bearish bias → Increase momentum_threshold_bear

## Limitations and Considerations

- **Trend Following Nature**: The strategy is reactive rather than predictive
- **Parameter Sensitivity**: Performance can vary based on parameter selection
- **Market Regime Dependency**: Works best in trending markets, may struggle in sideways markets
- **Transaction Costs**: Frequent rebalancing can impact net returns
- **Tax Implications**: Rebalancing may trigger taxable events

## Implementation Best Practices

1. **Start Conservative**: Begin with default parameters and adjust gradually
2. **Monitor Regularly**: Check signals at least weekly
3. **Review Performance**: Evaluate strategy metrics quarterly
4. **Adjust Parameters**: Optimize based on recent performance and current market conditions
5. **Consider Taxes**: Plan rebalancing around tax considerations

## Disclaimer

This strategy is provided for educational and research purposes only. It is not financial advice. Cryptocurrency investments are volatile and involve significant risk. Always do your own research and consider seeking advice from a qualified financial advisor before making investment decisions.
