"""
Reporting module for the Alloy BTC-PAXG strategy.
Generates trading signals, alerts, and performance reports.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generates trading signals and alerts for the Alloy BTC-PAXG strategy.
    """
    
    def __init__(self,
                 momentum_window: int = 30,
                 momentum_threshold_bull: float = 10,
                 momentum_threshold_bear: float = -5,
                 max_btc_allocation: float = 0.9,
                 min_btc_allocation: float = 0.2):
        """
        Initialize the signal generator.
        
        Args:
            momentum_window: Number of days to calculate momentum
            momentum_threshold_bull: Threshold for bullish momentum signal
            momentum_threshold_bear: Threshold for bearish momentum signal
            max_btc_allocation: Maximum allocation to BTC (0-1)
            min_btc_allocation: Minimum allocation to BTC (0-1)
        """
        self.momentum_window = momentum_window
        self.momentum_threshold_bull = momentum_threshold_bull
        self.momentum_threshold_bear = momentum_threshold_bear
        self.max_btc_allocation = max_btc_allocation
        self.min_btc_allocation = min_btc_allocation
        
        logger.info(f"SignalGenerator initialized with parameters: momentum_window={momentum_window}, "
                   f"momentum_threshold_bull={momentum_threshold_bull}, "
                   f"momentum_threshold_bear={momentum_threshold_bear}")
    
    def calculate_momentum(self, prices: pd.Series) -> pd.Series:
        """
        Calculate price momentum over the specified window.
        
        Args:
            prices: Series of asset prices
            
        Returns:
            Series of momentum values (percentage)
        """
        return (prices / prices.shift(self.momentum_window) - 1) * 100
    
    def calculate_volatility(self, prices: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculate rolling volatility over the specified window.
        
        Args:
            prices: Series of asset prices
            window: Window size for volatility calculation
            
        Returns:
            Series of annualized volatility values (percentage)
        """
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
        return volatility.fillna(method='bfill')
    
    def get_market_context(self, data: pd.DataFrame) -> Dict:
        """
        Generate a summary of current market conditions.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary with market context information
        """
        # Get latest prices
        btc_price = data['BTC'].iloc[-1]
        paxg_price = data['PAXG'].iloc[-1]
        
        # Calculate momentum
        btc_momentum = self.calculate_momentum(data['BTC'])
        current_btc_momentum = btc_momentum.iloc[-1]
        
        # Calculate volatility
        btc_volatility = self.calculate_volatility(data['BTC'])
        current_btc_volatility = btc_volatility.iloc[-1]
        
        # Calculate 30-day performance if enough data
        if len(data) >= 30:
            btc_30d_perf = ((data['BTC'].iloc[-1] / data['BTC'].iloc[-30]) - 1) * 100
            paxg_30d_perf = ((data['PAXG'].iloc[-1] / data['PAXG'].iloc[-30]) - 1) * 100
        else:
            days_available = len(data) - 1
            btc_30d_perf = ((data['BTC'].iloc[-1] / data['BTC'].iloc[0]) - 1) * 100
            paxg_30d_perf = ((data['PAXG'].iloc[-1] / data['PAXG'].iloc[0]) - 1) * 100
            logger.warning(f"Less than 30 days of data available, using {days_available} days")
        
        return {
            "date": data.index[-1],
            "btc_price": btc_price,
            "paxg_price": paxg_price,
            "btc_momentum": current_btc_momentum,
            "btc_volatility": current_btc_volatility,
            "btc_30d_perf": btc_30d_perf,
            "paxg_30d_perf": paxg_30d_perf
        }
    
    def generate_signal(self, data: pd.DataFrame, current_allocations: Dict[str, float] = None) -> Dict:
        """
        Generate a trading signal based on current market conditions.
        
        Args:
            data: DataFrame with price data
            current_allocations: Dictionary of current asset allocations (if None, assumes 50/50)
            
        Returns:
            Dictionary with signal information
        """
        # Get market context
        market_context = self.get_market_context(data)
        
        # Determine signal type based on momentum
        btc_momentum = market_context['btc_momentum']
        
        if btc_momentum > self.momentum_threshold_bull:
            # Bullish momentum - increase BTC allocation
            btc_allocation = self.max_btc_allocation
            paxg_allocation = 1 - btc_allocation
            decision_type = "bullish"
            decision_reason = f"Strong bullish momentum: {btc_momentum:.1f}% > {self.momentum_threshold_bull}%"
        
        elif btc_momentum < self.momentum_threshold_bear:
            # Bearish momentum - decrease BTC allocation
            btc_allocation = self.min_btc_allocation
            paxg_allocation = 1 - btc_allocation
            decision_type = "bearish"
            decision_reason = f"Bearish momentum: {btc_momentum:.1f}% < {self.momentum_threshold_bear}%"
        
        else:
            # Neutral momentum - allocate based on relative volatility
            btc_vol = market_context['btc_volatility']
            
            # Estimate PAXG volatility (typically much lower than BTC)
            paxg_vol = self.calculate_volatility(data['PAXG']).iloc[-1]
            
            if btc_vol == 0 or paxg_vol == 0:
                # Default allocation if volatility data is missing
                btc_allocation = 0.5
                paxg_allocation = 0.5
                decision_type = "neutral"
                decision_reason = "Default neutral allocation (volatility data missing)"
            else:
                # Allocate more to the less volatile asset
                total_vol = btc_vol + paxg_vol
                btc_allocation = max(
                    self.min_btc_allocation,
                    min(self.max_btc_allocation, 1 - (btc_vol / total_vol))
                )
                paxg_allocation = 1 - btc_allocation
                decision_type = "neutral"
                decision_reason = f"Allocation based on relative volatility (BTC: {btc_vol:.1f}%, PAXG: {paxg_vol:.1f}%)"
        
        # Calculate trade needed based on current allocations
        if current_allocations is None:
            # Assume starting from 50/50
            current_allocations = {'BTC': 0.5, 'PAXG': 0.5}
        
        # Calculate hypothetical portfolio value
        portfolio_value = 10000  # Hypothetical value for trade size calculation
        
        # Calculate current values and target values
        current_values = {
            asset: portfolio_value * alloc
            for asset, alloc in current_allocations.items()
        }
        
        target_values = {
            'BTC': portfolio_value * btc_allocation,
            'PAXG': portfolio_value * paxg_allocation
        }
        
        # Calculate trades
        trades = []
        for asset in ['BTC', 'PAXG']:
            current_value = current_values.get(asset, 0)
            target_value = target_values[asset]
            value_diff = target_value - current_value
            
            if abs(value_diff) / portfolio_value > 0.01:  # >1% change threshold
                price = market_context[f"{asset.lower()}_price"]
                size = abs(value_diff / price)
                trade_type = "BUY" if value_diff > 0 else "SELL"
                
                trades.append({
                    'asset': asset,
                    'type': trade_type,
                    'size': size,
                    'price': price,
                    'value_usd': abs(value_diff),
                    'old_allocation': current_allocations.get(asset, 0) * 100,
                    'new_allocation': target_values[asset] / portfolio_value * 100
                })
        
        # Compile signal
        signal = {
            'date': market_context['date'],
            'market_context': market_context,
            'decision_type': decision_type,
            'decision_reason': decision_reason,
            'recommended_allocations': {
                'BTC': btc_allocation,
                'PAXG': paxg_allocation
            },
            'trades': trades
        }
        
        logger.info(f"Generated signal: {decision_type} - {decision_reason}")
        return signal
    
    def generate_alert_message(self, signal: Dict) -> str:
        """
        Generate a formatted alert message for a trading signal.
        
        Args:
            signal: Signal dictionary from generate_signal
            
        Returns:
            Formatted alert message
        """
        alert = []
        alert.append("=== ALLOY BTC-PAXG TRADING SIGNAL ===")
        alert.append(f"Date: {signal['date'].strftime('%Y-%m-%d')}")
        
        # Market context
        alert.append("\nMARKET CONTEXT:")
        alert.append(f"BTC/USD: ${signal['market_context']['btc_price']:,.2f} (30d: {signal['market_context']['btc_30d_perf']:.1f}%)")
        alert.append(f"PAXG/USD: ${signal['market_context']['paxg_price']:,.2f} (30d: {signal['market_context']['paxg_30d_perf']:.1f}%)")
        alert.append(f"BTC Momentum: {signal['market_context']['btc_momentum']:.1f}%")
        
        # Decision
        alert.append(f"\nDECISION: {signal['decision_reason']}")
        
        # Recommended allocation
        alert.append("\nRECOMMENDED ALLOCATION:")
        alert.append(f"BTC: {signal['recommended_allocations']['BTC'] * 100:.1f}%")
        alert.append(f"PAXG: {signal['recommended_allocations']['PAXG'] * 100:.1f}%")
        
        # Actions
        if signal['trades']:
            alert.append("\nACTIONS REQUIRED:")
            for trade in signal['trades']:
                alert.append(f"- {trade['type']} {trade['asset']}: {trade['size']:.6f} units @ ${trade['price']:,.2f}")
        else:
            alert.append("\nNO TRADES REQUIRED")
        
        return "\n".join(alert)

class PerformanceReporter:
    """
    Generates performance reports for the Alloy BTC-PAXG strategy.
    """
    
    def __init__(self):
        """Initialize the performance reporter."""
        pass
    
    def generate_monthly_summary(self, trades_history: List[Dict]) -> str:
        """
        Generate a monthly summary of trading activity.
        
        Args:
            trades_history: List of trade dictionaries
            
        Returns:
            Formatted monthly summary
        """
        # Group trades by month
        monthly_trades = {}
        for trade in trades_history:
            month_key = trade['date'].strftime('%Y-%m')
            if month_key not in monthly_trades:
                monthly_trades[month_key] = []
            monthly_trades[month_key].append(trade)
        
        summary = []
        summary.append("=== MONTHLY TRADING SUMMARY ===\n")
        
        for month, trades in sorted(monthly_trades.items()):
            # Calculate trade volume
            btc_volume = sum(
                abs(trade['value_usd'])
                for t in trades
                for trade in t['trades']
                if trade['asset'] == 'BTC'
            )
            
            paxg_volume = sum(
                abs(trade['value_usd'])
                for t in trades
                for trade in t['trades']
                if trade['asset'] == 'PAXG'
            )
            
            # Analyze decision types
            decisions = [t['decision_reason'] for t in trades]
            momentum_decisions = len([d for d in decisions if 'momentum' in d.lower()])
            volatility_decisions = len([d for d in decisions if 'volatility' in d.lower()])
            
            summary.append(f"Month: {month}")
            summary.append(f"Number of rebalances: {len(trades)}")
            summary.append(f"Total traded volume: ${(btc_volume + paxg_volume):,.2f}")
            summary.append("Decision breakdown:")
            summary.append(f"- Momentum-based: {momentum_decisions}")
            summary.append(f"- Volatility-based: {volatility_decisions}")
            summary.append("-" * 40 + "\n")
        
        return "\n".join(summary)
    
    def generate_performance_summary(self, results: pd.DataFrame, metrics: Dict) -> str:
        """
        Generate a summary of strategy performance.
        
        Args:
            results: DataFrame with backtest results
            metrics: Dictionary of performance metrics
            
        Returns:
            Formatted performance summary
        """
        summary = []
        summary.append("=== PERFORMANCE SUMMARY ===\n")
        
        # Key metrics
        summary.append("Key Metrics:")
        summary.append(f"- Total Return: {metrics['rendement_total_alloy']:.2f}%")
        summary.append(f"- Annualized Return: {metrics['rendement_annualise_alloy']:.2f}%")
        summary.append(f"- Volatility: {metrics['volatilite_annualisee_alloy']:.2f}%")
        summary.append(f"- Sharpe Ratio: {metrics['ratio_sharpe_alloy']:.2f}")
        summary.append(f"- Maximum Drawdown: {metrics['drawdown_maximum_alloy']:.2f}%")
        
        # Benchmark comparison
        summary.append("\nBenchmark Comparison:")
        summary.append(f"- Alloy Strategy: {metrics['rendement_total_alloy']:.2f}%")
        summary.append(f"- Buy & Hold: {metrics['rendement_total_buy_hold']:.2f}%")
        summary.append(f"- Dollar Cost Averaging: {metrics['rendement_total_dca']:.2f}%")
        
        # Trading statistics
        summary.append("\nTrading Statistics:")
        summary.append(f"- Total Trades: {metrics['total_trades']}")
        summary.append(f"- Total Fees: ${metrics['total_fees']:.2f}")
        summary.append(f"- Average Allocation to BTC: {results['btc_allocation'].mean() * 100:.1f}%")
        summary.append(f"- Average Allocation to PAXG: {results['paxg_allocation'].mean() * 100:.1f}%")
        
        # Correlation
        if 'correlations' in metrics and not metrics['correlations'].empty:
            btc_correlation = metrics['correlations'].loc['portfolio', 'btc']
            summary.append(f"\nBTC Correlation: {btc_correlation:.4f}")
        
        return "\n".join(summary)
    
    def generate_html_report(self, results: pd.DataFrame, metrics: Dict, trades_history: List[Dict]) -> str:
        """
        Generate an HTML performance report.
        
        Args:
            results: DataFrame with backtest results
            metrics: Dictionary of performance metrics
            trades_history: List of trade dictionaries
            
        Returns:
            HTML report as a string
        """
        # Generate performance charts
        performance_chart = self._generate_performance_chart(results)
        allocation_chart = self._generate_allocation_chart(results)
        drawdown_chart = self._generate_drawdown_chart(results)
        
        # Create HTML report
        html = []
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Alloy BTC-PAXG Strategy Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .report-header { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
                .metrics-container { display: flex; flex-wrap: wrap; margin: 20px 0; }
                .metric-card { 
                    background-color: #fff; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 10px; 
                    width: 200px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
                .metric-label { font-size: 14px; color: #666; }
                .chart-container { margin: 30px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>Alloy BTC-PAXG Strategy Report</h1>
                <p>Period: {start_date} to {end_date}</p>
            </div>
        """.format(
            start_date=results.index[0].strftime('%Y-%m-%d'),
            end_date=results.index[-1].strftime('%Y-%m-%d')
        ))
        
        # Key Metrics Section
        html.append("""
            <h2>Key Performance Metrics</h2>
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {return_class}">{return_value}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Annualized Return</div>
                    <div class="metric-value {annual_return_class}">{annual_return}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {sharpe_class}">{sharpe}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{drawdown}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value">{volatility}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">BTC Correlation</div>
                    <div class="metric-value">{correlation}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{trades}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Fees</div>
                    <div class="metric-value">${fees}</div>
                </div>
            </div>
        """.format(
            return_value=f"{metrics['rendement_total_alloy']:.2f}",
            return_class="positive" if metrics['rendement_total_alloy'] > 0 else "negative",
            annual_return=f"{metrics['rendement_annualise_alloy']:.2f}",
            annual_return_class="positive" if metrics['rendement_annualise_alloy'] > 0 else "negative",
            sharpe=f"{metrics['ratio_sharpe_alloy']:.2f}",
            sharpe_class="positive" if metrics['ratio_sharpe_alloy'] > 0 else "negative",
            drawdown=f"{abs(metrics['drawdown_maximum_alloy']):.2f}",
            volatility=f"{metrics['volatilite_annualisee_alloy']:.2f}",
            correlation=f"{metrics['correlations'].loc['portfolio', 'btc']:.4f}" if 'correlations' in metrics and not metrics['correlations'].empty else "N/A",
            trades=metrics['total_trades'],
            fees=f"{metrics['total_fees']:.2f}"
        ))
        
        # Strategy Comparison
        html.append("""
            <h2>Strategy Comparison</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Total Return</th>
                    <th>Annualized Return</th>
                    <th>Volatility</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                </tr>
                <tr>
                    <td>Alloy Strategy</td>
                    <td class="{alloy_return_class}">{alloy_return}%</td>
                    <td class="{alloy_annual_class}">{alloy_annual}%</td>
                    <td>{alloy_vol}%</td>
                    <td class="{alloy_sharpe_class}">{alloy_sharpe}</td>
                    <td class="negative">{alloy_dd}%</td>
                </tr>
                <tr>
                    <td>Buy & Hold</td>
                    <td class="{bh_return_class}">{bh_return}%</td>
                    <td class="{bh_annual_class}">{bh_annual}%</td>
                    <td>{bh_vol}%</td>
                    <td class="{bh_sharpe_class}">{bh_sharpe}</td>
                    <td class="negative">{bh_dd}%</td>
                </tr>
                <tr>
                    <td>Dollar Cost Averaging</td>
                    <td class="{dca_return_class}">{dca_return}%</td>
                    <td class="{dca_annual_class}">{dca_annual}%</td>
                    <td>{dca_vol}%</td>
                    <td class="{dca_sharpe_class}">{dca_sharpe}</td>
                    <td class="negative">{dca_dd}%</td>
                </tr>
            </table>
        """.format(
            alloy_return=f"{metrics['rendement_total_alloy']:.2f}",
            alloy_return_class="positive" if metrics['rendement_total_alloy'] > 0 else "negative",
            alloy_annual=f"{metrics['rendement_annualise_alloy']:.2f}",
            alloy_annual_class="positive" if metrics['rendement_annualise_alloy'] > 0 else "negative",
            alloy_vol=f"{metrics['volatilite_annualisee_alloy']:.2f}",
            alloy_sharpe=f"{metrics['ratio_sharpe_alloy']:.2f}",
            alloy_sharpe_class="positive" if metrics['ratio_sharpe_alloy'] > 0 else "negative",
            alloy_dd=f"{abs(metrics['drawdown_maximum_alloy']):.2f}",
            
            bh_return=f"{metrics['rendement_total_buy_hold']:.2f}",
            bh_return_class="positive" if metrics['rendement_total_buy_hold'] > 0 else "negative",
            bh_annual=f"{metrics['rendement_annualise_buy_hold']:.2f}",
            bh_annual_class="positive" if metrics['rendement_annualise_buy_hold'] > 0 else "negative",
            bh_vol=f"{metrics['volatilite_annualisee_buy_hold']:.2f}",
            bh_sharpe=f"{metrics['ratio_sharpe_buy_hold']:.2f}",
            bh_sharpe_class="positive" if metrics['ratio_sharpe_buy_hold'] > 0 else "negative",
            bh_dd=f"{abs(metrics['drawdown_maximum_buy_hold']):.2f}",
            
            dca_return=f"{metrics['rendement_total_dca']:.2f}",
            dca_return_class="positive" if metrics['rendement_total_dca'] > 0 else "negative",
            dca_annual=f"{metrics['rendement_annualise_dca']:.2f}",
            dca_annual_class="positive" if metrics['rendement_annualise_dca'] > 0 else "negative",
            dca_vol=f"{metrics['volatilite_annualisee_dca']:.2f}",
            dca_sharpe=f"{metrics['ratio_sharpe_dca']:.2f}",
            dca_sharpe_class="positive" if metrics['ratio_sharpe_dca'] > 0 else "negative",
            dca_dd=f"{abs(metrics['drawdown_maximum_dca']):.2f}"
        ))
        
        # Charts
        html.append("""
            <h2>Performance Charts</h2>
            <div class="chart-container">
                <h3>Comparative Performance</h3>
                <img src="data:image/png;base64,{performance_chart}" alt="Performance Chart" width="100%">
            </div>
            
            <div class="chart-container">
                <h3>Portfolio Allocation</h3>
                <img src="data:image/png;base64,{allocation_chart}" alt="Allocation Chart" width="100%">
            </div>
            
            <div class="chart-container">
                <h3>Drawdown Analysis</h3>
                <img src="data:image/png;base64,{drawdown_chart}" alt="Drawdown Chart" width="100%">
            </div>
        """.format(
            performance_chart=performance_chart,
            allocation_chart=allocation_chart,
            drawdown_chart=drawdown_chart
        ))
        
        # Recent Trades
        if trades_history:
            html.append("<h2>Recent Trading Activity</h2>")
            html.append("<table>")
            html.append("<tr><th>Date</th><th>Decision</th><th>Actions</th></tr>")
            
            for trade in trades_history[-10:]:  # Show last 10 trades
                trade_details = []
                for t in trade['trades']:
                    trade_details.append(
                        f"{t['type']} {t['asset']}: {t['size']:.6f} @ ${t['price']:,.2f}"
                    )
                
                html.append(f"""
                <tr>
                    <td>{trade['date'].strftime('%Y-%m-%d')}</td>
                    <td>{trade['decision_reason']}</td>
                    <td>{' | '.join(trade_details)}</td>
                </tr>
                """)
            
            html.append("</table>")
        
        # Close HTML
        html.append("""
            <div style="margin-top: 40px; text-align: center; color: #666; font-size: 12px;">
                <p>Generated by Alloy BTC-PAXG Trading Assistant on {generation_date}</p>
                <p>Disclaimer: This report is for informational purposes only and does not constitute investment advice.</p>
            </div>
        </body>
        </html>
        """.format(generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        return "".join(html)
    
    def _generate_performance_chart(self, results: pd.DataFrame) -> str:
        """
        Generate base64-encoded performance chart image.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Base64-encoded PNG image
        """
        plt.figure(figsize=(10, 6))
        
        # Normalize values to base 100
        normalized_portfolio = results['portfolio_value'] / results['portfolio_value'].iloc[0] * 100
        normalized_buy_hold = results['buy_hold_value'] / results['buy_hold_value'].iloc[0] * 100
        normalized_dca = results['dca_value'] / results['dca_value'].iloc[0] * 100
        
        plt.plot(normalized_portfolio.index, normalized_portfolio, label='Alloy Strategy', linewidth=2)
        plt.plot(normalized_buy_hold.index, normalized_buy_hold, label='Buy & Hold', linestyle='--')
        plt.plot(normalized_dca.index, normalized_dca, label='DCA', linestyle=':')
        
        plt.title('Comparative Performance (Base 100)')
        plt.xlabel('Date')
        plt.ylabel('Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return self._fig_to_base64(plt)
    
    def _generate_allocation_chart(self, results: pd.DataFrame) -> str:
        """
        Generate base64-encoded allocation chart image.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Base64-encoded PNG image
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(results.index, results['btc_allocation'] * 100, label='BTC', color='orange', linewidth=2)
        plt.plot(results.index, results['paxg_allocation'] * 100, label='PAXG', color='purple', linewidth=2)
        
        plt.title('Portfolio Allocation')
        plt.xlabel('Date')
        plt.ylabel('Allocation (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 100)
        
        return self._fig_to_base64(plt)
    
    def _generate_drawdown_chart(self, results: pd.DataFrame) -> str:
        """
        Generate base64-encoded drawdown chart image.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Base64-encoded PNG image
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate drawdowns
        portfolio_peak = results['portfolio_value'].expanding(min_periods=1).max()
        portfolio_drawdown = ((results['portfolio_value'] - portfolio_peak) / portfolio_peak) * 100
        
        buy_hold_peak = results['buy_hold_value'].expanding(min_periods=1).max()
        buy_hold_drawdown = ((results['buy_hold_value'] - buy_hold_peak) / buy_hold_peak) * 100
        
        plt.plot(portfolio_drawdown.index, portfolio_drawdown, label='Alloy Strategy', linewidth=2)
        plt.plot(buy_hold_drawdown.index, buy_hold_drawdown, label='Buy & Hold', linestyle='--')
        
        plt.title('Drawdown Analysis')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return self._fig_to_base64(plt)
    
    def _fig_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64-encoded string.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64-encoded PNG image
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
