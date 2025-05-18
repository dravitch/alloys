"""
Backtesting engine for the Alloy BTC-PAXG strategy.
Handles running tests against historical data and calculating performance metrics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for running backtests of the Alloy BTC-PAXG strategy against historical data.
    Handles running the strategy, tracking performance, and calculating metrics.
    """
    
    def __init__(self, strategy):
        """
        Initialize the backtest engine with a strategy.
        
        Args:
            strategy: An instance of AlloyPortfolio
        """
        self.strategy = strategy
        logger.info("BacktestEngine initialized")
    
    def run(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run a backtest of the strategy against historical data.
        
        Args:
            historical_data: DataFrame with historical price data for BTC and PAXG
            
        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Starting backtest with {len(historical_data)} data points")
        
        # Reset the strategy to initial state
        self.strategy.reset()
        
        # Initialize results tracking
        results = []
        
        # Calculate initial positions
        initial_btc_price = historical_data['BTC'].iloc[0]
        initial_paxg_price = historical_data['PAXG'].iloc[0]
        initial_capital = self.strategy.initial_capital
        
        # Set initial positions (50/50 allocation)
        self.strategy.positions['BTC'] = (initial_capital * 0.5) / initial_btc_price
        self.strategy.positions['PAXG'] = (initial_capital * 0.5) / initial_paxg_price
        
        # Track buy and hold performance
        bh_btc_position = (initial_capital * 0.5) / initial_btc_price
        bh_paxg_position = (initial_capital * 0.5) / initial_paxg_price
        
        # Track DCA performance
        remaining_capital = initial_capital
        monthly_investment = initial_capital / (len(historical_data) / 30) if len(historical_data) > 30 else initial_capital / 12
        dca_btc_position = 0.0
        dca_paxg_position = 0.0
        last_investment_date = historical_data.index[0]
        
        # Initialize tracking variables
        cumulative_fees = 0.0
        
        # Main backtest loop
        for current_date, row in historical_data.iterrows():
            current_prices = {'BTC': row['BTC'], 'PAXG': row['PAXG']}
            
            # Calculate current portfolio value
            portfolio_value = (
                self.strategy.positions['BTC'] * current_prices['BTC'] +
                self.strategy.positions['PAXG'] * current_prices['PAXG']
            )
            
            # Check if rebalancing is needed
            if self.strategy.should_rebalance(current_date):
                # Determine optimal allocations
                btc_allocation, paxg_allocation, decision_reason = self.strategy.determine_allocation(
                    historical_data.loc[:current_date], current_date
                )
                
                # Update strategy allocations
                self.strategy.btc_allocation = btc_allocation
                self.strategy.paxg_allocation = paxg_allocation
                
                # Calculate and execute trades
                target_allocations = {'BTC': btc_allocation, 'PAXG': paxg_allocation}
                trades = self.strategy.calculate_trades(
                    portfolio_value, current_prices, target_allocations
                )
                
                if trades:
                    # Execute trades and calculate transaction costs
                    trade_costs = self.strategy.execute_trades(trades, current_prices)
                    cumulative_fees += trade_costs
                    
                    # Record trade in history
                    market_context = self.strategy.get_market_context(
                        historical_data, current_date
                    )
                    
                    self.strategy.trades_history.append({
                        'date': current_date,
                        'portfolio_value': portfolio_value,
                        'trades': trades,
                        'transaction_cost': trade_costs,
                        'market_context': market_context,
                        'decision_reason': decision_reason
                    })
                
                # Update last rebalance date
                self.strategy.last_rebalance_date = current_date
            
            # Update DCA performance
            days_since_last_investment = (current_date - last_investment_date).days
            if days_since_last_investment >= 30 and remaining_capital > 0:
                investment_amount = min(monthly_investment, remaining_capital)
                dca_btc_position += (investment_amount * 0.5) / current_prices['BTC']
                dca_paxg_position += (investment_amount * 0.5) / current_prices['PAXG']
                remaining_capital -= investment_amount
                last_investment_date = current_date
            
            # Calculate updated portfolio value after fees
            adjusted_portfolio_value = (
                self.strategy.positions['BTC'] * current_prices['BTC'] +
                self.strategy.positions['PAXG'] * current_prices['PAXG']
            ) - cumulative_fees
            
            # Calculate buy and hold value
            buy_hold_value = (
                bh_btc_position * current_prices['BTC'] +
                bh_paxg_position * current_prices['PAXG']
            )
            
            # Calculate DCA value
            dca_value = (
                dca_btc_position * current_prices['BTC'] +
                dca_paxg_position * current_prices['PAXG'] +
                remaining_capital
            )
            
            # Record results for this date
            results.append({
                'date': current_date,
                'portfolio_value': adjusted_portfolio_value,
                'buy_hold_value': buy_hold_value,
                'dca_value': dca_value,
                'btc_allocation': self.strategy.btc_allocation,
                'paxg_allocation': self.strategy.paxg_allocation,
                'btc_position': self.strategy.positions['BTC'] * current_prices['BTC'],
                'paxg_position': self.strategy.positions['PAXG'] * current_prices['PAXG'],
                'transaction_fees': 0.0,  # Will be filled on trade days
                'cumulative_fees': cumulative_fees
            })
            
            # Update transaction fees on trade days
            if self.strategy.trades_history and self.strategy.trades_history[-1]['date'] == current_date:
                results[-1]['transaction_fees'] = self.strategy.trades_history[-1]['transaction_cost']
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        
        logger.info(f"Backtest completed with {len(self.strategy.trades_history)} trades")
        return results_df
    
    def calculate_metrics(self, results: pd.DataFrame, historical_data: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for the backtest results.
        
        Args:
            results: DataFrame with backtest results
            historical_data: Original historical price data
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate daily returns
        portfolio_returns = results['portfolio_value'].pct_change().dropna()
        buy_hold_returns = results['buy_hold_value'].pct_change().dropna()
        dca_returns = results['dca_value'].pct_change().dropna()
        
        # Calculate drawdowns
        portfolio_drawdown = self._calculate_drawdown(results['portfolio_value'])
        buy_hold_drawdown = self._calculate_drawdown(results['buy_hold_value'])
        dca_drawdown = self._calculate_drawdown(results['dca_value'])
        btc_drawdown = self._calculate_drawdown(historical_data['BTC'])
        paxg_drawdown = self._calculate_drawdown(historical_data['PAXG'])
        
        # Calculate total returns
        portfolio_total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0] - 1) * 100
        buy_hold_total_return = (results['buy_hold_value'].iloc[-1] / results['buy_hold_value'].iloc[0] - 1) * 100
        dca_total_return = (results['dca_value'].iloc[-1] / results['dca_value'].iloc[0] - 1) * 100
        
        # Calculate annualized returns
        days = (results.index[-1] - results.index[0]).days
        years = days / 365.25
        
        # Only annualize if we have more than 30 days of data
        if days > 30:
            portfolio_annual_return = (portfolio_returns.mean() * 252) * 100
            buy_hold_annual_return = (buy_hold_returns.mean() * 252) * 100
            dca_annual_return = (dca_returns.mean() * 252) * 100
            
            portfolio_annual_vol = (portfolio_returns.std() * np.sqrt(252)) * 100
            buy_hold_annual_vol = (buy_hold_returns.std() * np.sqrt(252)) * 100
            dca_annual_vol = (dca_returns.std() * np.sqrt(252)) * 100
        else:
            # If less than 30 days, use total return
            portfolio_annual_return = portfolio_total_return
            buy_hold_annual_return = buy_hold_total_return
            dca_annual_return = dca_total_return
            
            portfolio_annual_vol = portfolio_returns.std() * 100
            buy_hold_annual_vol = buy_hold_returns.std() * 100
            dca_annual_vol = dca_returns.std() * 100
        
        # Calculate Sharpe ratio (using risk-free rate of 2%)
        risk_free_rate = 0.02  # 2% annual risk-free rate
        
        portfolio_sharpe = ((portfolio_annual_return / 100) - risk_free_rate) / (portfolio_annual_vol / 100) if portfolio_annual_vol > 0 else 0
        buy_hold_sharpe = ((buy_hold_annual_return / 100) - risk_free_rate) / (buy_hold_annual_vol / 100) if buy_hold_annual_vol > 0 else 0
        dca_sharpe = ((dca_annual_return / 100) - risk_free_rate) / (dca_annual_vol / 100) if dca_annual_vol > 0 else 0
        
        # Calculate correlation
        btc_returns = historical_data['BTC'].pct_change().dropna()
        correlations = pd.DataFrame({
            'portfolio': portfolio_returns,
            'btc': btc_returns.loc[portfolio_returns.index] if len(btc_returns) >= len(portfolio_returns) else pd.Series(0, index=portfolio_returns.index)
        }).corr()
        
        # Compile metrics
        metrics = {
            'total_trades': len(self.strategy.trades_history),
            'total_fees': results['cumulative_fees'].iloc[-1],
            'rendement_total_alloy': portfolio_total_return,
            'rendement_total_buy_hold': buy_hold_total_return,
            'rendement_total_dca': dca_total_return,
            'rendement_annualise_alloy': portfolio_annual_return,
            'rendement_annualise_buy_hold': buy_hold_annual_return,
            'rendement_annualise_dca': dca_annual_return,
            'volatilite_annualisee_alloy': portfolio_annual_vol,
            'volatilite_annualisee_buy_hold': buy_hold_annual_vol,
            'volatilite_annualisee_dca': dca_annual_vol,
            'ratio_sharpe_alloy': portfolio_sharpe,
            'ratio_sharpe_buy_hold': buy_hold_sharpe,
            'ratio_sharpe_dca': dca_sharpe,
            'drawdown_maximum_alloy': portfolio_drawdown.min() * 100,
            'drawdown_maximum_buy_hold': buy_hold_drawdown.min() * 100,
            'drawdown_maximum_dca': dca_drawdown.min() * 100,
            'drawdown_moyen_alloy': portfolio_drawdown.mean() * 100,
            'drawdown_moyen_btc': btc_drawdown.mean() * 100,
            'drawdown_moyen_paxg': paxg_drawdown.mean() * 100,
            'correlations': correlations
        }
        
        logger.info(f"Metrics calculated: Alloy Return: {portfolio_total_return:.2f}%, "
                   f"Buy & Hold Return: {buy_hold_total_return:.2f}%, "
                   f"DCA Return: {dca_total_return:.2f}%")
        
        return metrics
    
    def _calculate_drawdown(self, values: pd.Series) -> pd.Series:
        """
        Calculate the drawdown series for a given price series.
        
        Args:
            values: Series of asset prices or portfolio values
            
        Returns:
            Series of drawdown values (0 to -1)
        """
        # Calculate drawdown as (current value - peak value) / peak value
        rolling_max = values.expanding(min_periods=1).max()
        drawdown = (values - rolling_max) / rolling_max
        return drawdown
    
    def save_results(self, results: pd.DataFrame, filename: str):
        """
        Save backtest results to a CSV file.
        
        Args:
            results: DataFrame with backtest results
            filename: Path where to save the CSV file
        """
        try:
            results.to_csv(filename)
            logger.info(f"Backtest results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
