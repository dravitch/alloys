"""
Core strategy implementation for the Alloy BTC-PAXG portfolio.
Defines the dynamic allocation logic between BTC and PAXG based on market conditions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlloyPortfolio:
    """
    Implementation of the Alloy BTC-PAXG strategy with dynamic asset allocation.
    Uses momentum and volatility signals to balance between BTC and PAXG.
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 momentum_window: int = 30,
                 volatility_window: int = 60,
                 momentum_threshold_bull: float = 10,
                 momentum_threshold_bear: float = -5,
                 max_btc_allocation: float = 0.9,
                 min_btc_allocation: float = 0.2,
                 rebalance_frequency: int = 3,
                 transaction_cost: float = 0.001,
                 rebalance_threshold: float = 0.05):
        """
        Initialize the Alloy Portfolio strategy.
        
        Args:
            initial_capital: Initial investment amount in USD
            momentum_window: Number of days to calculate momentum
            volatility_window: Number of days to calculate volatility
            momentum_threshold_bull: Threshold for bullish momentum signal
            momentum_threshold_bear: Threshold for bearish momentum signal
            max_btc_allocation: Maximum allocation to BTC (0-1)
            min_btc_allocation: Minimum allocation to BTC (0-1)
            rebalance_frequency: Days between rebalancing checks
            transaction_cost: Transaction cost as a fraction of trade value
            rebalance_threshold: Minimum change in allocation to trigger a rebalance
        """
        # Strategy parameters
        self.initial_capital = initial_capital
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.momentum_threshold_bull = momentum_threshold_bull
        self.momentum_threshold_bear = momentum_threshold_bear
        self.max_btc_allocation = max_btc_allocation
        self.min_btc_allocation = min_btc_allocation
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.rebalance_threshold = rebalance_threshold
        
        # Initialize portfolio state
        self.btc_allocation = 0.5  # Start with 50/50 allocation
        self.paxg_allocation = 0.5
        self.positions = {'BTC': 0, 'PAXG': 0}
        self.last_rebalance_date = None
        self.trades_history = []
        self.last_momentum = None
        
        logger.info(f"AlloyPortfolio initialized with parameters: momentum_window={momentum_window}, "
                   f"volatility_window={volatility_window}, momentum_threshold_bull={momentum_threshold_bull}, "
                   f"momentum_threshold_bear={momentum_threshold_bear}, max_btc_allocation={max_btc_allocation}, "
                   f"min_btc_allocation={min_btc_allocation}, rebalance_frequency={rebalance_frequency}")
    
    def calculate_momentum(self, prices: pd.Series) -> pd.Series:
        """
        Calculate price momentum over the specified window.
        
        Args:
            prices: Series of asset prices
            
        Returns:
            Series of momentum values (percentage)
        """
        return (prices / prices.shift(self.momentum_window) - 1) * 100
    
    def calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling volatility over the specified window.
        
        Args:
            prices: Series of asset prices
            
        Returns:
            Series of annualized volatility values (percentage)
        """
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252) * 100
        return volatility.fillna(method='bfill')
    
    def get_market_context(self, data: pd.DataFrame, current_date: pd.Timestamp) -> Dict:
        """
        Generate a summary of current market conditions.
        
        Args:
            data: DataFrame with price data
            current_date: Date for which to generate the context
            
        Returns:
            Dictionary with market context information
        """
        # Get data up to current date
        historical_slice = data.loc[:current_date].copy()
        
        # Calculate momentum and volatility if enough data
        if len(historical_slice) > self.momentum_window:
            btc_momentum = self.calculate_momentum(historical_slice['BTC'])
            current_btc_momentum = btc_momentum.iloc[-1]
        else:
            current_btc_momentum = 0
        
        if len(historical_slice) > self.volatility_window:
            btc_volatility = self.calculate_volatility(historical_slice['BTC'])
            current_btc_volatility = btc_volatility.iloc[-1]
        else:
            current_btc_volatility = 0
        
        # Calculate 30-day performance if enough data
        if len(historical_slice) >= 30:
            btc_30d_perf = ((historical_slice['BTC'].iloc[-1] / historical_slice['BTC'].iloc[-30]) - 1) * 100
            paxg_30d_perf = ((historical_slice['PAXG'].iloc[-1] / historical_slice['PAXG'].iloc[-30]) - 1) * 100
        else:
            btc_30d_perf = 0
            paxg_30d_perf = 0
        
        return {
            "date": current_date,
            "btc_price": historical_slice['BTC'].iloc[-1],
            "paxg_price": historical_slice['PAXG'].iloc[-1],
            "btc_momentum": current_btc_momentum,
            "btc_volatility": current_btc_volatility,
            "btc_30d_perf": btc_30d_perf,
            "paxg_30d_perf": paxg_30d_perf
        }
    
    def determine_allocation(self, historical_data: pd.DataFrame, current_date: pd.Timestamp) -> Tuple[float, float, str]:
        """
        Determine the optimal allocation based on current market conditions.
        
        Args:
            historical_data: DataFrame with historical price data
            current_date: Current date for allocation decision
            
        Returns:
            Tuple of (btc_allocation, paxg_allocation, decision_reason)
        """
        # Get historical data up to current date
        historical_slice = historical_data.loc[:current_date].copy()
        
        # Calculate momentum
        btc_momentum = self.calculate_momentum(historical_slice['BTC'])
        current_btc_momentum = btc_momentum.iloc[-1] if not btc_momentum.empty else 0
        
        # Calculate volatility
        btc_volatility = self.calculate_volatility(historical_slice['BTC'])
        paxg_volatility = self.calculate_volatility(historical_slice['PAXG'])
        
        current_btc_vol = btc_volatility.iloc[-1] if not btc_volatility.empty else 0
        current_paxg_vol = paxg_volatility.iloc[-1] if not paxg_volatility.empty else 0
        
        old_btc_allocation = self.btc_allocation
        decision_reason = ""
        
        # Determine allocation based on momentum and volatility
        if current_btc_momentum > self.momentum_threshold_bull:
            # Bullish momentum - increase BTC allocation
            btc_allocation = self.max_btc_allocation
            paxg_allocation = 1 - btc_allocation
            decision_type = "bullish"
            decision_reason = f"Strong bullish momentum: {current_btc_momentum:.1f}% > {self.momentum_threshold_bull}%"
        
        elif current_btc_momentum < self.momentum_threshold_bear:
            # Bearish momentum - decrease BTC allocation
            btc_allocation = self.min_btc_allocation
            paxg_allocation = 1 - btc_allocation
            decision_type = "bearish"
            decision_reason = f"Bearish momentum: {current_btc_momentum:.1f}% < {self.momentum_threshold_bear}%"
        
        else:
            # Neutral momentum - allocate based on relative volatility
            if current_btc_vol == 0 or current_paxg_vol == 0:
                # Default allocation if volatility data is missing
                btc_allocation = 0.5
                paxg_allocation = 0.5
                decision_type = "neutral"
                decision_reason = "Default neutral allocation (volatility data missing)"
            else:
                # Allocate more to the less volatile asset
                total_vol = current_btc_vol + current_paxg_vol
                btc_allocation = max(
                    self.min_btc_allocation,
                    min(self.max_btc_allocation, 1 - (current_btc_vol / total_vol))
                )
                paxg_allocation = 1 - btc_allocation
                decision_type = "neutral"
                decision_reason = f"Allocation based on relative volatility (BTC: {current_btc_vol:.1f}%, PAXG: {current_paxg_vol:.1f}%)"
        
        # Log the allocation decision
        logger.debug(f"Allocation decision: {decision_reason}, BTC: {btc_allocation:.2f}, PAXG: {paxg_allocation:.2f}")
        
        return btc_allocation, paxg_allocation, decision_reason
    
    def calculate_trades(self, 
                        portfolio_value: float, 
                        current_prices: Dict[str, float],
                        target_allocations: Dict[str, float]) -> List[Dict]:
        """
        Calculate the trades needed to achieve the target allocations.
        
        Args:
            portfolio_value: Current portfolio value
            current_prices: Dictionary of current asset prices
            target_allocations: Dictionary of target asset allocations
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        
        # Calculate current values and allocations
        current_values = {
            asset: self.positions[asset] * price
            for asset, price in current_prices.items()
        }
        
        current_allocations = {
            asset: value / portfolio_value if portfolio_value > 0 else 0
            for asset, value in current_values.items()
        }
        
        # Calculate target values
        target_values = {
            asset: portfolio_value * allocation
            for asset, allocation in target_allocations.items()
        }
        
        # Calculate trades
        for asset in current_prices.keys():
            value_diff = target_values[asset] - current_values[asset]
            allocation_diff = target_allocations[asset] - current_allocations[asset]
            
            # Only trade if allocation difference exceeds threshold
            if abs(allocation_diff) > self.rebalance_threshold:
                size = abs(value_diff / current_prices[asset])
                trade_type = "BUY" if value_diff > 0 else "SELL"
                
                trades.append({
                    'asset': asset,
                    'type': trade_type,
                    'size': size,
                    'price': current_prices[asset],
                    'value_usd': abs(value_diff),
                    'old_allocation': current_allocations[asset] * 100,
                    'new_allocation': target_allocations[asset] * 100
                })
        
        return trades
    
    def execute_trades(self, trades: List[Dict], current_prices: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Execute the calculated trades and update positions.
        
        Args:
            trades: List of trade dictionaries
            current_prices: Dictionary of current asset prices
            
        Returns:
            Tuple of (transaction_cost, updated_positions)
        """
        total_transaction_cost = 0
        
        for trade in trades:
            asset = trade['asset']
            size = trade['size']
            trade_value = size * current_prices[asset]
            
            # Calculate transaction cost
            trade_cost = trade_value * self.transaction_cost
            total_transaction_cost += trade_cost
            
            # Update positions
            if trade['type'] == 'BUY':
                self.positions[asset] += size
            else:  # SELL
                self.positions[asset] -= size
        
        return total_transaction_cost
    
    def should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """
        Determine if the portfolio should be rebalanced on the current date.
        
        Args:
            current_date: Current date to check
            
        Returns:
            Boolean indicating whether to rebalance
        """
        # Always rebalance on the first day
        if self.last_rebalance_date is None:
            return True
        
        # Check if enough time has passed since last rebalance
        days_since_last = (current_date - self.last_rebalance_date).days
        return days_since_last >= self.rebalance_frequency
    
    def reset(self):
        """Reset the portfolio to its initial state."""
        self.btc_allocation = 0.5
        self.paxg_allocation = 0.5
        self.positions = {'BTC': 0, 'PAXG': 0}
        self.last_rebalance_date = None
        self.trades_history = []
        self.last_momentum = None
        
        logger.info("Portfolio reset to initial state")
