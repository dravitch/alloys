"""
Strategy parameter optimizer for the Alloy BTC-PAXG strategy.
Uses Optuna to find optimal parameters based on historical performance.
"""

import optuna
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from .strategy import AlloyPortfolio
from .backtester import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Optimizer for the Alloy BTC-PAXG strategy parameters.
    Uses Optuna to find optimal parameters based on historical performance.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 transaction_cost: float = 0.001,
                 optimization_metric: str = 'sharpe_ratio',
                 n_trials: int = 100):
        """
        Initialize the strategy optimizer.
        
        Args:
            initial_capital: Initial investment amount
            transaction_cost: Transaction cost as a fraction of trade value
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'risk_adjusted_return', 'drawdown_adjusted_return')
            n_trials: Number of optimization trials to run
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        
        logger.info(f"StrategyOptimizer initialized with {n_trials} trials, "
                   f"optimizing for {optimization_metric}")
    
    def optimize(self, historical_data: pd.DataFrame) -> Tuple[Dict, Dict, List]:
        """
        Run parameter optimization on historical data.
        
        Args:
            historical_data: DataFrame with historical price data
            
        Returns:
            Tuple of (best_params, best_metrics, all_results)
        """
        logger.info(f"Starting optimization with {self.n_trials} trials on {len(historical_data)} data points")
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        
        # Create a lambda that captures the historical_data
        objective = lambda trial: self._objective(trial, historical_data)
        
        # Run optimization
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # Run backtest with best parameters
        strategy = AlloyPortfolio(
            initial_capital=self.initial_capital,
            momentum_window=best_params['momentum_window'],
            volatility_window=best_params['volatility_window'],
            momentum_threshold_bull=best_params['momentum_threshold_bull'],
            momentum_threshold_bear=best_params['momentum_threshold_bear'],
            max_btc_allocation=best_params['max_btc_allocation'],
            min_btc_allocation=best_params['min_btc_allocation'],
            rebalance_frequency=best_params['rebalance_frequency'],
            transaction_cost=self.transaction_cost
        )
        
        backtest_engine = BacktestEngine(strategy)
        results = backtest_engine.run(historical_data)
        best_metrics = backtest_engine.calculate_metrics(results, historical_data)
        
        # Get all results for comparison
        all_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params = trial.params
                
                # Run backtest with these parameters
                strategy = AlloyPortfolio(
                    initial_capital=self.initial_capital,
                    momentum_window=params['momentum_window'],
                    volatility_window=params['volatility_window'],
                    momentum_threshold_bull=params['momentum_threshold_bull'],
                    momentum_threshold_bear=params['momentum_threshold_bear'],
                    max_btc_allocation=params['max_btc_allocation'],
                    min_btc_allocation=params['min_btc_allocation'],
                    rebalance_frequency=params['rebalance_frequency'],
                    transaction_cost=self.transaction_cost
                )
                
                backtest_engine = BacktestEngine(strategy)
                results = backtest_engine.run(historical_data)
                metrics = backtest_engine.calculate_metrics(results, historical_data)
                
                all_results.append({
                    'params': params,
                    'metrics': metrics,
                    'score': trial.value
                })
        
        # Sort results by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Optimization complete. Best {self.optimization_metric}: {study.best_value:.4f}")
        
        return best_params, best_metrics, all_results
    
    def _objective(self, trial, historical_data: pd.DataFrame) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            historical_data: DataFrame with historical price data
            
        Returns:
            Score to maximize
        """
        # Define parameter ranges
        params = {
            'momentum_window': trial.suggest_int('momentum_window', 10, 60),
            'volatility_window': trial.suggest_int('volatility_window', 20, 100),
            'momentum_threshold_bull': trial.suggest_float('momentum_threshold_bull', 0, 20),
            'momentum_threshold_bear': trial.suggest_float('momentum_threshold_bear', -20, 0),
            'max_btc_allocation': trial.suggest_float('max_btc_allocation', 0.5, 1.0),
            'min_btc_allocation': trial.suggest_float('min_btc_allocation', 0.0, 0.5),
            'rebalance_frequency': trial.suggest_int('rebalance_frequency', 1, 30),
        }
        
        # Ensure min_btc_allocation < max_btc_allocation
        if params['min_btc_allocation'] >= params['max_btc_allocation']:
            return float('-inf')
        
        # Ensure momentum_threshold_bear < momentum_threshold_bull
        if params['momentum_threshold_bear'] >= params['momentum_threshold_bull']:
            return float('-inf')
        
        # Create strategy with these parameters
        strategy = AlloyPortfolio(
            initial_capital=self.initial_capital,
            **params,
            transaction_cost=self.transaction_cost
        )
        
        # Run backtest
        backtest_engine = BacktestEngine(strategy)
        results = backtest_engine.run(historical_data)
        metrics = backtest_engine.calculate_metrics(results, historical_data)
        
        # Calculate score based on chosen optimization metric
        if self.optimization_metric == 'sharpe_ratio':
            score = metrics['ratio_sharpe_alloy']
        
        elif self.optimization_metric == 'total_return':
            score = metrics['rendement_total_alloy']
        
        elif self.optimization_metric == 'risk_adjusted_return':
            # Combine return and volatility
            score = metrics['rendement_annualise_alloy'] / (metrics['volatilite_annualisee_alloy'] + 1e-6)
        
        elif self.optimization_metric == 'drawdown_adjusted_return':
            # Penalize drawdowns
            drawdown_factor = 1 + abs(metrics['drawdown_maximum_alloy']) / 100
            score = metrics['rendement_annualise_alloy'] / drawdown_factor
        
        else:
            # Default to Sharpe ratio
            score = metrics['ratio_sharpe_alloy']
        
        # Add a penalty for excessive trading
        trade_penalty = max(0, metrics['total_trades'] - 52) / 52  # More than weekly is penalized
        
        # Add a penalty for excessive fees
        fee_penalty = metrics['total_fees'] / self.initial_capital
        
        # Adjust score with penalties
        adjusted_score = score - trade_penalty - 10 * fee_penalty
        
        # Log trial progress (but not too frequently)
        if trial.number % 10 == 0 or trial.number == 0:
            logger.info(f"Trial {trial.number}: Score = {adjusted_score:.4f}, "
                       f"Return = {metrics['rendement_annualise_alloy']:.2f}%, "
                       f"Sharpe = {metrics['ratio_sharpe_alloy']:.2f}, "
                       f"Trades = {metrics['total_trades']}")
        
        return adjusted_score
    
    def save_optimization_results(self, best_params: Dict, filename: str):
        """
        Save optimization results to a JSON file.
        
        Args:
            best_params: Dictionary of best parameters
            filename: Path where to save the JSON file
        """
        try:
            with open(filename, 'w') as f:
                json.dump(best_params, f, indent=4)
            logger.info(f"Optimization results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving optimization results: {str(e)}")
    
    def load_optimization_results(self, filename: str) -> Dict:
        """
        Load optimization results from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            Dictionary of parameters
        """
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            logger.info(f"Optimization results loaded from {filename}")
            return params
        except Exception as e:
            logger.error(f"Error loading optimization results: {str(e)}")
            return {}
