"""
Test suite for the Alloy BTC-PAXG strategy.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from alloy.strategy import AlloyPortfolio
from alloy.backtester import BacktestEngine
from alloy.optimizer import StrategyOptimizer
from alloy.reporting import SignalGenerator, PerformanceReporter

class TestAlloyStrategy(unittest.TestCase):
    """Test cases for the AlloyPortfolio strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample historical dataset
        dates = pd.date_range(start='2020-01-01', end='2020-03-01')
        btc_prices = pd.Series(np.linspace(8000, 10000, len(dates)), index=dates)
        paxg_prices = pd.Series(np.linspace(1500, 1600, len(dates)), index=dates)
        
        self.historical_data = pd.DataFrame({
            'BTC': btc_prices,
            'PAXG': paxg_prices
        })
        
        # Create strategy instance
        self.strategy = AlloyPortfolio(
            initial_capital=10000,
            momentum_window=20,
            volatility_window=30,
            momentum_threshold_bull=10,
            momentum_threshold_bear=-5,
            max_btc_allocation=0.9,
            min_btc_allocation=0.2,
            rebalance_frequency=5,
            transaction_cost=0.001
        )
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.initial_capital, 10000)
        self.assertEqual(self.strategy.momentum_window, 20)
        self.assertEqual(self.strategy.btc_allocation, 0.5)
        self.assertEqual(self.strategy.paxg_allocation, 0.5)
        self.assertEqual(len(self.strategy.positions), 2)
    
    def test_calculate_momentum(self):
        """Test momentum calculation."""
        momentum = self.strategy.calculate_momentum(self.historical_data['BTC'])
        
        # Check result shape
        self.assertEqual(len(momentum), len(self.historical_data))
        
        # First N values should be NaN where N is momentum_window
        self.assertTrue(momentum.iloc[:self.strategy.momentum_window].isna().all())
        
        # Check momentum values
        btc_first = self.historical_data['BTC'].iloc[0]
        btc_last = self.historical_data['BTC'].iloc[-1]
        window = self.strategy.momentum_window
        
        # For a linear increase, the momentum should be positive
        expected_momentum = ((btc_last / btc_first) - 1) * 100
        self.assertGreater(momentum.iloc[-1], 0)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        volatility = self.strategy.calculate_volatility(self.historical_data['BTC'])
        
        # Check result shape
        self.assertEqual(len(volatility), len(self.historical_data))
        
        # Check if volatility values are reasonable
        self.assertGreaterEqual(volatility.min(), 0)
    
    def test_determine_allocation(self):
        """Test allocation determination."""
        # Create a scenario with strong bullish momentum
        dates = pd.date_range(start='2020-01-01', end='2020-03-01')
        btc_prices = pd.Series(np.linspace(8000, 12000, len(dates)), index=dates)
        paxg_prices = pd.Series(np.linspace(1500, 1600, len(dates)), index=dates)
        
        strong_bull_data = pd.DataFrame({
            'BTC': btc_prices,
            'PAXG': paxg_prices
        })
        
        # Force momentum to be above threshold
        with patch.object(self.strategy, 'calculate_momentum', return_value=pd.Series([15] * len(dates), index=dates)):
            btc_alloc, paxg_alloc, reason = self.strategy.determine_allocation(strong_bull_data, dates[-1])
            
            # Should allocate maximum to BTC
            self.assertEqual(btc_alloc, self.strategy.max_btc_allocation)
            self.assertEqual(paxg_alloc, 1 - self.strategy.max_btc_allocation)
            self.assertTrue("bullish" in reason.lower())
    
    def test_calculate_trades(self):
        """Test trade calculation."""
        # Current prices
        current_prices = {'BTC': 10000, 'PAXG': 1600}
        
        # Current allocation is 50/50, target is 80/20
        portfolio_value = 10000
        target_allocations = {'BTC': 0.8, 'PAXG': 0.2}
        
        # Assume current positions
        self.strategy.positions = {'BTC': 0.5, 'PAXG': 3.125}  # 5000 in BTC, 5000 in PAXG
        
        trades = self.strategy.calculate_trades(portfolio_value, current_prices, target_allocations)
        
        # Should have two trades (one for each asset)
        self.assertEqual(len(trades), 2)
        
        # Verify trade details
        btc_trade = next((t for t in trades if t['asset'] == 'BTC'), None)
        paxg_trade = next((t for t in trades if t['asset'] == 'PAXG'), None)
        
        self.assertEqual(btc_trade['type'], 'BUY')
        self.assertEqual(paxg_trade['type'], 'SELL')
        
        # Check trade values
        self.assertAlmostEqual(btc_trade['value_usd'], 3000, places=0)  # Buy $3000 more of BTC
        self.assertAlmostEqual(paxg_trade['value_usd'], 3000, places=0)  # Sell $3000 of PAXG
    
    def test_reset(self):
        """Test strategy reset."""
        # Modify state
        self.strategy.btc_allocation = 0.8
        self.strategy.positions = {'BTC': 1, 'PAXG': 2}
        self.strategy.last_rebalance_date = datetime.now()
        
        # Reset
        self.strategy.reset()
        
        # Check reset state
        self.assertEqual(self.strategy.btc_allocation, 0.5)
        self.assertEqual(self.strategy.paxg_allocation, 0.5)
        self.assertEqual(self.strategy.positions, {'BTC': 0, 'PAXG': 0})
        self.assertIsNone(self.strategy.last_rebalance_date)

class TestBacktester(unittest.TestCase):
    """Test cases for the BacktestEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample historical dataset
        dates = pd.date_range(start='2020-01-01', end='2020-02-01')
        btc_prices = pd.Series(np.linspace(8000, 10000, len(dates)), index=dates)
        paxg_prices = pd.Series(np.linspace(1500, 1600, len(dates)), index=dates)
        
        self.historical_data = pd.DataFrame({
            'BTC': btc_prices,
            'PAXG': paxg_prices
        })
        
        # Create strategy and backtest engine
        self.strategy = AlloyPortfolio(
            initial_capital=10000,
            momentum_window=10,
            volatility_window=15,
            rebalance_frequency=7
        )
        
        self.backtest_engine = BacktestEngine(self.strategy)
    
    def test_run_backtest(self):
        """Test backtest execution."""
        # Run the backtest
        results = self.backtest_engine.run(self.historical_data)
        
        # Check results format
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), len(self.historical_data))
        
        # Check expected columns
        expected_columns = [
            'portfolio_value', 'buy_hold_value', 'dca_value',
            'btc_allocation', 'paxg_allocation',
            'btc_position', 'paxg_position'
        ]
        
        for col in expected_columns:
            self.assertIn(col, results.columns)
        
        # Check initial and final values
        self.assertAlmostEqual(results['portfolio_value'].iloc[0], 10000, delta=100)
        self.assertGreater(results['portfolio_value'].iloc[-1], 0)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Run the backtest first
        results = self.backtest_engine.run(self.historical_data)
        
        # Calculate metrics
        metrics = self.backtest_engine.calculate_metrics(results, self.historical_data)
        
        # Check metrics exist
        expected_metrics = [
            'rendement_total_alloy', 'rendement_total_buy_hold',
            'rendement_annualise_alloy', 'volatilite_annualisee_alloy',
            'ratio_sharpe_alloy', 'drawdown_maximum_alloy'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check metrics are reasonable
        self.assertGreaterEqual(metrics['rendement_total_alloy'], -100)
        self.assertGreaterEqual(metrics['volatilite_annualisee_alloy'], 0)
        self.assertLessEqual(metrics['drawdown_maximum_alloy'], 0)

class TestSignalGenerator(unittest.TestCase):
    """Test cases for the SignalGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample historical dataset
        dates = pd.date_range(start='2020-01-01', end='2020-03-01')
        btc_prices = pd.Series(np.linspace(8000, 10000, len(dates)), index=dates)
        paxg_prices = pd.Series(np.linspace(1500, 1600, len(dates)), index=dates)
        
        self.historical_data = pd.DataFrame({
            'BTC': btc_prices,
            'PAXG': paxg_prices
        })
        
        # Create signal generator
        self.signal_generator = SignalGenerator(
            momentum_window=20,
            momentum_threshold_bull=10,
            momentum_threshold_bear=-5
        )
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Generate a signal
        signal = self.signal_generator.generate_signal(self.historical_data)
        
        # Check signal format
        self.assertIsInstance(signal, dict)
        
        # Check expected keys
        expected_keys = [
            'date', 'market_context', 'decision_type',
            'decision_reason', 'recommended_allocations', 'trades'
        ]
        
        for key in expected_keys:
            self.assertIn(key, signal)
        
        # Check allocations add up to 1
        allocations = signal['recommended_allocations']
        self.assertAlmostEqual(allocations['BTC'] + allocations['PAXG'], 1.0)
    
    def test_generate_alert_message(self):
        """Test alert message generation."""
        # First generate a signal
        signal = self.signal_generator.generate_signal(self.historical_data)
        
        # Generate alert message
        alert = self.signal_generator.generate_alert_message(signal)
        
        # Check alert format
        self.assertIsInstance(alert, str)
        self.assertTrue(len(alert) > 0)
        
        # Check content
        self.assertIn("TRADING SIGNAL", alert)
        self.assertIn("MARKET CONTEXT", alert)
        self.assertIn("RECOMMENDED ALLOCATION", alert)

if __name__ == '__main__':
    unittest.main()
