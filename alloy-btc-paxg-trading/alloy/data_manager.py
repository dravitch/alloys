"""
Data management module for the Alloy BTC-PAXG strategy.
Handles loading, processing, and validating price data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages data operations for the Alloy trading system.
    Handles loading, preprocessing, and validation of price data.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the DataManager.
        
        Args:
            cache_enabled: Whether to cache downloaded data to avoid redundant API calls
        """
        self.cache_enabled = cache_enabled
        self.data_cache = {}
        logger.info("DataManager initialized")
    
    def load_data(self, start_date: str, end_date: str, 
                 assets: List[str] = ['BTC-USD', 'PAXG-USD'],
                 column: str = 'Close') -> Optional[pd.DataFrame]:
        """
        Load historical price data for the specified assets.
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            assets: List of asset tickers to download
            column: Which price column to use ('Open', 'High', 'Low', 'Close')
            
        Returns:
            DataFrame with price data or None if download fails
        """
        # Check if data is already in cache
        cache_key = f"{start_date}_{end_date}_{'-'.join(assets)}_{column}"
        if self.cache_enabled and cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        try:
            # Add buffer for technical indicators calculation
            start_with_buffer = (pd.to_datetime(start_date) - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
            
            # Download data for each asset
            logger.info(f"Downloading data for {assets} from {start_with_buffer} to {end_date}")
            asset_data = {}
            
            for asset in assets:
                ticker_data = yf.download(asset, start=start_with_buffer, end=end_date, progress=False)
                
                if ticker_data.empty:
                    logger.warning(f"No data found for {asset}")
                    continue
                    
                asset_data[asset] = ticker_data[column]
            
            # Create combined DataFrame
            data = pd.DataFrame({asset.split('-')[0]: series for asset, series in asset_data.items()})
            
            # Ensure all required assets have data
            if not all(asset.split('-')[0] in data.columns for asset in assets):
                missing = [asset for asset in assets if asset.split('-')[0] not in data.columns]
                logger.error(f"Missing data for assets: {missing}")
                return None
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Filter to requested date range
            mask = (data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))
            data = data.loc[mask].copy()
            
            # Validate the processed data
            if not self._validate_data(data):
                return None
            
            # Cache the result if caching is enabled
            if self.cache_enabled:
                self.data_cache[cache_key] = data
            
            logger.info(f"Successfully loaded data with shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with potentially missing values
            
        Returns:
            DataFrame with handled missing values
        """
        # Check for missing values
        missing_count = data.isna().sum()
        if missing_count.sum() > 0:
            logger.warning(f"Found missing values: {missing_count}")
            
            # Forward fill missing values
            data = data.fillna(method='ffill')
            
            # If there are still missing values at the beginning, backward fill
            if data.isna().sum().sum() > 0:
                data = data.fillna(method='bfill')
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the processed data before returning it.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if DataFrame is empty
        if data.empty:
            logger.error("Data validation failed: DataFrame is empty")
            return False
        
        # Check for minimum length
        if len(data) < 30:
            logger.warning(f"Data validation warning: Only {len(data)} data points available")
        
        # Check for remaining NaN values
        if data.isna().sum().sum() > 0:
            logger.error("Data validation failed: Still have NaN values after handling")
            return False
        
        # Check for zero or negative prices
        if (data <= 0).any().any():
            logger.error("Data validation failed: Found zero or negative prices")
            return False
        
        return True
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns for each asset.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with daily returns added
        """
        returns_df = data.copy()
        
        # Calculate percentage returns
        for col in data.columns:
            returns_df[f'{col}_return'] = data[col].pct_change()
        
        return returns_df
    
    def get_latest_prices(self, assets: List[str] = ['BTC-USD', 'PAXG-USD']) -> Dict[str, float]:
        """
        Get the latest prices for the specified assets.
        
        Args:
            assets: List of asset tickers
            
        Returns:
            Dictionary mapping asset names to their latest prices
        """
        try:
            latest_prices = {}
            
            for asset in assets:
                ticker = yf.Ticker(asset)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    latest_prices[asset.split('-')[0]] = hist['Close'].iloc[-1]
                else:
                    logger.warning(f"Could not get latest price for {asset}")
            
            return latest_prices
            
        except Exception as e:
            logger.error(f"Error getting latest prices: {str(e)}")
            return {}
    
    def get_market_data_summary(self, window: int = 30) -> Dict:
        """
        Get a summary of current market data.
        
        Args:
            window: Number of days to look back for metrics
            
        Returns:
            Dictionary with market summary metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=window*2)  # Extra buffer
            
            data = self.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if data is None or data.empty:
                return {}
            
            # Calculate metrics
            btc_current = data['BTC'].iloc[-1]
            paxg_current = data['PAXG'].iloc[-1]
            
            btc_perf = (btc_current / data['BTC'].iloc[-window] - 1) * 100
            paxg_perf = (paxg_current / data['PAXG'].iloc[-window] - 1) * 100
            
            # Calculate volatility (annualized)
            btc_returns = data['BTC'].pct_change().dropna()
            paxg_returns = data['PAXG'].pct_change().dropna()
            
            btc_vol = btc_returns[-window:].std() * np.sqrt(252) * 100
            paxg_vol = paxg_returns[-window:].std() * np.sqrt(252) * 100
            
            return {
                'btc_price': btc_current,
                'paxg_price': paxg_current,
                'btc_perf': btc_perf,
                'paxg_perf': paxg_perf,
                'btc_vol': btc_vol,
                'paxg_vol': paxg_vol,
                'date': end_date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            logger.error(f"Error getting market summary: {str(e)}")
            return {}
            
    def clear_cache(self):
        """Clear the data cache."""
        self.data_cache = {}
        logger.info("Data cache cleared")
