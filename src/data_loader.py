"""
Data loading module for the Cross-Sectional Momentum Strategy.

Handles downloading and preprocessing price data from Yahoo Finance.
Data is cached locally to avoid repeated API calls.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")

from .config import Config, get_config


def get_data_path() -> Path:
    """Return the path to the data directory."""
    return Path(__file__).parent.parent / "data"


def load_universe_data(
    config: Optional[Config] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load adjusted close prices for all stocks in the universe.

    Parameters
    ----------
    config : Config, optional
        Strategy configuration. Uses default if not provided.
    use_cache : bool, default True
        Whether to use cached data if available.
    verbose : bool, default True
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and stock tickers as columns.
        Values are adjusted close prices.

    Notes
    -----
    - Uses Yahoo Finance adjusted close prices (accounts for splits and dividends)
    - Data is resampled to monthly frequency (end of month)
    - Missing values are forward-filled then backward-filled
    """
    if config is None:
        config = get_config()

    cache_file = get_data_path() / "universe_prices.parquet"

    # Try to load from cache
    if use_cache and cache_file.exists():
        if verbose:
            print(f"Loading cached data from {cache_file}")
        prices = pd.read_parquet(cache_file)
        return prices

    # Download fresh data
    if verbose:
        print(f"Downloading data for {len(config.DOW_JONES_30)} stocks...")

    tickers = config.DOW_JONES_30.copy()

    # Download daily data
    try:
        data = yf.download(
            tickers=tickers,
            start=config.START_DATE,
            end=config.END_DATE,
            auto_adjust=True,  # Use adjusted prices
            progress=verbose,
            group_by='ticker',  # Group by ticker for consistent format
        )
    except Exception as e:
        if verbose:
            print(f"Error downloading data: {e}")
        # Return empty DataFrame
        return pd.DataFrame()

    # Handle empty data
    if data.empty:
        if verbose:
            print("Warning: No data downloaded")
        return pd.DataFrame()

    # Extract Close prices - handle different yfinance output formats
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker case with MultiIndex columns
        # Try to get Close prices
        try:
            # New yfinance format: (ticker, field)
            if data.columns.nlevels == 2:
                # Check if first level is tickers or fields
                first_level = data.columns.get_level_values(0).unique()
                if 'Close' in first_level:
                    prices = data["Close"]
                else:
                    # Format is (ticker, field)
                    prices = data.xs('Close', axis=1, level=1)
            else:
                prices = data["Close"]
        except KeyError:
            # Fallback: try to extract Close from each ticker
            prices_dict = {}
            for ticker in tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        prices_dict[ticker] = data[ticker]["Close"]
                    elif (ticker, "Close") in data.columns:
                        prices_dict[ticker] = data[(ticker, "Close")]
                except (KeyError, TypeError):
                    continue
            prices = pd.DataFrame(prices_dict)
    else:
        # Single ticker case
        if "Close" in data.columns:
            prices = data[["Close"]]
            prices.columns = tickers[:1]
        else:
            prices = data
            prices.columns = tickers[:len(data.columns)]

    # Ensure we have data
    if prices.empty:
        if verbose:
            print("Warning: Could not extract price data")
        return pd.DataFrame()

    # Resample to monthly frequency (end of month)
    prices = prices.resample("ME").last()

    # Handle missing values
    # Forward-fill first (carry last known price)
    # Then backward-fill (for stocks that start trading later)
    prices = prices.ffill().bfill()

    # Report data quality
    if verbose:
        print(f"\nData loaded: {prices.shape[0]} months, {prices.shape[1]} stocks")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

        missing = prices.isna().sum().sum()
        if missing > 0:
            print(f"Warning: {missing} missing values remain")
        else:
            print("No missing values")

    # Cache the data
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_file)
    if verbose:
        print(f"Data cached to {cache_file}")

    return prices


def load_benchmark_data(
    config: Optional[Config] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load benchmark (S&P 500) price data.

    Parameters
    ----------
    config : Config, optional
        Strategy configuration. Uses default if not provided.
    use_cache : bool, default True
        Whether to use cached data if available.
    verbose : bool, default True
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and 'SPY' column.
        Values are adjusted close prices.
    """
    if config is None:
        config = get_config()

    cache_file = get_data_path() / "benchmark_prices.parquet"

    # Try to load from cache
    if use_cache and cache_file.exists():
        if verbose:
            print(f"Loading cached benchmark data from {cache_file}")
        prices = pd.read_parquet(cache_file)
        return prices

    # Download fresh data
    if verbose:
        print(f"Downloading benchmark data ({config.BENCHMARK_TICKER})...")

    data = yf.download(
        tickers=config.BENCHMARK_TICKER,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=True,
        progress=verbose,
    )

    # Extract Close prices
    prices = data[["Close"]].copy()
    prices.columns = [config.BENCHMARK_TICKER]

    # Keep daily data for regime filter (SMA200 needs daily)
    prices_daily = prices.copy()

    # Also create monthly version
    prices_monthly = prices.resample("ME").last()

    # Cache both versions
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices_monthly.to_parquet(cache_file)

    cache_file_daily = get_data_path() / "benchmark_prices_daily.parquet"
    prices_daily.to_parquet(cache_file_daily)

    if verbose:
        print(f"Benchmark data cached to {cache_file}")

    return prices_monthly


def load_benchmark_daily(
    config: Optional[Config] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load daily benchmark data (needed for regime filter SMA200).

    Parameters
    ----------
    config : Config, optional
        Strategy configuration. Uses default if not provided.
    use_cache : bool, default True
        Whether to use cached data if available.
    verbose : bool, default True
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with daily DatetimeIndex and benchmark column.
    """
    if config is None:
        config = get_config()

    cache_file = get_data_path() / "benchmark_prices_daily.parquet"

    # Try to load from cache
    if use_cache and cache_file.exists():
        if verbose:
            print(f"Loading cached daily benchmark data from {cache_file}")
        return pd.read_parquet(cache_file)

    # Need to download - call load_benchmark_data which creates both files
    load_benchmark_data(config, use_cache=False, verbose=verbose)

    return pd.read_parquet(cache_file)


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Compute simple returns from prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex.
    periods : int, default 1
        Number of periods for return calculation.

    Returns
    -------
    pd.DataFrame
        Returns with same shape as input (first `periods` rows will be NaN).
    """
    return prices.pct_change(periods=periods)


def compute_log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Compute log returns from prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex.
    periods : int, default 1
        Number of periods for return calculation.

    Returns
    -------
    pd.DataFrame
        Log returns with same shape as input.
    """
    return np.log(prices / prices.shift(periods))


def clear_cache() -> None:
    """Remove all cached data files."""
    data_path = get_data_path()
    if data_path.exists():
        for file in data_path.glob("*.parquet"):
            file.unlink()
            print(f"Removed {file}")
