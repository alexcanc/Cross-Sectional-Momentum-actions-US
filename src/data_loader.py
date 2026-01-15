"""
Data loading module for the Cross-Sectional Momentum Strategy.

Handles downloading and preprocessing price data from Yahoo Finance.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")

from .config import Config, get_config


def get_data_path() -> Path:
    """Return the path to the data directory."""
    return Path(__file__).parent.parent / "data"


def _safe_cache_write(df: pd.DataFrame, cache_file: Path) -> bool:
    """Safely write DataFrame to cache file."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        return True
    except Exception:
        return False


def _safe_cache_read(cache_file: Path) -> Optional[pd.DataFrame]:
    """Safely read DataFrame from cache file."""
    try:
        if cache_file.exists():
            return pd.read_parquet(cache_file)
    except Exception:
        pass
    return None


def _extract_close_prices(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Extract Close prices from yfinance data, handling different formats."""
    if data.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if data.columns.nlevels == 2:
                first_level = data.columns.get_level_values(0).unique().tolist()

                # Check format: (Price, Ticker) or (Ticker, Price)
                if 'Close' in first_level:
                    return data["Close"].copy()
                elif any(t in first_level for t in tickers):
                    # Format is (Ticker, Price)
                    return data.xs('Close', axis=1, level=1).copy()
        except (KeyError, TypeError):
            pass

        # Fallback: extract manually
        prices_dict = {}
        for ticker in tickers:
            for col in data.columns:
                if ticker in col and 'Close' in str(col):
                    try:
                        prices_dict[ticker] = data[col]
                        break
                    except (KeyError, TypeError):
                        continue
        if prices_dict:
            return pd.DataFrame(prices_dict)

    # Single column case
    if "Close" in data.columns:
        df = data[["Close"]].copy()
        df.columns = tickers[:1]
        return df

    # Last resort: assume all columns are prices
    return data.copy()


def _download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    verbose: bool = False
) -> pd.DataFrame:
    """Download stock data from Yahoo Finance with error handling."""
    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=verbose,
            threads=True,
        )
        return data
    except Exception as e:
        if verbose:
            print(f"Download error: {e}")
        return pd.DataFrame()


def load_universe_data(
    config: Optional[Config] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load adjusted close prices for all stocks in the universe.

    Returns DataFrame with DatetimeIndex and stock tickers as columns.
    """
    if config is None:
        config = get_config()

    cache_file = get_data_path() / "universe_prices.parquet"

    # Try cache first
    if use_cache:
        cached = _safe_cache_read(cache_file)
        if cached is not None and not cached.empty:
            if verbose:
                print(f"Loaded from cache: {cached.shape}")
            return cached

    # Download fresh data
    if verbose:
        print(f"Downloading {len(config.DOW_JONES_30)} stocks...")

    tickers = config.DOW_JONES_30.copy()
    data = _download_stock_data(tickers, config.START_DATE, config.END_DATE, verbose)

    if data.empty:
        if verbose:
            print("Warning: No data downloaded")
        return pd.DataFrame()

    # Extract close prices
    prices = _extract_close_prices(data, tickers)

    if prices.empty:
        if verbose:
            print("Warning: Could not extract prices")
        return pd.DataFrame()

    # Resample to monthly
    prices = prices.resample("ME").last()

    # Fill missing values
    prices = prices.ffill().bfill()

    if verbose:
        print(f"Loaded: {prices.shape[0]} months, {prices.shape[1]} stocks")

    # Try to cache
    _safe_cache_write(prices, cache_file)

    return prices


def load_benchmark_data(
    config: Optional[Config] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load monthly benchmark (SPY) price data."""
    if config is None:
        config = get_config()

    cache_file = get_data_path() / "benchmark_prices.parquet"

    # Try cache
    if use_cache:
        cached = _safe_cache_read(cache_file)
        if cached is not None and not cached.empty:
            return cached

    # Download
    if verbose:
        print(f"Downloading benchmark ({config.BENCHMARK_TICKER})...")

    data = _download_stock_data(
        [config.BENCHMARK_TICKER],
        config.START_DATE,
        config.END_DATE,
        verbose
    )

    if data.empty:
        return pd.DataFrame()

    # Extract close prices
    if "Close" in data.columns:
        prices = data[["Close"]].copy()
    elif isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data.xs('Close', axis=1, level=1).copy()
        except (KeyError, TypeError):
            prices = _extract_close_prices(data, [config.BENCHMARK_TICKER])
    else:
        prices = data.copy()

    if prices.empty:
        return pd.DataFrame()

    prices.columns = [config.BENCHMARK_TICKER]

    # Save daily version for regime filter
    prices_daily = prices.copy()
    cache_file_daily = get_data_path() / "benchmark_prices_daily.parquet"
    _safe_cache_write(prices_daily, cache_file_daily)

    # Resample to monthly
    prices_monthly = prices.resample("ME").last()

    # Cache monthly
    _safe_cache_write(prices_monthly, cache_file)

    return prices_monthly


def load_benchmark_daily(
    config: Optional[Config] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load daily benchmark data (for SMA200 regime filter)."""
    if config is None:
        config = get_config()

    cache_file = get_data_path() / "benchmark_prices_daily.parquet"

    # Try cache
    if use_cache:
        cached = _safe_cache_read(cache_file)
        if cached is not None and not cached.empty:
            return cached

    # Download fresh
    if verbose:
        print(f"Downloading daily benchmark ({config.BENCHMARK_TICKER})...")

    data = _download_stock_data(
        [config.BENCHMARK_TICKER],
        config.START_DATE,
        config.END_DATE,
        verbose
    )

    if data.empty:
        return pd.DataFrame()

    # Extract close prices
    if "Close" in data.columns:
        prices = data[["Close"]].copy()
    elif isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data.xs('Close', axis=1, level=1).copy()
        except (KeyError, TypeError):
            prices = _extract_close_prices(data, [config.BENCHMARK_TICKER])
    else:
        prices = data.copy()

    if prices.empty:
        return pd.DataFrame()

    prices.columns = [config.BENCHMARK_TICKER]

    # Cache
    _safe_cache_write(prices, cache_file)

    return prices


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Compute simple returns from prices."""
    return prices.pct_change(periods=periods, fill_method=None)


def compute_log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Compute log returns from prices."""
    return np.log(prices / prices.shift(periods))


def clear_cache() -> None:
    """Remove all cached data files."""
    data_path = get_data_path()
    if data_path.exists():
        for file in data_path.glob("*.parquet"):
            try:
                file.unlink()
                print(f"Removed {file}")
            except Exception:
                pass
