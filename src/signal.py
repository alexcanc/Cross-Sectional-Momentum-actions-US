"""
Momentum signal computation module.

Implements the classic 12-1 momentum signal following Jegadeesh & Titman (1993).

IMPORTANT: No data leakage
- Signal at time t uses only information available at time t
- We skip the most recent month to avoid short-term reversal effects
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

from .config import Config, get_config


def compute_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """
    Compute the 12-1 momentum signal for all stocks.

    The signal is computed as:
        Momentum(t) = Price(t-skip) / Price(t-lookback) - 1

    This measures the return from t-12 to t-1 (excluding the most recent month).

    Parameters
    ----------
    prices : pd.DataFrame
        Monthly price data with DatetimeIndex and stock tickers as columns.
    lookback : int, default 12
        Total lookback period in months.
    skip : int, default 1
        Number of recent months to skip (to avoid short-term reversal).

    Returns
    -------
    pd.DataFrame
        Momentum scores with same index as prices.
        First `lookback` rows will be NaN.

    Notes
    -----
    - No future information is used (look-ahead bias free)
    - Signal at time t can be used to form portfolio at time t+1
    - Higher momentum score = stronger positive momentum

    Example
    -------
    For lookback=12 and skip=1:
    - At end of December, we compute return from Jan to Nov (11 months)
    - This signal is used to form the January portfolio
    """
    # Compute return over the momentum window
    # Price at t-skip / Price at t-lookback - 1
    momentum = prices.shift(skip) / prices.shift(lookback) - 1

    return momentum


def rank_momentum(
    momentum: pd.DataFrame,
    method: str = "average",
) -> pd.DataFrame:
    """
    Rank stocks cross-sectionally by momentum each period.

    Parameters
    ----------
    momentum : pd.DataFrame
        Momentum scores from compute_momentum_signal.
    method : str, default 'average'
        Ranking method for ties. Options: 'average', 'min', 'max', 'first', 'dense'.

    Returns
    -------
    pd.DataFrame
        Cross-sectional ranks (1 = lowest momentum, N = highest momentum).

    Notes
    -----
    - Ranking is done independently for each time period
    - NaN values are excluded from ranking
    - Higher rank = higher momentum = candidate for long portfolio
    """
    # Rank across columns (stocks) for each row (date)
    # ascending=True means rank 1 = lowest value
    ranks = momentum.rank(axis=1, method=method, ascending=True)

    return ranks


def get_long_short_positions(
    momentum: pd.DataFrame,
    n_long: int = 10,
    n_short: int = 10,
    config: Optional[Config] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify long and short positions based on momentum ranking.

    Parameters
    ----------
    momentum : pd.DataFrame
        Momentum scores from compute_momentum_signal.
    n_long : int, default 10
        Number of stocks in long portfolio.
    n_short : int, default 10
        Number of stocks in short portfolio.
    config : Config, optional
        Strategy configuration.

    Returns
    -------
    long_positions : pd.DataFrame
        Boolean DataFrame indicating long positions.
    short_positions : pd.DataFrame
        Boolean DataFrame indicating short positions.

    Notes
    -----
    - Long: top n_long stocks by momentum
    - Short: bottom n_short stocks by momentum
    - Positions are mutually exclusive (no overlap)
    """
    if config is None:
        config = get_config()

    # Get ranks
    ranks = rank_momentum(momentum)

    # Count valid (non-NaN) stocks per period
    n_valid = ranks.notna().sum(axis=1)

    # Long positions: top n_long stocks (highest ranks)
    # Rank threshold for long: must be > (n_valid - n_long)
    long_threshold = n_valid - n_long
    long_positions = ranks.gt(long_threshold, axis=0) & ranks.notna()

    # Short positions: bottom n_short stocks (lowest ranks)
    # Rank threshold for short: must be <= n_short
    short_positions = ranks.le(n_short) & ranks.notna()

    return long_positions, short_positions


def compute_portfolio_weights(
    momentum: pd.DataFrame,
    portfolio_type: str = "long_short",
    n_long: int = 10,
    n_short: int = 10,
    weighting: str = "equal_weight",
) -> pd.DataFrame:
    """
    Compute portfolio weights based on momentum signal.

    Parameters
    ----------
    momentum : pd.DataFrame
        Momentum scores from compute_momentum_signal.
    portfolio_type : str, default 'long_short'
        'long_short' or 'long_only'.
    n_long : int, default 10
        Number of stocks in long portfolio.
    n_short : int, default 10
        Number of stocks in short portfolio.
    weighting : str, default 'equal_weight'
        'equal_weight' or 'rank_weight'.

    Returns
    -------
    pd.DataFrame
        Portfolio weights. Positive = long, negative = short.
        For long_short: weights sum to 0 (dollar-neutral)
        For long_only: weights sum to 1

    Notes
    -----
    - Equal weight: Each position has equal absolute weight
    - Rank weight: Positions weighted by momentum rank (stronger signal = larger position)
    """
    long_pos, short_pos = get_long_short_positions(momentum, n_long, n_short)

    # Initialize weights
    weights = pd.DataFrame(0.0, index=momentum.index, columns=momentum.columns)

    if weighting == "equal_weight":
        # Long weights: +1/n_long each
        long_weight = 1.0 / n_long
        weights = weights.add(long_pos.astype(float) * long_weight)

        if portfolio_type == "long_short":
            # Short weights: -1/n_short each
            short_weight = -1.0 / n_short
            weights = weights.add(short_pos.astype(float) * short_weight)

    elif weighting == "rank_weight":
        # Weight by rank (higher rank = larger weight)
        ranks = rank_momentum(momentum)
        n_stocks = ranks.notna().sum(axis=1)

        # Normalize ranks to [0, 1]
        norm_ranks = (ranks.sub(1, axis=0)).div(n_stocks - 1, axis=0)

        # Long weights proportional to (rank - threshold)
        long_weights = long_pos.astype(float) * norm_ranks
        long_weights = long_weights.div(long_weights.sum(axis=1), axis=0)
        weights = weights.add(long_weights.fillna(0))

        if portfolio_type == "long_short":
            # Short weights proportional to (threshold - rank)
            short_weights = short_pos.astype(float) * (1 - norm_ranks)
            short_weights = short_weights.div(short_weights.sum(axis=1), axis=0)
            weights = weights.sub(short_weights.fillna(0))

    return weights


def compute_signal_summary(momentum: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics of momentum signal.

    Parameters
    ----------
    momentum : pd.DataFrame
        Momentum scores from compute_momentum_signal.

    Returns
    -------
    pd.DataFrame
        Summary statistics per stock and cross-sectionally.
    """
    # Per-stock statistics
    stock_stats = momentum.describe().T
    stock_stats.columns = [f"mom_{col}" for col in stock_stats.columns]

    # Add hit rate (% of positive momentum periods)
    stock_stats["hit_rate"] = (momentum > 0).sum() / momentum.notna().sum()

    # Cross-sectional statistics per period
    cross_sectional = pd.DataFrame({
        "mean": momentum.mean(axis=1),
        "std": momentum.std(axis=1),
        "min": momentum.min(axis=1),
        "max": momentum.max(axis=1),
        "spread": momentum.max(axis=1) - momentum.min(axis=1),
    })

    return stock_stats, cross_sectional
