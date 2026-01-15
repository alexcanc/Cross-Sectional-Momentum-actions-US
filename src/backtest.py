"""
Backtesting engine for the Cross-Sectional Momentum Strategy.

Handles portfolio construction, rebalancing, and return calculation
with realistic transaction costs.

IMPORTANT: No look-ahead bias
- Signal at t is used to form portfolio at t+1
- Returns are computed using prices at t+1 and t+2
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .config import Config, get_config
from .signal import compute_momentum_signal, compute_portfolio_weights


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Core results
    strategy_returns: pd.Series  # Monthly strategy returns (after costs)
    strategy_returns_gross: pd.Series  # Monthly returns before costs
    benchmark_returns: pd.Series  # Monthly benchmark returns
    cumulative_returns: pd.Series  # Cumulative strategy returns
    cumulative_benchmark: pd.Series  # Cumulative benchmark returns

    # Portfolio details
    weights: pd.DataFrame  # Portfolio weights over time
    turnover: pd.Series  # Monthly turnover
    transaction_costs: pd.Series  # Monthly transaction costs

    # Configuration used
    config: Config


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns from prices."""
    return prices.pct_change(fill_method=None)


def compute_turnover(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute portfolio turnover.

    Turnover = sum of absolute weight changes at each rebalance.
    We account for drift from returns before rebalancing.

    Parameters
    ----------
    weights : pd.DataFrame
        Target portfolio weights.
    returns : pd.DataFrame
        Asset returns.

    Returns
    -------
    pd.Series
        Turnover per period (two-way turnover, i.e., buys + sells).
    """
    # Compute drifted weights (what weights become after returns)
    # w_drifted = w * (1 + r) / sum(w * (1 + r))

    weights_prev = weights.shift(1).fillna(0)

    # Simple approximation: just compute absolute change in target weights
    # This slightly overestimates turnover but is conservative
    turnover = weights.sub(weights_prev).abs().sum(axis=1)

    return turnover


def run_backtest(
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.DataFrame] = None,
    config: Optional[Config] = None,
    regime_signal: Optional[pd.Series] = None,
) -> BacktestResult:
    """
    Run the momentum strategy backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Monthly price data for universe stocks.
    benchmark_prices : pd.DataFrame, optional
        Monthly benchmark prices. If None, uses equal-weight universe.
    config : Config, optional
        Strategy configuration.
    regime_signal : pd.Series, optional
        Regime filter signal (1 = risk-on, 0 = risk-off).
        If provided, exposure is scaled during risk-off periods.

    Returns
    -------
    BacktestResult
        Complete backtest results.

    Notes
    -----
    Timeline for avoiding look-ahead bias:
    - End of month t: observe prices, compute signal
    - Start of month t+1: form portfolio based on signal
    - End of month t+1: observe returns, rebalance

    In practice with monthly data:
    - Signal at t uses prices up to t
    - Portfolio formed at t earns return from t to t+1
    """
    if config is None:
        config = get_config()

    # Step 1: Compute momentum signal
    # Signal at t uses prices from t-12 to t-1
    momentum = compute_momentum_signal(
        prices,
        lookback=config.MOMENTUM_LOOKBACK,
        skip=config.MOMENTUM_SKIP,
    )

    # Step 2: Compute target portfolio weights
    weights = compute_portfolio_weights(
        momentum,
        portfolio_type=config.PORTFOLIO_TYPE,
        n_long=config.N_LONG,
        n_short=config.N_SHORT,
        weighting=config.WEIGHTING,
    )

    # Step 3: Apply regime filter if provided
    if regime_signal is not None:
        # Scale weights by regime signal
        # Risk-off: reduce exposure to RISK_OFF_EXPOSURE
        exposure_multiplier = regime_signal.reindex(weights.index).fillna(1)
        exposure_multiplier = exposure_multiplier.replace(
            0, config.RISK_OFF_EXPOSURE
        )
        weights = weights.mul(exposure_multiplier, axis=0)

    # Step 4: Compute asset returns
    # Return at t = (price at t / price at t-1) - 1
    asset_returns = compute_returns(prices)

    # Step 5: Compute portfolio returns
    # IMPORTANT: Weights at t determine returns from t to t+1
    # So we use weights shifted by 1 period
    weights_lagged = weights.shift(1)

    # Portfolio return = sum of (weight * return)
    portfolio_returns_gross = (weights_lagged * asset_returns).sum(axis=1)

    # Step 6: Compute turnover and transaction costs
    turnover = compute_turnover(weights, asset_returns)

    # Transaction costs = turnover * cost per trade
    # Turnover is two-way (buys + sells), so multiply by cost once
    transaction_costs = turnover * config.transaction_cost

    # Net returns = gross returns - transaction costs
    portfolio_returns_net = portfolio_returns_gross - transaction_costs

    # Step 7: Compute benchmark returns
    if benchmark_prices is not None:
        if isinstance(benchmark_prices, pd.DataFrame):
            benchmark_returns = benchmark_prices.pct_change(fill_method=None).iloc[:, 0]
        else:
            benchmark_returns = benchmark_prices.pct_change(fill_method=None)
    else:
        # Use equal-weight universe as benchmark
        benchmark_returns = asset_returns.mean(axis=1)

    # Align indices
    common_idx = portfolio_returns_net.dropna().index
    portfolio_returns_net = portfolio_returns_net.loc[common_idx]
    portfolio_returns_gross = portfolio_returns_gross.loc[common_idx]
    benchmark_returns = benchmark_returns.reindex(common_idx).fillna(0)
    weights = weights.loc[common_idx]
    turnover = turnover.loc[common_idx]
    transaction_costs = transaction_costs.loc[common_idx]

    # Step 8: Compute cumulative returns
    cumulative_strategy = (1 + portfolio_returns_net).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()

    return BacktestResult(
        strategy_returns=portfolio_returns_net,
        strategy_returns_gross=portfolio_returns_gross,
        benchmark_returns=benchmark_returns,
        cumulative_returns=cumulative_strategy,
        cumulative_benchmark=cumulative_benchmark,
        weights=weights,
        turnover=turnover,
        transaction_costs=transaction_costs,
        config=config,
    )


def run_backtest_long_only(
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.DataFrame] = None,
    config: Optional[Config] = None,
    regime_signal: Optional[pd.Series] = None,
) -> BacktestResult:
    """
    Run the momentum strategy backtest (long-only variant).

    This is a convenience wrapper that sets portfolio_type='long_only'.
    """
    if config is None:
        config = get_config()

    # Override portfolio type
    config.PORTFOLIO_TYPE = "long_only"
    config.N_SHORT = 0

    return run_backtest(prices, benchmark_prices, config, regime_signal)


def compute_drawdowns(cumulative_returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series from cumulative returns.

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative returns (growth of $1).

    Returns
    -------
    pd.Series
        Drawdown at each point (negative values).
    """
    # Running maximum
    running_max = cumulative_returns.cummax()

    # Drawdown = (current - max) / max
    drawdowns = (cumulative_returns - running_max) / running_max

    return drawdowns


def analyze_drawdowns(cumulative_returns: pd.Series) -> pd.DataFrame:
    """
    Analyze drawdown periods.

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative returns (growth of $1).

    Returns
    -------
    pd.DataFrame
        Table of major drawdowns with start, trough, end, depth, duration.
    """
    drawdowns = compute_drawdowns(cumulative_returns)

    # Find drawdown periods
    is_underwater = drawdowns < 0

    # Identify drawdown periods (sequences of underwater returns)
    period_changes = is_underwater.astype(int).diff().fillna(0)
    period_starts = period_changes[period_changes == 1].index
    period_ends = period_changes[period_changes == -1].index

    # Match starts with ends
    drawdown_periods = []

    for start in period_starts:
        # Find the next end after this start
        future_ends = period_ends[period_ends > start]
        if len(future_ends) > 0:
            end = future_ends[0]
        else:
            end = cumulative_returns.index[-1]

        # Find trough
        period_dd = drawdowns.loc[start:end]
        trough_date = period_dd.idxmin()
        trough_value = period_dd.min()

        # Duration in months
        duration = len(cumulative_returns.loc[start:end])

        drawdown_periods.append({
            "start": start,
            "trough": trough_date,
            "end": end,
            "depth": trough_value,
            "duration_months": duration,
        })

    df = pd.DataFrame(drawdown_periods)

    if len(df) > 0:
        # Sort by depth (worst first)
        df = df.sort_values("depth").reset_index(drop=True)

    return df
