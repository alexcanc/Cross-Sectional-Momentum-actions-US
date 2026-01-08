"""
Performance metrics module.

Computes standard risk-adjusted performance metrics for strategy evaluation.
All metrics follow industry standards and academic conventions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass


# Annualization factor for monthly data
MONTHS_PER_YEAR = 12
TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Returns
    total_return: float
    cagr: float

    # Risk
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # in months

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Other
    avg_turnover: float
    total_transaction_costs: float
    hit_rate: float  # % of positive months
    best_month: float
    worst_month: float
    skewness: float
    kurtosis: float


def compute_cagr(cumulative_returns: pd.Series) -> float:
    """
    Compute Compound Annual Growth Rate.

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative returns (growth of $1).

    Returns
    -------
    float
        CAGR as a decimal (e.g., 0.10 for 10%).
    """
    n_years = len(cumulative_returns) / MONTHS_PER_YEAR
    if n_years <= 0:
        return 0.0

    final_value = cumulative_returns.iloc[-1]
    initial_value = 1.0  # Assuming we start with $1

    cagr = (final_value / initial_value) ** (1 / n_years) - 1

    return cagr


def compute_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Compute annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    annualize : bool, default True
        Whether to annualize the volatility.

    Returns
    -------
    float
        Volatility as a decimal.
    """
    vol = returns.std()

    if annualize:
        vol = vol * np.sqrt(MONTHS_PER_YEAR)

    return vol


def compute_downside_volatility(
    returns: pd.Series,
    threshold: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Compute downside (semi-) volatility.

    Only considers returns below the threshold (typically 0).

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    threshold : float, default 0.0
        Return threshold for downside.
    annualize : bool, default True
        Whether to annualize.

    Returns
    -------
    float
        Downside volatility as a decimal.
    """
    downside_returns = returns[returns < threshold]

    if len(downside_returns) == 0:
        return 0.0

    downside_vol = np.sqrt((downside_returns ** 2).mean())

    if annualize:
        downside_vol = downside_vol * np.sqrt(MONTHS_PER_YEAR)

    return downside_vol


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.

    Returns
    -------
    float
        Sharpe ratio.
    """
    # Convert annual risk-free to monthly
    monthly_rf = (1 + risk_free_rate) ** (1 / MONTHS_PER_YEAR) - 1

    excess_returns = returns - monthly_rf
    avg_excess = excess_returns.mean()
    vol = excess_returns.std()

    if vol == 0:
        return 0.0

    # Annualize
    sharpe = (avg_excess / vol) * np.sqrt(MONTHS_PER_YEAR)

    return sharpe


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.
    target_return : float, default 0.0
        Target return (MAR) for downside deviation.

    Returns
    -------
    float
        Sortino ratio.
    """
    monthly_rf = (1 + risk_free_rate) ** (1 / MONTHS_PER_YEAR) - 1

    excess_returns = returns - monthly_rf
    avg_excess = excess_returns.mean() * MONTHS_PER_YEAR

    downside_vol = compute_downside_volatility(returns, target_return)

    if downside_vol == 0:
        return np.inf if avg_excess > 0 else 0.0

    sortino = avg_excess / downside_vol

    return sortino


def compute_max_drawdown(cumulative_returns: pd.Series) -> tuple:
    """
    Compute maximum drawdown.

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative returns (growth of $1).

    Returns
    -------
    tuple
        (max_drawdown, drawdown_start, drawdown_trough, drawdown_end)
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max

    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()

    # Find peak before trough
    peak_date = cumulative_returns.loc[:trough_date].idxmax()

    # Find recovery (if any)
    post_trough = cumulative_returns.loc[trough_date:]
    recovered = post_trough[post_trough >= cumulative_returns.loc[peak_date]]

    if len(recovered) > 0:
        recovery_date = recovered.index[0]
    else:
        recovery_date = cumulative_returns.index[-1]

    return max_dd, peak_date, trough_date, recovery_date


def compute_calmar_ratio(
    returns: pd.Series,
    cumulative_returns: pd.Series,
) -> float:
    """
    Compute Calmar ratio (CAGR / Max Drawdown).

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    cumulative_returns : pd.Series
        Cumulative returns.

    Returns
    -------
    float
        Calmar ratio.
    """
    cagr = compute_cagr(cumulative_returns)
    max_dd, _, _, _ = compute_max_drawdown(cumulative_returns)

    if max_dd == 0:
        return np.inf if cagr > 0 else 0.0

    # max_dd is negative, so we take absolute value
    calmar = cagr / abs(max_dd)

    return calmar


def compute_metrics(
    returns: pd.Series,
    cumulative_returns: pd.Series,
    turnover: Optional[pd.Series] = None,
    transaction_costs: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute all performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    cumulative_returns : pd.Series
        Cumulative returns.
    turnover : pd.Series, optional
        Monthly turnover.
    transaction_costs : pd.Series, optional
        Monthly transaction costs.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.

    Returns
    -------
    PerformanceMetrics
        All computed metrics.
    """
    # Basic return stats
    total_return = cumulative_returns.iloc[-1] - 1
    cagr = compute_cagr(cumulative_returns)

    # Risk metrics
    volatility = compute_volatility(returns)
    downside_volatility = compute_downside_volatility(returns)
    max_dd, peak, trough, recovery = compute_max_drawdown(cumulative_returns)

    # Duration of max drawdown (in months)
    dd_duration = len(cumulative_returns.loc[peak:recovery])

    # Risk-adjusted metrics
    sharpe = compute_sharpe_ratio(returns, risk_free_rate)
    sortino = compute_sortino_ratio(returns, risk_free_rate)
    calmar = compute_calmar_ratio(returns, cumulative_returns)

    # Turnover and costs
    avg_turnover = turnover.mean() if turnover is not None else 0.0
    total_costs = transaction_costs.sum() if transaction_costs is not None else 0.0

    # Other statistics
    hit_rate = (returns > 0).sum() / len(returns)
    best_month = returns.max()
    worst_month = returns.min()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        downside_volatility=downside_volatility,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        avg_turnover=avg_turnover,
        total_transaction_costs=total_costs,
        hit_rate=hit_rate,
        best_month=best_month,
        worst_month=worst_month,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def compute_annual_returns(returns: pd.Series) -> pd.Series:
    """
    Compute annual returns from monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns with DatetimeIndex.

    Returns
    -------
    pd.Series
        Annual returns indexed by year.
    """
    # Group by year and compound
    annual = (1 + returns).groupby(returns.index.year).prod() - 1

    return annual


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 12,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    window : int, default 12
        Rolling window in months.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.
    """
    monthly_rf = (1 + risk_free_rate) ** (1 / MONTHS_PER_YEAR) - 1
    excess = returns - monthly_rf

    rolling_mean = excess.rolling(window=window).mean()
    rolling_std = excess.rolling(window=window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(MONTHS_PER_YEAR)

    return rolling_sharpe


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 12,
) -> pd.Series:
    """
    Compute rolling annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    window : int, default 12
        Rolling window in months.

    Returns
    -------
    pd.Series
        Rolling volatility.
    """
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(MONTHS_PER_YEAR)

    return rolling_vol


def format_metrics_table(
    metrics: PerformanceMetrics,
    benchmark_metrics: Optional[PerformanceMetrics] = None,
) -> pd.DataFrame:
    """
    Format metrics as a comparison table.

    Parameters
    ----------
    metrics : PerformanceMetrics
        Strategy metrics.
    benchmark_metrics : PerformanceMetrics, optional
        Benchmark metrics for comparison.

    Returns
    -------
    pd.DataFrame
        Formatted metrics table.
    """
    data = {
        "Metric": [
            "Total Return",
            "CAGR",
            "Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Hit Rate",
            "Best Month",
            "Worst Month",
            "Skewness",
            "Kurtosis",
            "Avg Turnover",
            "Total Costs",
        ],
        "Strategy": [
            f"{metrics.total_return:.2%}",
            f"{metrics.cagr:.2%}",
            f"{metrics.volatility:.2%}",
            f"{metrics.sharpe_ratio:.2f}",
            f"{metrics.sortino_ratio:.2f}",
            f"{metrics.max_drawdown:.2%}",
            f"{metrics.calmar_ratio:.2f}",
            f"{metrics.hit_rate:.1%}",
            f"{metrics.best_month:.2%}",
            f"{metrics.worst_month:.2%}",
            f"{metrics.skewness:.2f}",
            f"{metrics.kurtosis:.2f}",
            f"{metrics.avg_turnover:.2%}",
            f"{metrics.total_transaction_costs:.2%}",
        ],
    }

    if benchmark_metrics is not None:
        data["Benchmark"] = [
            f"{benchmark_metrics.total_return:.2%}",
            f"{benchmark_metrics.cagr:.2%}",
            f"{benchmark_metrics.volatility:.2%}",
            f"{benchmark_metrics.sharpe_ratio:.2f}",
            f"{benchmark_metrics.sortino_ratio:.2f}",
            f"{benchmark_metrics.max_drawdown:.2%}",
            f"{benchmark_metrics.calmar_ratio:.2f}",
            f"{benchmark_metrics.hit_rate:.1%}",
            f"{benchmark_metrics.best_month:.2%}",
            f"{benchmark_metrics.worst_month:.2%}",
            f"{benchmark_metrics.skewness:.2f}",
            f"{benchmark_metrics.kurtosis:.2f}",
            "N/A",
            "N/A",
        ]

    return pd.DataFrame(data)
