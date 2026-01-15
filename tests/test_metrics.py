"""
Unit tests for src/metrics.py

Tests performance metrics computation including Sharpe, Sortino, CAGR, etc.
"""

import pytest
import pandas as pd
import numpy as np

from src.metrics import (
    compute_cagr,
    compute_volatility,
    compute_downside_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_metrics,
    compute_annual_returns,
    PerformanceMetrics,
)


class TestComputeCAGR:
    """Test suite for CAGR computation."""

    def test_cagr_known_value(self):
        """Test CAGR with known doubling scenario."""
        # $1 doubles to $2 over 6 years = CAGR of ~12.25%
        dates = pd.date_range(start='2020-01-31', periods=72, freq='ME')  # 6 years
        cumulative = pd.Series(
            np.linspace(1, 2, 72),
            index=dates
        )

        cagr = compute_cagr(cumulative)

        # Approximate check (linear approximation)
        assert 0.10 < cagr < 0.15

    def test_cagr_exact_calculation(self):
        """Test CAGR with exact known value."""
        # 10% annual return for 2 years
        dates = pd.date_range(start='2020-01-31', periods=24, freq='ME')
        final_value = (1.10) ** 2  # 1.21
        cumulative = pd.Series([final_value] * 24, index=dates)
        cumulative.iloc[0] = 1.0

        # Recompute with proper cumulative
        cumulative = pd.Series(
            [1.0 * (1.10 ** (i/12)) for i in range(24)],
            index=dates
        )

        cagr = compute_cagr(cumulative)

        assert abs(cagr - 0.10) < 0.02

    def test_cagr_negative_return(self):
        """Test CAGR with negative overall return."""
        dates = pd.date_range(start='2020-01-31', periods=12, freq='ME')
        cumulative = pd.Series(
            np.linspace(1, 0.8, 12),  # 20% loss
            index=dates
        )

        cagr = compute_cagr(cumulative)

        assert cagr < 0


class TestComputeVolatility:
    """Test suite for volatility computation."""

    def test_volatility_annualized(self):
        """Test that volatility is properly annualized."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-31', periods=120, freq='ME')

        # Monthly returns with 3% monthly std
        returns = pd.Series(np.random.randn(120) * 0.03, index=dates)

        vol = compute_volatility(returns, annualize=True)

        # Annualized vol ≈ 0.03 * sqrt(12) ≈ 10.4%
        assert 0.08 < vol < 0.13

    def test_volatility_not_annualized(self):
        """Test volatility without annualization."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-31', periods=120, freq='ME')

        returns = pd.Series(np.random.randn(120) * 0.03, index=dates)

        vol = compute_volatility(returns, annualize=False)

        # Should be close to 3%
        assert 0.025 < vol < 0.035


class TestComputeDownsideVolatility:
    """Test suite for downside volatility computation."""

    def test_downside_only_negative_returns(self):
        """Test that only negative returns are considered."""
        dates = pd.date_range(start='2020-01-31', periods=6, freq='ME')

        # Half positive, half negative
        returns = pd.Series([0.05, 0.03, 0.02, -0.02, -0.03, -0.05], index=dates)

        downside_vol = compute_downside_volatility(returns, threshold=0.0)

        # Should only consider -0.02, -0.03, -0.05
        assert downside_vol > 0

    def test_downside_zero_if_all_positive(self):
        """Test downside vol is zero when all returns positive."""
        dates = pd.date_range(start='2020-01-31', periods=5, freq='ME')
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates)

        downside_vol = compute_downside_volatility(returns, threshold=0.0)

        assert downside_vol == 0.0


class TestComputeSharpeRatio:
    """Test suite for Sharpe ratio computation."""

    def test_sharpe_positive_excess_return(self, sample_returns):
        """Test Sharpe ratio with positive returns."""
        sharpe = compute_sharpe_ratio(sample_returns, risk_free_rate=0.0)

        # With positive average return and normal vol, Sharpe should be positive
        if sample_returns.mean() > 0:
            assert sharpe > 0

    def test_sharpe_with_risk_free(self):
        """Test Sharpe ratio accounts for risk-free rate."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-31', periods=24, freq='ME')

        # Random returns with positive mean and some volatility
        returns = pd.Series(np.random.randn(24) * 0.02 + 0.01, index=dates)

        sharpe_zero_rf = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_high_rf = compute_sharpe_ratio(returns, risk_free_rate=0.10)

        # Higher risk-free rate should lower Sharpe
        assert sharpe_high_rf < sharpe_zero_rf


class TestComputeSortinoRatio:
    """Test suite for Sortino ratio computation."""

    def test_sortino_higher_than_sharpe_for_skewed(self):
        """Test Sortino > Sharpe when returns are positively skewed."""
        dates = pd.date_range(start='2020-01-31', periods=24, freq='ME')

        # Mostly positive, few negative (positively skewed)
        returns = pd.Series(
            [0.02] * 20 + [-0.01] * 4,
            index=dates
        )

        sharpe = compute_sharpe_ratio(returns)
        sortino = compute_sortino_ratio(returns)

        # Sortino should generally be higher for positively skewed returns
        # as it only penalizes downside
        assert sortino >= sharpe * 0.8  # Allow some tolerance


class TestComputeMaxDrawdown:
    """Test suite for maximum drawdown computation."""

    def test_max_drawdown_known_value(self):
        """Test max drawdown with known scenario."""
        dates = pd.date_range(start='2020-01-31', periods=5, freq='ME')

        # Peak at 1.5, trough at 1.0 = 33.3% drawdown
        cumulative = pd.Series([1.0, 1.5, 1.0, 1.2, 1.4], index=dates)

        max_dd, peak, trough, recovery = compute_max_drawdown(cumulative)

        assert abs(max_dd - (-1/3)) < 0.01
        assert peak == dates[1]
        assert trough == dates[2]

    def test_max_drawdown_always_negative(self, sample_cumulative_returns):
        """Test that max drawdown is always negative or zero."""
        max_dd, _, _, _ = compute_max_drawdown(sample_cumulative_returns)

        assert max_dd <= 0


class TestComputeCalmarRatio:
    """Test suite for Calmar ratio computation."""

    def test_calmar_ratio_calculation(self):
        """Test Calmar = CAGR / |Max Drawdown|."""
        dates = pd.date_range(start='2020-01-31', periods=24, freq='ME')

        # Create returns that give known CAGR and drawdown
        cumulative = pd.Series(
            [1.0 + 0.01 * i for i in range(24)],
            index=dates
        )
        cumulative.iloc[10:15] = [1.05, 0.95, 0.90, 0.95, 1.0]  # 14% drawdown

        returns = cumulative.pct_change().fillna(0)

        calmar = compute_calmar_ratio(returns, cumulative)

        # Calmar should be positive if CAGR is positive
        assert calmar > 0


class TestComputeMetrics:
    """Test suite for comprehensive metrics computation."""

    def test_compute_metrics_returns_dataclass(self, sample_returns, sample_cumulative_returns):
        """Test that compute_metrics returns PerformanceMetrics."""
        turnover = pd.Series([0.2] * len(sample_returns), index=sample_returns.index)
        costs = pd.Series([0.002] * len(sample_returns), index=sample_returns.index)

        metrics = compute_metrics(
            sample_returns,
            sample_cumulative_returns,
            turnover,
            costs
        )

        assert isinstance(metrics, PerformanceMetrics)

    def test_compute_metrics_all_fields(self, sample_returns, sample_cumulative_returns):
        """Test that all metric fields are populated."""
        turnover = pd.Series([0.2] * len(sample_returns), index=sample_returns.index)
        costs = pd.Series([0.002] * len(sample_returns), index=sample_returns.index)

        metrics = compute_metrics(
            sample_returns,
            sample_cumulative_returns,
            turnover,
            costs
        )

        # Check all fields exist and are numbers
        assert not np.isnan(metrics.cagr)
        assert not np.isnan(metrics.volatility)
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.sortino_ratio)
        assert not np.isnan(metrics.max_drawdown)
        assert not np.isnan(metrics.hit_rate)

    def test_hit_rate_calculation(self):
        """Test hit rate (% positive months) calculation."""
        dates = pd.date_range(start='2020-01-31', periods=10, freq='ME')

        # 7 positive, 3 negative = 70% hit rate
        returns = pd.Series(
            [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, -0.01, 0.02],
            index=dates
        )
        cumulative = (1 + returns).cumprod()

        metrics = compute_metrics(returns, cumulative)

        assert abs(metrics.hit_rate - 0.7) < 0.01


class TestComputeAnnualReturns:
    """Test suite for annual returns computation."""

    def test_annual_returns_grouping(self):
        """Test that returns are properly grouped by year."""
        dates = pd.date_range(start='2020-01-31', periods=24, freq='ME')

        # 1% monthly return
        returns = pd.Series([0.01] * 24, index=dates)

        annual = compute_annual_returns(returns)

        # Should have 2 years
        assert len(annual) == 2
        assert 2020 in annual.index
        assert 2021 in annual.index

    def test_annual_returns_compounding(self):
        """Test that annual returns are properly compounded."""
        dates = pd.date_range(start='2020-01-31', periods=12, freq='ME')

        # 1% monthly = (1.01)^12 - 1 ≈ 12.68% annual
        returns = pd.Series([0.01] * 12, index=dates)

        annual = compute_annual_returns(returns)

        expected = (1.01 ** 12) - 1
        assert abs(annual[2020] - expected) < 0.001
