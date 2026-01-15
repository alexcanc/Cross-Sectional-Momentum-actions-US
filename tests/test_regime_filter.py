"""
Unit tests for src/regime_filter.py

Tests market regime detection using SMA-200 filter.
"""

import pytest
import pandas as pd
import numpy as np

from src.regime_filter import (
    compute_sma,
    compute_regime_sma,
    resample_regime_to_monthly,
    compute_regime_filter,
    apply_regime_filter,
)
from src.config import Config


class TestComputeSMA:
    """Test suite for SMA computation."""

    def test_sma_window_size(self, sample_benchmark_daily):
        """Test that SMA respects window size."""
        sma = compute_sma(sample_benchmark_daily.iloc[:, 0], window=20)

        # First 19 values should be NaN
        assert sma.iloc[:19].isna().all()

        # From index 19 onwards, should have values
        assert sma.iloc[19:].notna().all()

    def test_sma_calculation(self):
        """Test SMA calculation with known values."""
        dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
        prices = pd.Series([10, 20, 30, 40, 50], index=dates)

        sma = compute_sma(prices, window=3)

        # SMA at index 2: (10+20+30)/3 = 20
        assert abs(sma.iloc[2] - 20) < 0.01

        # SMA at index 3: (20+30+40)/3 = 30
        assert abs(sma.iloc[3] - 30) < 0.01

        # SMA at index 4: (30+40+50)/3 = 40
        assert abs(sma.iloc[4] - 40) < 0.01

    def test_sma_200_default(self, sample_benchmark_daily):
        """Test default SMA-200 window."""
        sma = compute_sma(sample_benchmark_daily.iloc[:, 0], window=200)

        # First 199 values should be NaN
        assert sma.iloc[:199].isna().all()

        # Should have values after window
        assert sma.iloc[199:].notna().all()


class TestComputeRegimeSMA:
    """Test suite for SMA-based regime detection."""

    def test_regime_binary_values(self, sample_benchmark_daily):
        """Test that regime signal is binary (0 or 1)."""
        regime = compute_regime_sma(sample_benchmark_daily, sma_window=50)

        valid_regime = regime.dropna()
        assert set(valid_regime.unique()).issubset({0, 1})

    def test_regime_risk_on_when_above_sma(self):
        """Test risk-on (1) when price > SMA."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

        # Price always above its SMA (strong uptrend)
        prices = pd.Series([100 + i for i in range(100)], index=dates)
        df = pd.DataFrame({'SPY': prices})

        regime = compute_regime_sma(df, sma_window=20)

        # After warmup, should be risk-on (price always rising)
        valid_regime = regime.iloc[20:]
        assert (valid_regime == 1).all()

    def test_regime_risk_off_when_below_sma(self):
        """Test risk-off (0) when price < SMA."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

        # Price always below its SMA (strong downtrend)
        prices = pd.Series([200 - i for i in range(100)], index=dates)
        df = pd.DataFrame({'SPY': prices})

        regime = compute_regime_sma(df, sma_window=20)

        # After warmup and trend establishes, should be mostly risk-off
        valid_regime = regime.iloc[40:]
        assert (valid_regime == 0).mean() > 0.8


class TestResampleRegimeToMonthly:
    """Test suite for regime resampling."""

    def test_resample_last(self):
        """Test that 'last' method takes end-of-month value."""
        dates = pd.date_range(start='2020-01-01', periods=60, freq='D')

        # Alternating regime
        regime_daily = pd.Series([i % 2 for i in range(60)], index=dates)

        regime_monthly = resample_regime_to_monthly(regime_daily, method='last')

        # Should have monthly frequency
        assert len(regime_monthly) <= 3

    def test_resample_majority(self):
        """Test that 'majority' method takes most common value."""
        dates = pd.date_range(start='2020-01-01', periods=31, freq='D')

        # Mostly 1s with a few 0s
        regime_daily = pd.Series([1] * 25 + [0] * 6, index=dates)

        regime_monthly = resample_regime_to_monthly(regime_daily, method='majority')

        # Majority is 1
        assert regime_monthly.iloc[0] == 1

    def test_resample_invalid_method(self, sample_benchmark_daily):
        """Test that invalid method raises error."""
        regime_daily = compute_regime_sma(sample_benchmark_daily, sma_window=50)

        with pytest.raises(ValueError):
            resample_regime_to_monthly(regime_daily, method='invalid')


class TestComputeRegimeFilter:
    """Test suite for main regime filter function."""

    def test_regime_filter_returns_monthly(self, sample_benchmark_daily):
        """Test that regime filter returns monthly data."""
        regime = compute_regime_filter(sample_benchmark_daily)

        # Check it's monthly frequency (ME = month end)
        assert regime.index.freq is None or 'M' in str(regime.index.freq)

    def test_regime_filter_binary_output(self, sample_benchmark_daily):
        """Test that output is binary."""
        regime = compute_regime_filter(sample_benchmark_daily)

        assert set(regime.unique()).issubset({0, 1})

    def test_regime_filter_with_config(self, sample_benchmark_daily):
        """Test regime filter uses config parameters."""
        config = Config()
        config.REGIME_MA_PERIOD = 50  # Shorter window for test data

        regime = compute_regime_filter(sample_benchmark_daily, config)

        # Should produce valid output
        assert len(regime) > 0
        assert regime.notna().any()


class TestApplyRegimeFilter:
    """Test suite for applying regime filter to weights."""

    def test_apply_regime_reduces_exposure(self):
        """Test that risk-off regime reduces exposure."""
        dates = pd.date_range(start='2020-01-31', periods=4, freq='ME')

        weights = pd.DataFrame({
            'A': [0.5, 0.5, 0.5, 0.5],
            'B': [0.5, 0.5, 0.5, 0.5],
        }, index=dates)

        regime = pd.Series([1, 0, 1, 0], index=dates)

        adjusted = apply_regime_filter(weights, regime, risk_off_exposure=0.5)

        # Risk-on periods: full weights
        assert adjusted.loc[dates[0]].sum() == 1.0
        assert adjusted.loc[dates[2]].sum() == 1.0

        # Risk-off periods: 50% exposure
        assert abs(adjusted.loc[dates[1]].sum() - 0.5) < 0.01
        assert abs(adjusted.loc[dates[3]].sum() - 0.5) < 0.01

    def test_apply_regime_full_risk_off(self):
        """Test 0% exposure during complete risk-off."""
        dates = pd.date_range(start='2020-01-31', periods=2, freq='ME')

        weights = pd.DataFrame({
            'A': [0.5, 0.5],
            'B': [0.5, 0.5],
        }, index=dates)

        regime = pd.Series([1, 0], index=dates)

        adjusted = apply_regime_filter(weights, regime, risk_off_exposure=0.0)

        # Risk-off with 0 exposure = all cash
        assert adjusted.loc[dates[1]].sum() == 0.0

    def test_apply_regime_preserves_structure(self):
        """Test that weight structure is preserved."""
        dates = pd.date_range(start='2020-01-31', periods=2, freq='ME')

        weights = pd.DataFrame({
            'A': [0.6, 0.7],
            'B': [0.4, 0.3],
        }, index=dates)

        regime = pd.Series([1, 0], index=dates)

        adjusted = apply_regime_filter(weights, regime, risk_off_exposure=0.5)

        # Risk-off: weights scaled proportionally
        # A: 0.7 * 0.5 = 0.35, B: 0.3 * 0.5 = 0.15
        assert abs(adjusted.loc[dates[1], 'A'] - 0.35) < 0.01
        assert abs(adjusted.loc[dates[1], 'B'] - 0.15) < 0.01

    def test_apply_regime_handles_missing_dates(self):
        """Test that missing regime dates are filled with 1 (risk-on)."""
        dates = pd.date_range(start='2020-01-31', periods=4, freq='ME')

        weights = pd.DataFrame({
            'A': [0.5, 0.5, 0.5, 0.5],
            'B': [0.5, 0.5, 0.5, 0.5],
        }, index=dates)

        # Regime only covers first 2 dates
        regime = pd.Series([1, 0], index=dates[:2])

        adjusted = apply_regime_filter(weights, regime, risk_off_exposure=0.5)

        # Missing dates should default to risk-on (full exposure)
        assert adjusted.loc[dates[2]].sum() == 1.0
        assert adjusted.loc[dates[3]].sum() == 1.0
