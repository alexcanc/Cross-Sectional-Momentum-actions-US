"""
Unit tests for src/signal.py

Tests momentum signal computation and portfolio weight calculation.
"""

import pytest
import pandas as pd
import numpy as np

from src.signal import (
    compute_momentum_signal,
    rank_momentum,
    get_long_short_positions,
    compute_portfolio_weights,
)


class TestMomentumSignal:
    """Test suite for momentum signal computation."""

    def test_momentum_signal_shape(self, sample_prices):
        """Test that momentum signal has correct shape."""
        momentum = compute_momentum_signal(sample_prices, lookback=12, skip=1)

        assert momentum.shape == sample_prices.shape
        assert list(momentum.columns) == list(sample_prices.columns)
        assert list(momentum.index) == list(sample_prices.index)

    def test_momentum_signal_nans(self, sample_prices):
        """Test that first lookback rows are NaN (no look-ahead bias)."""
        momentum = compute_momentum_signal(sample_prices, lookback=12, skip=1)

        # First 12 rows should be NaN
        assert momentum.iloc[:12].isna().all().all()

        # After lookback, should have values
        assert momentum.iloc[12:].notna().all().all()

    def test_momentum_signal_calculation(self):
        """Test momentum calculation with known values."""
        # Create simple test data
        dates = pd.date_range(start='2020-01-31', periods=14, freq='ME')
        prices = pd.DataFrame({
            'A': [100] * 11 + [120, 130, 140],  # Flat then up
            'B': [100] * 11 + [80, 70, 60],     # Flat then down
        }, index=dates)

        momentum = compute_momentum_signal(prices, lookback=12, skip=1)

        # Momentum = price[t-skip] / price[t-lookback] - 1
        # At index 12: price[11] / price[0] - 1
        # For A: 120 / 100 - 1 = 0.2
        # For B: 80 / 100 - 1 = -0.2
        assert abs(momentum.loc[dates[12], 'A'] - 0.2) < 0.001
        assert abs(momentum.loc[dates[12], 'B'] - (-0.2)) < 0.001

    def test_momentum_with_different_lookback(self, sample_prices):
        """Test momentum with different lookback periods."""
        mom_6 = compute_momentum_signal(sample_prices, lookback=6, skip=1)
        mom_12 = compute_momentum_signal(sample_prices, lookback=12, skip=1)

        # 6-month lookback should have fewer NaN rows
        assert mom_6.iloc[5].isna().all()
        assert mom_6.iloc[6].notna().all()

        assert mom_12.iloc[11].isna().all()
        assert mom_12.iloc[12].notna().all()


class TestRankMomentum:
    """Test suite for cross-sectional ranking."""

    def test_rank_shape(self, sample_momentum):
        """Test that ranks have correct shape."""
        ranks = rank_momentum(sample_momentum)

        assert ranks.shape == sample_momentum.shape

    def test_rank_values(self, sample_momentum):
        """Test that ranks are valid (1 to N)."""
        ranks = rank_momentum(sample_momentum)

        # Ranks should be between 1 and N (number of stocks)
        n_stocks = sample_momentum.shape[1]

        for idx in ranks.index:
            row_ranks = ranks.loc[idx].dropna()
            assert row_ranks.min() >= 1
            assert row_ranks.max() <= n_stocks

    def test_rank_ordering(self):
        """Test that higher momentum gets higher rank."""
        dates = pd.date_range(start='2020-01-31', periods=1, freq='ME')
        momentum = pd.DataFrame({
            'A': [0.3],
            'B': [0.1],
            'C': [-0.1],
        }, index=dates)

        ranks = rank_momentum(momentum)

        # A should have highest rank, C lowest
        assert ranks.loc[dates[0], 'A'] > ranks.loc[dates[0], 'B']
        assert ranks.loc[dates[0], 'B'] > ranks.loc[dates[0], 'C']


class TestLongShortPositions:
    """Test suite for position identification."""

    def test_long_positions_count(self, sample_momentum):
        """Test that correct number of long positions identified."""
        long_pos, short_pos = get_long_short_positions(
            sample_momentum, n_long=3, n_short=3
        )

        # Each row should have exactly 3 long positions
        for idx in long_pos.index:
            assert long_pos.loc[idx].sum() == 3

    def test_short_positions_count(self, sample_momentum):
        """Test that correct number of short positions identified."""
        long_pos, short_pos = get_long_short_positions(
            sample_momentum, n_long=3, n_short=3
        )

        # Each row should have exactly 3 short positions
        for idx in short_pos.index:
            assert short_pos.loc[idx].sum() == 3

    def test_no_overlap(self, sample_momentum):
        """Test that long and short positions don't overlap."""
        long_pos, short_pos = get_long_short_positions(
            sample_momentum, n_long=3, n_short=3
        )

        # No stock should be both long and short
        overlap = (long_pos & short_pos).sum().sum()
        assert overlap == 0

    def test_winners_are_long(self):
        """Test that top momentum stocks are selected for long."""
        dates = pd.date_range(start='2020-01-31', periods=1, freq='ME')
        momentum = pd.DataFrame({
            'A': [0.3],   # Winner
            'B': [0.2],   # Winner
            'C': [0.0],
            'D': [-0.2],  # Loser
            'E': [-0.3],  # Loser
        }, index=dates)

        long_pos, short_pos = get_long_short_positions(
            momentum, n_long=2, n_short=2
        )

        # A and B should be long
        assert long_pos.loc[dates[0], 'A'] == True
        assert long_pos.loc[dates[0], 'B'] == True

        # D and E should be short
        assert short_pos.loc[dates[0], 'D'] == True
        assert short_pos.loc[dates[0], 'E'] == True


class TestPortfolioWeights:
    """Test suite for portfolio weight computation."""

    def test_long_short_weights_sum_to_zero(self, sample_momentum):
        """Test that long/short weights sum to zero (dollar-neutral)."""
        weights = compute_portfolio_weights(
            sample_momentum,
            portfolio_type="long_short",
            n_long=3,
            n_short=3,
        )

        # Weights should sum to ~0 (dollar neutral)
        for idx in weights.index:
            assert abs(weights.loc[idx].sum()) < 0.001

    def test_long_only_weights_sum_to_one(self, sample_momentum):
        """Test that long-only weights sum to one."""
        weights = compute_portfolio_weights(
            sample_momentum,
            portfolio_type="long_only",
            n_long=3,
            n_short=0,
        )

        # Weights should sum to 1
        for idx in weights.index:
            assert abs(weights.loc[idx].sum() - 1.0) < 0.001

    def test_equal_weight_values(self, sample_momentum):
        """Test equal weight calculation."""
        weights = compute_portfolio_weights(
            sample_momentum,
            portfolio_type="long_short",
            n_long=5,
            n_short=5,
            weighting="equal_weight",
        )

        # Each long position should have weight 1/5 = 0.2
        # Each short position should have weight -1/5 = -0.2
        for idx in weights.index:
            row = weights.loc[idx]
            long_weights = row[row > 0]
            short_weights = row[row < 0]

            if len(long_weights) > 0:
                assert all(abs(w - 0.2) < 0.001 for w in long_weights)
            if len(short_weights) > 0:
                assert all(abs(w - (-0.2)) < 0.001 for w in short_weights)

    def test_weights_are_valid(self, sample_momentum):
        """Test that all weights are valid numbers."""
        weights = compute_portfolio_weights(
            sample_momentum,
            portfolio_type="long_short",
            n_long=3,
            n_short=3,
        )

        # No NaN or infinite values
        assert not weights.isna().any().any()
        assert not np.isinf(weights).any().any()
