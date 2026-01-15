"""
Unit tests for src/config.py

Tests configuration initialization and parameter validation.
"""

import pytest
from src.config import Config, get_config


class TestConfig:
    """Test suite for Config dataclass."""

    def test_config_initialization(self):
        """Test that Config initializes with correct default values."""
        config = Config()

        assert config.MOMENTUM_LOOKBACK == 12
        assert config.MOMENTUM_SKIP == 1
        assert config.N_LONG == 10
        assert config.N_SHORT == 10
        assert config.PORTFOLIO_TYPE == "long_short"
        assert config.TRANSACTION_COST_BPS == 10.0

    def test_dow_jones_30_initialization(self):
        """Test that DOW_JONES_30 list is properly initialized."""
        config = Config()

        assert config.DOW_JONES_30 is not None
        assert len(config.DOW_JONES_30) == 30
        assert "AAPL" in config.DOW_JONES_30
        assert "MSFT" in config.DOW_JONES_30
        assert "JPM" in config.DOW_JONES_30

    def test_transaction_cost_property(self):
        """Test transaction cost conversion from bps to decimal."""
        config = Config()

        # 10 bps = 0.001
        assert config.transaction_cost == 0.001

        # Test with different value
        config.TRANSACTION_COST_BPS = 20.0
        assert config.transaction_cost == 0.002

    def test_get_config_returns_new_instance(self):
        """Test that get_config returns a fresh Config instance."""
        config1 = get_config()
        config2 = get_config()

        # Should be different instances
        assert config1 is not config2

        # But with same default values
        assert config1.MOMENTUM_LOOKBACK == config2.MOMENTUM_LOOKBACK

    def test_config_modification(self):
        """Test that config parameters can be modified."""
        config = Config()

        config.MOMENTUM_LOOKBACK = 6
        config.N_LONG = 5
        config.PORTFOLIO_TYPE = "long_only"

        assert config.MOMENTUM_LOOKBACK == 6
        assert config.N_LONG == 5
        assert config.PORTFOLIO_TYPE == "long_only"

    def test_regime_filter_defaults(self):
        """Test regime filter default parameters."""
        config = Config()

        assert config.REGIME_MA_PERIOD == 200
        assert config.RISK_OFF_EXPOSURE == 0.5

    def test_date_range_defaults(self):
        """Test default date range for backtest."""
        config = Config()

        assert config.START_DATE == "2005-01-01"
        assert config.END_DATE == "2024-12-31"

    def test_benchmark_ticker_default(self):
        """Test benchmark ticker default value."""
        config = Config()

        assert config.BENCHMARK_TICKER == "SPY"
