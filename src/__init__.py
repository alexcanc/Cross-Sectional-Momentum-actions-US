"""
Cross-Sectional Momentum Strategy - US Equities

A quantitative research project implementing the classic 12-1 momentum strategy
on Dow Jones 30 constituents.
"""

from .config import Config
from .data_loader import load_universe_data, load_benchmark_data
from .signal import compute_momentum_signal, rank_momentum
from .backtest import run_backtest, BacktestResult
from .metrics import compute_metrics, compute_annual_returns
from .regime_filter import compute_regime_filter, apply_regime_filter
from .ml_regime import compute_ml_regime_signal, MLRegimeModel, get_feature_importance
from .walk_forward import run_walk_forward, WalkForwardAnalysis

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "Config",
    "load_universe_data",
    "load_benchmark_data",
    "compute_momentum_signal",
    "rank_momentum",
    "run_backtest",
    "BacktestResult",
    "compute_metrics",
    "compute_annual_returns",
    "compute_regime_filter",
    "apply_regime_filter",
]
