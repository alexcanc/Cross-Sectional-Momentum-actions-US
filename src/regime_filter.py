"""
Market Regime Filter module.

Implements market regime detection to dynamically adjust strategy exposure.
Two approaches:
1. SMA200 baseline: Simple moving average trend filter
2. ML-based (optional): Logistic regression classifier

The goal is to reduce exposure during adverse market conditions,
improving risk-adjusted returns without excessive complexity.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import Config, get_config


def compute_sma(prices: pd.Series, window: int = 200) -> pd.Series:
    """
    Compute Simple Moving Average.

    Parameters
    ----------
    prices : pd.Series
        Daily price series.
    window : int, default 200
        Rolling window size in days.

    Returns
    -------
    pd.Series
        SMA series.
    """
    return prices.rolling(window=window, min_periods=window).mean()


def compute_regime_sma(
    benchmark_daily: pd.DataFrame,
    sma_window: int = 200,
    config: Optional[Config] = None,
) -> pd.Series:
    """
    Compute market regime using SMA200 filter.

    Regime = 1 (Risk-On) if price > SMA200
    Regime = 0 (Risk-Off) if price < SMA200

    Parameters
    ----------
    benchmark_daily : pd.DataFrame
        Daily benchmark prices (e.g., SPY).
    sma_window : int, default 200
        SMA window in days.
    config : Config, optional
        Strategy configuration.

    Returns
    -------
    pd.Series
        Binary regime signal (1 = risk-on, 0 = risk-off).

    Notes
    -----
    This is the baseline approach, following Faber (2007).
    Simple, transparent, and historically effective.
    """
    if config is None:
        config = get_config()

    # Get price series
    if isinstance(benchmark_daily, pd.DataFrame):
        prices = benchmark_daily.iloc[:, 0]
    else:
        prices = benchmark_daily

    # Compute SMA
    sma = compute_sma(prices, window=sma_window)

    # Regime: 1 if price > SMA, 0 otherwise
    regime = (prices > sma).astype(int)

    return regime


def resample_regime_to_monthly(
    regime_daily: pd.Series,
    method: str = "last",
) -> pd.Series:
    """
    Resample daily regime signal to monthly frequency.

    Parameters
    ----------
    regime_daily : pd.Series
        Daily regime signal.
    method : str, default 'last'
        Resampling method: 'last' (end of month) or 'majority' (most common).

    Returns
    -------
    pd.Series
        Monthly regime signal.

    Notes
    -----
    Using 'last' means the regime at month-end determines next month's exposure.
    This is look-ahead bias free: we know the regime at month-end before trading.
    """
    if method == "last":
        return regime_daily.resample("ME").last()
    elif method == "majority":
        # Most common regime during the month
        return regime_daily.resample("ME").apply(
            lambda x: 1 if x.mean() > 0.5 else 0
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_regime_filter(
    benchmark_daily: pd.DataFrame,
    config: Optional[Config] = None,
) -> pd.Series:
    """
    Main function to compute regime filter signal.

    Parameters
    ----------
    benchmark_daily : pd.DataFrame
        Daily benchmark prices.
    config : Config, optional
        Strategy configuration.

    Returns
    -------
    pd.Series
        Monthly regime signal (1 = risk-on, 0 = risk-off).
    """
    if config is None:
        config = get_config()

    # Compute daily regime using SMA200
    regime_daily = compute_regime_sma(
        benchmark_daily,
        sma_window=config.REGIME_MA_PERIOD,
        config=config,
    )

    # Resample to monthly
    regime_monthly = resample_regime_to_monthly(regime_daily, method="last")

    return regime_monthly


def apply_regime_filter(
    weights: pd.DataFrame,
    regime: pd.Series,
    risk_off_exposure: float = 0.5,
) -> pd.DataFrame:
    """
    Apply regime filter to portfolio weights.

    Parameters
    ----------
    weights : pd.DataFrame
        Original portfolio weights.
    regime : pd.Series
        Regime signal (1 = risk-on, 0 = risk-off).
    risk_off_exposure : float, default 0.5
        Exposure multiplier during risk-off periods.

    Returns
    -------
    pd.DataFrame
        Adjusted portfolio weights.
    """
    # Align regime to weights index
    regime_aligned = regime.reindex(weights.index).fillna(1)

    # Create exposure multiplier
    # Risk-on: 1.0 (full exposure)
    # Risk-off: risk_off_exposure (reduced exposure)
    exposure = regime_aligned.replace({1: 1.0, 0: risk_off_exposure})

    # Apply to weights
    adjusted_weights = weights.mul(exposure, axis=0)

    return adjusted_weights


# =============================================================================
# OPTIONAL: ML-BASED REGIME FILTER
# =============================================================================

def compute_regime_features(
    benchmark_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute features for ML-based regime classification.

    Features are designed to capture:
    - Trend (momentum of benchmark)
    - Volatility (realized vol)
    - Regime persistence (recent regime states)

    Parameters
    ----------
    benchmark_daily : pd.DataFrame
        Daily benchmark prices.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with daily frequency.

    Notes
    -----
    All features use only past information (no look-ahead bias).
    Features are simple and interpretable - no black box.
    """
    if isinstance(benchmark_daily, pd.DataFrame):
        prices = benchmark_daily.iloc[:, 0]
    else:
        prices = benchmark_daily

    returns = prices.pct_change(fill_method=None)

    features = pd.DataFrame(index=prices.index)

    # Feature 1: 20-day momentum (short-term trend)
    features["mom_20d"] = prices / prices.shift(20) - 1

    # Feature 2: 60-day momentum (medium-term trend)
    features["mom_60d"] = prices / prices.shift(60) - 1

    # Feature 3: 20-day realized volatility
    features["vol_20d"] = returns.rolling(20).std() * np.sqrt(252)

    # Feature 4: 60-day realized volatility
    features["vol_60d"] = returns.rolling(60).std() * np.sqrt(252)

    # Feature 5: Price vs SMA50
    features["price_vs_sma50"] = prices / prices.rolling(50).mean() - 1

    # Feature 6: Price vs SMA200
    features["price_vs_sma200"] = prices / prices.rolling(200).mean() - 1

    # Feature 7: Volatility ratio (short/long)
    features["vol_ratio"] = features["vol_20d"] / features["vol_60d"]

    # Feature 8: RSI-like momentum (normalized)
    gains = returns.clip(lower=0).rolling(14).sum()
    losses = (-returns.clip(upper=0)).rolling(14).sum()
    features["rsi_14d"] = gains / (gains + losses)

    return features.dropna()


def create_regime_labels(
    benchmark_daily: pd.DataFrame,
    forward_window: int = 21,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Create regime labels based on forward returns.

    This is used for training the ML model.

    Parameters
    ----------
    benchmark_daily : pd.DataFrame
        Daily benchmark prices.
    forward_window : int, default 21
        Days to look forward for return.
    threshold : float, default 0.0
        Return threshold for positive regime.

    Returns
    -------
    pd.Series
        Binary labels (1 = positive forward return, 0 = negative).

    Notes
    -----
    IMPORTANT: These labels use future information and are ONLY for training.
    In production, the model predicts without knowing future returns.
    """
    if isinstance(benchmark_daily, pd.DataFrame):
        prices = benchmark_daily.iloc[:, 0]
    else:
        prices = benchmark_daily

    # Forward return
    forward_return = prices.shift(-forward_window) / prices - 1

    # Binary label
    labels = (forward_return > threshold).astype(int)

    return labels


def train_regime_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    train_end_date: str,
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train logistic regression classifier for regime detection.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame.
    labels : pd.Series
        Binary regime labels.
    train_end_date : str
        End date for training data (YYYY-MM-DD).

    Returns
    -------
    Tuple[LogisticRegression, StandardScaler]
        Trained model and scaler.

    Notes
    -----
    We use a simple logistic regression for interpretability.
    The model coefficients show which features matter most.
    """
    # Align features and labels
    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    # Split by date (not random - time series!)
    train_mask = X.index <= train_end_date
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_regime_ml(
    features: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
) -> pd.Series:
    """
    Predict regime using trained ML model.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame (can include unseen dates).
    model : LogisticRegression
        Trained classifier.
    scaler : StandardScaler
        Fitted scaler.

    Returns
    -------
    pd.Series
        Predicted regime probabilities.
    """
    X_scaled = scaler.transform(features)
    proba = model.predict_proba(X_scaled)[:, 1]

    return pd.Series(proba, index=features.index, name="regime_proba")


def compute_regime_ml(
    benchmark_daily: pd.DataFrame,
    train_end_date: str = "2015-12-31",
    probability_threshold: float = 0.5,
) -> Tuple[pd.Series, LogisticRegression, StandardScaler]:
    """
    Compute regime using ML approach.

    Parameters
    ----------
    benchmark_daily : pd.DataFrame
        Daily benchmark prices.
    train_end_date : str
        End of training period.
    probability_threshold : float, default 0.5
        Threshold for binary regime decision.

    Returns
    -------
    Tuple[pd.Series, LogisticRegression, StandardScaler]
        Regime signal, trained model, and scaler.

    Notes
    -----
    This is the optional ML enhancement.
    The baseline SMA200 approach is often sufficient.
    """
    # Compute features
    features = compute_regime_features(benchmark_daily)

    # Create labels
    labels = create_regime_labels(benchmark_daily)

    # Train model
    model, scaler = train_regime_classifier(features, labels, train_end_date)

    # Predict on all data
    regime_proba = predict_regime_ml(features, model, scaler)

    # Convert to binary
    regime_binary = (regime_proba > probability_threshold).astype(int)

    return regime_binary, model, scaler


def analyze_regime_performance(
    regime: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Analyze strategy performance by regime.

    Parameters
    ----------
    regime : pd.Series
        Binary regime signal.
    benchmark_returns : pd.Series
        Benchmark returns.

    Returns
    -------
    pd.DataFrame
        Performance statistics by regime.
    """
    # Align
    common_idx = regime.index.intersection(benchmark_returns.index)
    regime = regime.loc[common_idx]
    returns = benchmark_returns.loc[common_idx]

    # Split by regime
    risk_on_returns = returns[regime == 1]
    risk_off_returns = returns[regime == 0]

    stats = pd.DataFrame({
        "Risk-On": {
            "Count (months)": len(risk_on_returns),
            "Mean Return": risk_on_returns.mean(),
            "Volatility": risk_on_returns.std() * np.sqrt(12),
            "Sharpe": risk_on_returns.mean() / risk_on_returns.std() * np.sqrt(12),
            "Hit Rate": (risk_on_returns > 0).mean(),
        },
        "Risk-Off": {
            "Count (months)": len(risk_off_returns),
            "Mean Return": risk_off_returns.mean(),
            "Volatility": risk_off_returns.std() * np.sqrt(12),
            "Sharpe": risk_off_returns.mean() / risk_off_returns.std() * np.sqrt(12),
            "Hit Rate": (risk_off_returns > 0).mean(),
        },
    })

    return stats.T
