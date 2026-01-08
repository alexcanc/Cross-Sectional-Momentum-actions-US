"""
Machine Learning Regime Filter

Utilise une Logistic Regression pour prédire les régimes de marché.
L'objectif est d'améliorer le filtre SMA200 baseline.

Approche:
1. Features: indicateurs de trend et volatilité
2. Target: rendement futur positif/négatif
3. Validation: split temporel (pas de look-ahead bias)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MLRegimeModel:
    """Conteneur pour le modèle ML et ses métadonnées."""
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list
    train_end_date: str
    metrics: dict


def compute_features(prices: pd.Series) -> pd.DataFrame:
    """
    Calcule les features pour la prédiction de régime.

    Features utilisées (toutes interprétables):
    - mom_1m: Momentum 1 mois (tendance court terme)
    - mom_3m: Momentum 3 mois (tendance moyen terme)
    - mom_6m: Momentum 6 mois (tendance long terme)
    - vol_20d: Volatilité réalisée 20 jours
    - vol_60d: Volatilité réalisée 60 jours
    - vol_ratio: Ratio vol court/long (stress indicator)
    - price_sma50: Prix vs SMA50
    - price_sma200: Prix vs SMA200
    """
    returns = prices.pct_change()

    features = pd.DataFrame(index=prices.index)

    # Momentum (tendance)
    features['mom_1m'] = prices / prices.shift(21) - 1
    features['mom_3m'] = prices / prices.shift(63) - 1
    features['mom_6m'] = prices / prices.shift(126) - 1

    # Volatilité
    features['vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
    features['vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
    features['vol_ratio'] = features['vol_20d'] / features['vol_60d']

    # Position vs moyennes mobiles
    features['price_sma50'] = prices / prices.rolling(50).mean() - 1
    features['price_sma200'] = prices / prices.rolling(200).mean() - 1

    return features.dropna()


def compute_target(prices: pd.Series, forward_days: int = 21) -> pd.Series:
    """
    Calcule le target: rendement futur positif (1) ou négatif (0).

    Note: Cette fonction utilise des données futures et ne doit être
    utilisée QUE pour l'entraînement, jamais en production.
    """
    forward_return = prices.shift(-forward_days) / prices - 1
    target = (forward_return > 0).astype(int)
    return target


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_end: str,
    model_type: str = 'logistic'
) -> MLRegimeModel:
    """
    Entraîne le modèle de classification de régime.

    IMPORTANT: Split temporel pour éviter le look-ahead bias.
    """
    # Aligner features et target
    common_idx = features.index.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]

    # Split temporel
    train_mask = X.index <= train_end
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle
    if model_type == 'logistic':
        model = LogisticRegression(
            penalty='l2',
            C=0.1,  # Régularisation forte pour éviter overfitting
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,  # Peu profond pour éviter overfitting
            class_weight='balanced',
            random_state=42
        )

    model.fit(X_train_scaled, y_train)

    # Métriques
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_precision': precision_score(y_train, y_pred_train),
        'test_precision': precision_score(y_test, y_pred_test),
        'train_f1': f1_score(y_train, y_pred_train),
        'test_f1': f1_score(y_test, y_pred_test),
        'train_size': len(y_train),
        'test_size': len(y_test),
    }

    return MLRegimeModel(
        model=model,
        scaler=scaler,
        feature_names=list(features.columns),
        train_end_date=train_end,
        metrics=metrics
    )


def predict_regime(
    features: pd.DataFrame,
    ml_model: MLRegimeModel
) -> pd.DataFrame:
    """
    Prédit le régime avec probabilités.

    Retourne:
    - prediction: 1 (Risk-On) ou 0 (Risk-Off)
    - probability: probabilité de Risk-On
    """
    X_scaled = ml_model.scaler.transform(features)

    predictions = ml_model.model.predict(X_scaled)
    probabilities = ml_model.model.predict_proba(X_scaled)[:, 1]

    result = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    }, index=features.index)

    return result


def get_feature_importance(ml_model: MLRegimeModel) -> pd.DataFrame:
    """
    Retourne l'importance des features.

    Pour Logistic Regression: coefficients
    Pour Random Forest: feature importances
    """
    if hasattr(ml_model.model, 'coef_'):
        # Logistic Regression
        importance = np.abs(ml_model.model.coef_[0])
    else:
        # Random Forest
        importance = ml_model.model.feature_importances_

    df = pd.DataFrame({
        'feature': ml_model.feature_names,
        'importance': importance
    })

    return df.sort_values('importance', ascending=False)


def compute_ml_regime_signal(
    benchmark_daily: pd.DataFrame,
    train_end: str = '2018-12-31',
    threshold: float = 0.5
) -> Tuple[pd.Series, MLRegimeModel]:
    """
    Pipeline complet pour générer le signal de régime ML.

    Paramètres:
    - benchmark_daily: Prix journaliers du benchmark (SPY)
    - train_end: Date de fin d'entraînement
    - threshold: Seuil de probabilité pour Risk-On

    Retourne:
    - regime_signal: Signal mensuel (1=Risk-On, 0=Risk-Off)
    - ml_model: Modèle entraîné (pour analyse)
    """
    if isinstance(benchmark_daily, pd.DataFrame):
        prices = benchmark_daily.iloc[:, 0]
    else:
        prices = benchmark_daily

    # Features et target
    features = compute_features(prices)
    target = compute_target(prices, forward_days=21)

    # Entraînement
    ml_model = train_model(features, target, train_end)

    # Prédiction sur toutes les données
    predictions = predict_regime(features, ml_model)

    # Signal binaire
    regime_daily = (predictions['probability'] > threshold).astype(int)

    # Resample mensuel (fin de mois)
    regime_monthly = regime_daily.resample('ME').last()

    return regime_monthly, ml_model


def compare_regimes(
    regime_baseline: pd.Series,
    regime_ml: pd.Series,
    benchmark_returns: pd.Series
) -> pd.DataFrame:
    """
    Compare les performances des deux filtres de régime.
    """
    # Aligner les indices
    common_idx = regime_baseline.index.intersection(
        regime_ml.index
    ).intersection(benchmark_returns.index)

    baseline = regime_baseline.loc[common_idx]
    ml = regime_ml.loc[common_idx]
    returns = benchmark_returns.loc[common_idx]

    # Calculs
    results = []

    for name, regime in [('SMA200', baseline), ('ML', ml)]:
        risk_on_returns = returns[regime == 1]
        risk_off_returns = returns[regime == 0]

        results.append({
            'Méthode': name,
            'Temps Risk-On': f"{regime.mean():.1%}",
            'Rend. moyen Risk-On': f"{risk_on_returns.mean()*12:.1%}",
            'Rend. moyen Risk-Off': f"{risk_off_returns.mean()*12:.1%}",
            'Sharpe Risk-On': f"{risk_on_returns.mean()/risk_on_returns.std()*np.sqrt(12):.2f}",
        })

    return pd.DataFrame(results)
