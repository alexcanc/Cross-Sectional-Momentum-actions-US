"""
Walk-Forward Analysis pour le modèle de régime.

Le Walk-Forward est la méthode standard en quant pour valider un modèle:
1. On découpe l'historique en fenêtres
2. Pour chaque fenêtre: entraînement sur le passé, test sur le futur
3. On agrège les résultats pour avoir une vue réaliste

C'est plus robuste que train/test unique car:
- On teste sur plusieurs périodes de marché différentes
- On simule le réentraînement périodique (comme en prod)
- On détecte si le modèle se dégrade dans le temps
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .ml_regime import compute_features, compute_target


@dataclass
class WalkForwardResult:
    """Résultat d'une fenêtre de walk-forward."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_accuracy: float
    test_accuracy: float
    train_f1: float
    test_f1: float
    test_auc: float
    train_size: int
    test_size: int
    training_time: float


@dataclass
class WalkForwardAnalysis:
    """Résultat complet du walk-forward."""
    results: List[WalkForwardResult]
    avg_test_accuracy: float
    avg_test_f1: float
    avg_test_auc: float
    std_test_accuracy: float
    stability_score: float  # Écart-type normalisé des performances


def generate_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 2,
    step_years: int = 1,
) -> List[Tuple[str, str, str, str]]:
    """
    Génère les splits pour le walk-forward.

    Paramètres:
    - train_years: Taille de la fenêtre d'entraînement
    - test_years: Taille de la fenêtre de test
    - step_years: Décalage entre chaque fold

    Retourne liste de (train_start, train_end, test_start, test_end)
    """
    splits = []

    start_year = dates.min().year
    end_year = dates.max().year

    current_train_start = start_year

    while True:
        train_end = current_train_start + train_years
        test_start = train_end
        test_end = test_start + test_years

        if test_end > end_year:
            break

        splits.append((
            f"{current_train_start}-01-01",
            f"{train_end}-01-01",
            f"{test_start}-01-01",
            f"{test_end}-01-01",
        ))

        current_train_start += step_years

    return splits


def run_walk_forward(
    prices: pd.Series,
    train_years: int = 5,
    test_years: int = 2,
    step_years: int = 1,
    model_type: str = 'logistic',
    verbose: bool = True,
) -> WalkForwardAnalysis:
    """
    Exécute le walk-forward analysis complet.

    Paramètres:
    - prices: Prix journaliers du benchmark
    - train_years: Taille fenêtre train
    - test_years: Taille fenêtre test
    - step_years: Décalage entre folds
    - model_type: 'logistic' ou 'gradient_boosting'
    - verbose: Afficher la progression

    Retourne:
    - WalkForwardAnalysis avec tous les résultats
    """
    if verbose:
        print("=" * 60)
        print("WALK-FORWARD ANALYSIS")
        print("=" * 60)
        print(f"Train window: {train_years} ans")
        print(f"Test window:  {test_years} ans")
        print(f"Step:         {step_years} an")
        print(f"Model:        {model_type}")
        print("=" * 60)
        print()

    # Calculer features et target
    features = compute_features(prices)
    target = compute_target(prices, forward_days=21)

    # Aligner
    common_idx = features.index.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]

    # Générer les splits
    splits = generate_walk_forward_splits(X.index, train_years, test_years, step_years)

    if verbose:
        print(f"Nombre de folds: {len(splits)}")
        print()

    results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
        if verbose:
            print(f"Fold {i+1}/{len(splits)}: Train {train_start[:4]}-{train_end[:4]} | Test {test_start[:4]}-{test_end[:4]}")

        start_time = time.time()

        # Split des données
        train_mask = (X.index >= train_start) & (X.index < train_end)
        test_mask = (X.index >= test_start) & (X.index < test_end)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modèle
        if model_type == 'logistic':
            model = LogisticRegression(
                penalty='l2',
                C=0.1,
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )

        # Entraînement
        model.fit(X_train_scaled, y_train)

        # Prédictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

        training_time = time.time() - start_time

        # Métriques
        result = WalkForwardResult(
            fold=i + 1,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_accuracy=accuracy_score(y_train, y_train_pred),
            test_accuracy=accuracy_score(y_test, y_test_pred),
            train_f1=f1_score(y_train, y_train_pred),
            test_f1=f1_score(y_test, y_test_pred),
            test_auc=roc_auc_score(y_test, y_test_proba),
            train_size=len(y_train),
            test_size=len(y_test),
            training_time=training_time
        )

        results.append(result)

        if verbose:
            print(f"  → Train Acc: {result.train_accuracy:.2%} | Test Acc: {result.test_accuracy:.2%} | AUC: {result.test_auc:.3f} | Time: {result.training_time:.1f}s")

    if verbose:
        print()
        print("=" * 60)

    # Agrégation
    test_accuracies = [r.test_accuracy for r in results]
    test_f1s = [r.test_f1 for r in results]
    test_aucs = [r.test_auc for r in results]

    analysis = WalkForwardAnalysis(
        results=results,
        avg_test_accuracy=np.mean(test_accuracies),
        avg_test_f1=np.mean(test_f1s),
        avg_test_auc=np.mean(test_aucs),
        std_test_accuracy=np.std(test_accuracies),
        stability_score=1 - (np.std(test_accuracies) / np.mean(test_accuracies))
    )

    if verbose:
        print("RÉSULTATS AGRÉGÉS")
        print("-" * 40)
        print(f"Accuracy moyenne: {analysis.avg_test_accuracy:.2%} (± {analysis.std_test_accuracy:.2%})")
        print(f"F1 Score moyen:   {analysis.avg_test_f1:.2%}")
        print(f"AUC moyen:        {analysis.avg_test_auc:.3f}")
        print(f"Score stabilité:  {analysis.stability_score:.2%}")
        print("=" * 60)

    return analysis


def plot_walk_forward_results(analysis: WalkForwardAnalysis) -> None:
    """Affiche les résultats du walk-forward graphiquement."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy par fold
    folds = [r.fold for r in analysis.results]
    train_acc = [r.train_accuracy for r in analysis.results]
    test_acc = [r.test_accuracy for r in analysis.results]

    axes[0, 0].plot(folds, train_acc, 'o-', label='Train', linewidth=2)
    axes[0, 0].plot(folds, test_acc, 's-', label='Test', linewidth=2)
    axes[0, 0].axhline(0.5, color='gray', linestyle='--', label='Random')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy par fold')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.4, 0.8)

    # 2. Overfitting (Train - Test)
    overfitting = [r.train_accuracy - r.test_accuracy for r in analysis.results]
    colors = ['red' if o > 0.1 else 'green' for o in overfitting]
    axes[0, 1].bar(folds, overfitting, color=colors, alpha=0.7)
    axes[0, 1].axhline(0, color='black', linewidth=0.8)
    axes[0, 1].axhline(0.1, color='red', linestyle='--', label='Seuil overfitting')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Train Acc - Test Acc')
    axes[0, 1].set_title('Détection overfitting')
    axes[0, 1].legend()

    # 3. AUC par fold
    test_auc = [r.test_auc for r in analysis.results]
    axes[1, 0].bar(folds, test_auc, color='steelblue', alpha=0.7)
    axes[1, 0].axhline(0.5, color='gray', linestyle='--', label='Random')
    axes[1, 0].axhline(analysis.avg_test_auc, color='red', linestyle='--',
                        label=f'Moyenne: {analysis.avg_test_auc:.3f}')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('AUC ROC par fold')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0.4, 0.7)

    # 4. Temps d'entraînement
    train_times = [r.training_time for r in analysis.results]
    axes[1, 1].bar(folds, train_times, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Temps (secondes)')
    axes[1, 1].set_title('Temps d\'entraînement')

    plt.tight_layout()
    plt.savefig('models/walk_forward_results.png', dpi=150)
    plt.show()

    print("Graphique sauvegardé: models/walk_forward_results.png")


def generate_walk_forward_report(analysis: WalkForwardAnalysis) -> str:
    """Génère un rapport texte du walk-forward."""
    lines = []
    lines.append("=" * 70)
    lines.append("WALK-FORWARD ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Nombre de folds: {len(analysis.results)}")
    lines.append("")
    lines.append("RÉSULTATS PAR FOLD")
    lines.append("-" * 70)
    lines.append(f"{'Fold':>4} {'Train Period':<20} {'Test Period':<20} {'Acc Train':>10} {'Acc Test':>10}")
    lines.append("-" * 70)

    for r in analysis.results:
        train_period = f"{r.train_start[:4]}-{r.train_end[:4]}"
        test_period = f"{r.test_start[:4]}-{r.test_end[:4]}"
        lines.append(f"{r.fold:>4} {train_period:<20} {test_period:<20} {r.train_accuracy:>10.2%} {r.test_accuracy:>10.2%}")

    lines.append("-" * 70)
    lines.append("")
    lines.append("MÉTRIQUES AGRÉGÉES")
    lines.append("-" * 40)
    lines.append(f"Accuracy moyenne (test):  {analysis.avg_test_accuracy:.2%}")
    lines.append(f"Écart-type accuracy:      {analysis.std_test_accuracy:.2%}")
    lines.append(f"F1 Score moyen (test):    {analysis.avg_test_f1:.2%}")
    lines.append(f"AUC moyen (test):         {analysis.avg_test_auc:.3f}")
    lines.append(f"Score de stabilité:       {analysis.stability_score:.2%}")
    lines.append("")
    lines.append("INTERPRÉTATION")
    lines.append("-" * 40)

    if analysis.avg_test_accuracy > 0.55:
        lines.append("[OK] Le modele a un pouvoir predictif significatif")
    else:
        lines.append("[!!] Le pouvoir predictif est limite")

    if analysis.stability_score > 0.9:
        lines.append("[OK] Le modele est stable dans le temps")
    else:
        lines.append("[!!] Performance variable selon les periodes")

    avg_overfit = np.mean([r.train_accuracy - r.test_accuracy for r in analysis.results])
    if avg_overfit < 0.1:
        lines.append("[OK] Pas de signe majeur d'overfitting")
    else:
        lines.append("[!!] Risque d'overfitting detecte")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
