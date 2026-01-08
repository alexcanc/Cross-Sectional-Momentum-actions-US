"""
Script d'entraînement du modèle ML de régime.

Usage:
    python train_model.py                    # Entraînement avec paramètres par défaut
    python train_model.py --fresh            # Force le téléchargement de nouvelles données
    python train_model.py --train-end 2020   # Spécifie la fin de la période d'entraînement

Le script:
1. Télécharge les données les plus récentes
2. Calcule les features
3. Entraîne le modèle (Logistic Regression)
4. Sauvegarde le modèle + métriques
5. Génère un rapport de progression
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from src.data_loader import load_benchmark_daily, clear_cache
from src.ml_regime import (
    compute_features,
    compute_target,
    train_model,
    predict_regime,
    get_feature_importance,
    MLRegimeModel
)


MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne le modèle ML de régime")
    parser.add_argument("--fresh", action="store_true",
                        help="Force le téléchargement de nouvelles données")
    parser.add_argument("--train-end", type=str, default="2018-12-31",
                        help="Date de fin d'entraînement (format: YYYY-MM-DD)")
    parser.add_argument("--model-type", type=str, default="logistic",
                        choices=["logistic", "random_forest"],
                        help="Type de modèle")
    return parser.parse_args()


def log(message: str):
    """Affiche un message avec timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def save_model(ml_model: MLRegimeModel, version: str):
    """Sauvegarde le modèle et ses métadonnées."""
    model_path = MODELS_DIR / f"regime_model_{version}.pkl"
    metadata_path = MODELS_DIR / f"regime_model_{version}_metadata.json"

    # Sauvegarder le modèle
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": ml_model.model,
            "scaler": ml_model.scaler,
            "feature_names": ml_model.feature_names,
        }, f)

    # Sauvegarder les métadonnées
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "train_end_date": ml_model.train_end_date,
        "metrics": ml_model.metrics,
        "feature_names": ml_model.feature_names,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Copier vers latest
    latest_path = MODELS_DIR / "latest_model.pkl"
    latest_meta_path = MODELS_DIR / "latest_metadata.json"

    with open(model_path, "rb") as src:
        with open(latest_path, "wb") as dst:
            dst.write(src.read())

    with open(metadata_path, "r", encoding="utf-8") as src:
        with open(latest_meta_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())

    return model_path, metadata_path


def load_latest_model():
    """Charge le dernier modèle sauvegardé."""
    model_path = MODELS_DIR / "latest_model.pkl"
    metadata_path = MODELS_DIR / "latest_metadata.json"

    if not model_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return data, metadata


def generate_report(ml_model: MLRegimeModel, predictions: pd.DataFrame):
    """Génère un rapport d'entraînement."""
    report = []
    report.append("=" * 60)
    report.append("RAPPORT D'ENTRAÎNEMENT - MODÈLE DE RÉGIME")
    report.append("=" * 60)
    report.append("")
    report.append(f"Date d'entraînement: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Fin période train:   {ml_model.train_end_date}")
    report.append("")
    report.append("MÉTRIQUES")
    report.append("-" * 40)
    report.append(f"  Train Accuracy: {ml_model.metrics['train_accuracy']:.2%}")
    report.append(f"  Test Accuracy:  {ml_model.metrics['test_accuracy']:.2%}")
    report.append(f"  Train F1 Score: {ml_model.metrics['train_f1']:.2%}")
    report.append(f"  Test F1 Score:  {ml_model.metrics['test_f1']:.2%}")
    report.append("")
    report.append(f"  Taille Train: {ml_model.metrics['train_size']} jours")
    report.append(f"  Taille Test:  {ml_model.metrics['test_size']} jours")
    report.append("")
    report.append("IMPORTANCE DES FEATURES")
    report.append("-" * 40)

    importance = get_feature_importance(ml_model)
    for _, row in importance.iterrows():
        bar = "#" * int(row['importance'] * 20)
        report.append(f"  {row['feature']:15} {bar} ({row['importance']:.3f})")

    report.append("")
    report.append("SIGNAL ACTUEL")
    report.append("-" * 40)

    last_proba = predictions['probability'].iloc[-1]
    regime = "RISK-ON" if last_proba > 0.5 else "RISK-OFF"
    report.append(f"  Probabilité Risk-On: {last_proba:.1%}")
    report.append(f"  Régime actuel:       {regime}")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


def main():
    args = parse_args()

    log("Démarrage de l'entraînement du modèle de régime")
    log("")

    # 1. Charger les données
    log("1. Chargement des données...")

    if args.fresh:
        log("   → Suppression du cache...")
        clear_cache()

    benchmark_daily = load_benchmark_daily(verbose=False)
    spy = benchmark_daily.iloc[:, 0]
    log(f"   → Données: {spy.index[0].date()} à {spy.index[-1].date()}")
    log(f"   → {len(spy)} observations")

    # 2. Calculer les features
    log("")
    log("2. Calcul des features...")
    features = compute_features(spy)
    log(f"   → {len(features.columns)} features calculées")
    log(f"   → {len(features)} observations valides")

    # 3. Calculer le target
    log("")
    log("3. Calcul du target (rendement futur)...")
    target = compute_target(spy, forward_days=21)
    positive_rate = target.dropna().mean()
    log(f"   → Distribution: {positive_rate:.1%} positifs, {1-positive_rate:.1%} négatifs")

    # 4. Entraîner le modèle
    log("")
    log(f"4. Entraînement du modèle ({args.model_type})...")
    log(f"   → Période train: jusqu'au {args.train_end}")

    ml_model = train_model(features, target, args.train_end, model_type=args.model_type)

    log(f"   → Train Accuracy: {ml_model.metrics['train_accuracy']:.2%}")
    log(f"   → Test Accuracy:  {ml_model.metrics['test_accuracy']:.2%}")

    # 5. Prédictions
    log("")
    log("5. Génération des prédictions...")
    predictions = predict_regime(features, ml_model)

    current_proba = predictions['probability'].iloc[-1]
    current_regime = "RISK-ON" if current_proba > 0.5 else "RISK-OFF"
    log(f"   → Régime actuel: {current_regime} (proba: {current_proba:.1%})")

    # 6. Sauvegarder le modèle
    log("")
    log("6. Sauvegarde du modèle...")
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path, metadata_path = save_model(ml_model, version)
    log(f"   → Modèle: {model_path}")
    log(f"   → Métadonnées: {metadata_path}")

    # 7. Générer le rapport
    log("")
    log("7. Génération du rapport...")
    report = generate_report(ml_model, predictions)

    report_path = MODELS_DIR / f"training_report_{version}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"   → Rapport: {report_path}")

    # Afficher le rapport
    log("")
    print(report)

    log("")
    log("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()
