"""
Script pour exécuter le Walk-Forward Analysis.

Usage:
    python run_walk_forward.py                    # Configuration par défaut
    python run_walk_forward.py --train-years 3    # Fenêtre train de 3 ans
    python run_walk_forward.py --model gb         # Utiliser Gradient Boosting

Ce script:
1. Charge les données de marché
2. Exécute le walk-forward sur plusieurs périodes
3. Génère un rapport détaillé
4. Sauvegarde les graphiques
"""

import argparse
from datetime import datetime
from pathlib import Path

from src.data_loader import load_benchmark_daily
from src.walk_forward import (
    run_walk_forward,
    plot_walk_forward_results,
    generate_walk_forward_report
)


MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis du modèle ML")
    parser.add_argument("--train-years", type=int, default=5,
                        help="Taille de la fenêtre d'entraînement (années)")
    parser.add_argument("--test-years", type=int, default=2,
                        help="Taille de la fenêtre de test (années)")
    parser.add_argument("--step-years", type=int, default=1,
                        help="Décalage entre les folds (années)")
    parser.add_argument("--model", type=str, default="logistic",
                        choices=["logistic", "gradient_boosting"],
                        help="Type de modèle")
    parser.add_argument("--no-plot", action="store_true",
                        help="Ne pas afficher les graphiques")
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print("=" * 60)
    print(" WALK-FORWARD ANALYSIS ".center(60))
    print(" Validation robuste du modele de regime ".center(60))
    print("=" * 60)
    print()

    # Charger les données
    print("Chargement des données...")
    benchmark_daily = load_benchmark_daily(verbose=False)
    spy = benchmark_daily.iloc[:, 0]
    print(f"Période: {spy.index[0].date()} → {spy.index[-1].date()}")
    print(f"Observations: {len(spy)}")
    print()

    # Exécuter le walk-forward
    analysis = run_walk_forward(
        prices=spy,
        train_years=args.train_years,
        test_years=args.test_years,
        step_years=args.step_years,
        model_type=args.model,
        verbose=True
    )

    # Générer le rapport
    print()
    report = generate_walk_forward_report(analysis)

    # Sauvegarder le rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = MODELS_DIR / f"walk_forward_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapport sauvegardé: {report_path}")

    # Afficher les graphiques
    if not args.no_plot:
        try:
            plot_walk_forward_results(analysis)
        except Exception as e:
            print(f"Impossible d'afficher les graphiques: {e}")

    print()
    print("Walk-Forward Analysis terminé!")
    print()


if __name__ == "__main__":
    main()
