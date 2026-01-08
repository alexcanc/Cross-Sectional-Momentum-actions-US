"""
Streamlit Dashboard - Cross-Sectional Momentum Strategy

Lancez l'application avec: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import des modules du projet
from src.data_loader import load_universe_data, load_benchmark_data, load_benchmark_daily
from src.signal import compute_momentum_signal, get_long_short_positions
from src.backtest import run_backtest, compute_drawdowns
from src.regime_filter import compute_regime_filter
from src.ml_regime import compute_ml_regime_signal, get_feature_importance
from src.metrics import compute_metrics, compute_annual_returns
from src.config import Config

# Configuration de la page
st.set_page_config(
    page_title="Momentum Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .explanation-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    """Charge les données avec cache."""
    prices = load_universe_data(verbose=False)
    benchmark = load_benchmark_data(verbose=False)
    benchmark_daily = load_benchmark_daily(verbose=False)
    return prices, benchmark, benchmark_daily


def main():
    # Header
    st.markdown('<p class="main-header">📈 Cross-Sectional Momentum</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Stratégie quantitative sur actions US (Dow Jones 30)</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    # Paramètres de la stratégie
    st.sidebar.subheader("Signal")
    lookback = st.sidebar.slider("Lookback (mois)", 6, 24, 12, help="Période de calcul du momentum")
    skip = st.sidebar.slider("Skip (mois)", 0, 3, 1, help="Mois récent à exclure")

    st.sidebar.subheader("Portefeuille")
    n_long = st.sidebar.slider("Nombre Long", 5, 15, 10)
    n_short = st.sidebar.slider("Nombre Short", 0, 15, 10)
    portfolio_type = "long_short" if n_short > 0 else "long_only"

    st.sidebar.subheader("Coûts")
    transaction_cost = st.sidebar.slider("Coût transaction (bps)", 0, 50, 10)

    st.sidebar.subheader("Filtre de régime")
    regime_type = st.sidebar.radio(
        "Type de filtre",
        ["Aucun", "SMA200 (baseline)", "Machine Learning"],
        index=1
    )
    risk_off_exposure = st.sidebar.slider(
        "Exposition Risk-Off", 0.0, 1.0, 0.5,
        disabled=(regime_type == "Aucun")
    )

    if regime_type == "Machine Learning":
        ml_train_end = st.sidebar.text_input("Fin période train", "2018-12-31")

    # Chargement des données
    with st.spinner("Chargement des données..."):
        prices, benchmark, benchmark_daily = load_data()

    # Configuration
    config = Config()
    config.MOMENTUM_LOOKBACK = lookback
    config.MOMENTUM_SKIP = skip
    config.N_LONG = n_long
    config.N_SHORT = n_short
    config.PORTFOLIO_TYPE = portfolio_type
    config.TRANSACTION_COST_BPS = transaction_cost
    config.RISK_OFF_EXPOSURE = risk_off_exposure

    # Calcul du régime
    regime = None
    ml_model = None

    if regime_type == "SMA200 (baseline)":
        regime = compute_regime_filter(benchmark_daily, config)
    elif regime_type == "Machine Learning":
        with st.spinner("Entraînement du modèle ML..."):
            regime, ml_model = compute_ml_regime_signal(benchmark_daily, train_end=ml_train_end)

    # Backtest
    results = run_backtest(prices, benchmark, config, regime_signal=regime)
    metrics = compute_metrics(
        results.strategy_returns,
        results.cumulative_returns,
        results.turnover,
        results.transaction_costs
    )

    # Tabs
    if ml_model is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance", "🔬 Signal", "📉 Risque", "🤖 ML", "📚 Méthodologie"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🔬 Signal", "📉 Risque", "📚 Méthodologie"])

    # TAB 1: Performance
    with tab1:
        st.header("Performance de la stratégie")

        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CAGR", f"{metrics.cagr:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        with col4:
            st.metric("Volatilité", f"{metrics.volatility:.2%}")

        # Graphique de performance
        st.subheader("Performance cumulée")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results.cumulative_returns.index,
            y=results.cumulative_returns.values,
            name="Stratégie",
            line=dict(width=2, color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=results.cumulative_benchmark.index,
            y=results.cumulative_benchmark.values,
            name="S&P 500",
            line=dict(width=2, color='gray', dash='dash')
        ))

        fig.update_layout(
            yaxis_type="log",
            yaxis_title="Croissance de $1",
            xaxis_title="",
            hovermode='x unified',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Explication
        st.markdown("""
        <div class="explanation-box">
        <strong>💡 Lecture du graphique:</strong><br>
        Le graphique montre la croissance de 1$ investi au début de la période.
        L'échelle logarithmique permet de comparer les rendements proportionnels.
        </div>
        """, unsafe_allow_html=True)

        # Rendements annuels
        st.subheader("Rendements annuels")
        annual = compute_annual_returns(results.strategy_returns)
        annual_bench = compute_annual_returns(results.benchmark_returns)

        annual_df = pd.DataFrame({
            'Stratégie': annual,
            'S&P 500': annual_bench
        })

        fig2 = px.bar(annual_df, barmode='group',
                      color_discrete_sequence=['#1f77b4', 'gray'])
        fig2.update_layout(
            yaxis_title="Rendement",
            xaxis_title="Année",
            height=350,
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # TAB 2: Signal
    with tab2:
        st.header("Signal Momentum")

        # Calcul du momentum
        momentum = compute_momentum_signal(prices, lookback, skip)

        # Explication
        st.markdown("""
        <div class="explanation-box">
        <strong>📐 Formule du signal:</strong><br>
        <code>Momentum = Prix(t-1) / Prix(t-12) - 1</code><br><br>
        Le signal mesure le rendement sur les 12 derniers mois, en excluant le mois le plus récent
        pour éviter l'effet de mean reversion à court terme.
        </div>
        """, unsafe_allow_html=True)

        # Dernier classement
        st.subheader("Classement actuel")
        last_date = momentum.dropna().index[-1]
        last_momentum = momentum.loc[last_date].sort_values(ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Positions LONG (Top 10)**")
            long_stocks = last_momentum.head(n_long)
            for ticker, mom in long_stocks.items():
                st.write(f"• **{ticker}**: {mom:.2%}")

        with col2:
            if n_short > 0:
                st.markdown("**🔴 Positions SHORT (Bottom 10)**")
                short_stocks = last_momentum.tail(n_short)
                for ticker, mom in short_stocks.items():
                    st.write(f"• **{ticker}**: {mom:.2%}")

        # Distribution du momentum
        st.subheader("Distribution du momentum")
        fig3 = px.histogram(last_momentum.values, nbins=15,
                           labels={'value': 'Momentum', 'count': 'Nombre d\'actions'})
        fig3.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # TAB 3: Risque
    with tab3:
        st.header("Analyse des risques")

        # Drawdowns
        st.subheader("Drawdowns")
        drawdowns = compute_drawdowns(results.cumulative_returns)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='red'),
            name='Drawdown'
        ))
        fig4.add_hline(y=-0.2, line_dash="dash", line_color="orange",
                       annotation_text="Bear Market (-20%)")
        fig4.update_layout(
            yaxis_title="Drawdown",
            height=350,
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Métriques de risque
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        with col2:
            st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
        with col3:
            st.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")

        # Explication
        st.markdown("""
        <div class="explanation-box">
        <strong>📊 Définitions:</strong><br>
        • <strong>Drawdown</strong>: Baisse depuis le plus haut historique<br>
        • <strong>Sortino</strong>: Sharpe ajusté pour la volatilité baissière uniquement<br>
        • <strong>Calmar</strong>: CAGR / Max Drawdown
        </div>
        """, unsafe_allow_html=True)

        # Turnover
        st.subheader("Turnover et coûts")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Turnover mensuel moyen", f"{metrics.avg_turnover:.1%}")
        with col2:
            st.metric("Coûts totaux (sur la période)", f"{metrics.total_transaction_costs:.2%}")

    # TAB 4: ML (si activé)
    if ml_model is not None:
        with tab4:
            st.header("Modèle Machine Learning")

            # Métriques du modèle
            st.subheader("Performance du modèle")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy Train", f"{ml_model.metrics['train_accuracy']:.1%}")
            with col2:
                st.metric("Accuracy Test", f"{ml_model.metrics['test_accuracy']:.1%}")
            with col3:
                st.metric("F1 Train", f"{ml_model.metrics['train_f1']:.1%}")
            with col4:
                st.metric("F1 Test", f"{ml_model.metrics['test_f1']:.1%}")

            st.markdown("""
            <div class="explanation-box">
            <strong>💡 Interprétation:</strong><br>
            • <strong>Accuracy</strong>: % de prédictions correctes<br>
            • <strong>F1 Score</strong>: Équilibre précision/rappel<br>
            • Une accuracy de 55-60% est normale pour la prédiction de marchés
            </div>
            """, unsafe_allow_html=True)

            # Importance des features
            st.subheader("Importance des features")

            importance = get_feature_importance(ml_model)
            fig_imp = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(height=350, showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("""
            <div class="explanation-box">
            <strong>📊 Features utilisées:</strong><br>
            • <strong>mom_Xm</strong>: Momentum sur X mois (tendance)<br>
            • <strong>vol_Xd</strong>: Volatilité réalisée sur X jours<br>
            • <strong>price_smaX</strong>: Position du prix vs moyenne mobile
            </div>
            """, unsafe_allow_html=True)

            # Signal actuel
            st.subheader("Signal ML actuel")

            from src.ml_regime import compute_features, predict_regime
            features = compute_features(benchmark_daily.iloc[:, 0])
            predictions = predict_regime(features, ml_model)

            last_proba = predictions['probability'].iloc[-1]
            last_regime = "RISK-ON" if last_proba > 0.5 else "RISK-OFF"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilité Risk-On", f"{last_proba:.1%}")
            with col2:
                if last_regime == "RISK-ON":
                    st.success(f"Régime: {last_regime}")
                else:
                    st.error(f"Régime: {last_regime}")

            # Graphique de la probabilité
            fig_proba = go.Figure()
            fig_proba.add_trace(go.Scatter(
                x=predictions.index,
                y=predictions['probability'],
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color='#1f77b4'),
                name='Probabilité Risk-On'
            ))
            fig_proba.add_hline(y=0.5, line_dash="dash", line_color="red",
                               annotation_text="Seuil 50%")
            fig_proba.update_layout(
                yaxis_title="Probabilité",
                height=300,
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig_proba, use_container_width=True)

    # TAB 5 (ou 4): Méthodologie
    methodology_tab = tab5 if ml_model is not None else tab4
    with methodology_tab:
        st.header("Méthodologie")

        st.markdown("""
        ### 1. Le momentum en finance

        L'effet momentum est l'une des anomalies les plus documentées en finance:
        **les actions qui ont bien performé continuent de bien performer à moyen terme**.

        Ce phénomène a été découvert par Jegadeesh & Titman (1993) et persiste depuis.

        ### 2. Signal 12-1

        Le signal **12-1** mesure le rendement sur 12 mois en excluant le dernier mois:

        ```
        Momentum = P(t-1) / P(t-12) - 1
        ```

        **Pourquoi exclure le dernier mois?**
        Le rendement du mois le plus récent montre souvent un effet de mean reversion
        (retour à la moyenne) qui pollue le signal momentum.

        ### 3. Construction du portefeuille

        Chaque mois:
        1. Calculer le momentum pour toutes les actions
        2. Classer les actions par momentum
        3. Acheter les top 10 (LONG)
        4. Vendre à découvert les bottom 10 (SHORT)
        5. Pondération equal-weight

        ### 4. Filtre de régime

        Le filtre SMA200 est une règle simple:
        - Si S&P 500 > Moyenne mobile 200 jours → **Risk-On** (exposition normale)
        - Si S&P 500 < Moyenne mobile 200 jours → **Risk-Off** (exposition réduite)

        ### 5. Limites de l'étude

        - **Survivorship bias**: On utilise les constituants actuels du DJ30
        - **Coûts de short**: Non modélisés
        - **Slippage**: Non inclus
        - **Capacité**: Limitée par la taille de l'univers

        ### Références

        - Jegadeesh & Titman (1993). *Returns to Buying Winners and Selling Losers*
        - Faber (2007). *A Quantitative Approach to Tactical Asset Allocation*
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Projet de recherche quantitative | Données: Yahoo Finance |
    <a href="https://github.com" target="_blank">Code source</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
