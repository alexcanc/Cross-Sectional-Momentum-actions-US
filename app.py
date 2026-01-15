"""
Cross-Sectional Momentum Strategy Dashboard
Imperial College London - Quantitative Research Project

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Project modules
from src.data_loader import load_universe_data, load_benchmark_data, load_benchmark_daily
from src.signal import compute_momentum_signal
from src.backtest import run_backtest, compute_drawdowns
from src.regime_filter import compute_regime_filter
from src.ml_regime import compute_ml_regime_signal, get_feature_importance
from src.metrics import compute_metrics, compute_annual_returns
from src.config import Config

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Momentum Strategy | Quant Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Modern Professional Design
# =============================================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }

    .metric-positive { color: #10b981; }
    .metric-negative { color: #ef4444; }
    .metric-neutral { color: #667eea; }

    /* Info Box */
    .info-box {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #374151;
    }

    .info-box strong {
        color: #1f2937;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }

    /* Stock List */
    .stock-item {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        background: #f9fafb;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    .stock-ticker {
        font-weight: 600;
        color: #1f2937;
    }

    .stock-momentum-positive { color: #10b981; font-weight: 500; }
    .stock-momentum-negative { color: #ef4444; font-weight: 500; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #9ca3af;
        font-size: 0.85rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }

    .footer a {
        color: #667eea;
        text-decoration: none;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .sidebar-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.75rem; }
        .metric-value { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load and cache market data."""
    prices = load_universe_data(verbose=False)
    benchmark = load_benchmark_data(verbose=False)
    benchmark_daily = load_benchmark_daily(verbose=False)
    return prices, benchmark, benchmark_daily


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_percentage(value, decimals=2):
    """Format a value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value, decimals=2):
    """Format a ratio value."""
    return f"{value:.{decimals}f}"


def render_metric_card(value, label, format_type="percentage", color_class="neutral"):
    """Render a styled metric card."""
    if format_type == "percentage":
        formatted = format_percentage(value)
    else:
        formatted = format_ratio(value)

    return f"""
    <div class="metric-card">
        <div class="metric-value metric-{color_class}">{formatted}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # =========================================================================
    # HERO SECTION
    # =========================================================================
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Cross-Sectional Momentum Strategy</div>
        <div class="hero-subtitle">Quantitative Research on US Equities (Dow Jones 30)</div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SIDEBAR - Strategy Configuration
    # =========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Strategy Parameters")
        st.caption("Adjust these to see how they affect performance")

        st.markdown('<div class="sidebar-title">📊 Signal Parameters</div>', unsafe_allow_html=True)
        lookback = st.slider(
            "Lookback Period (months)",
            min_value=6, max_value=24, value=12,
            help="How far back to look when measuring momentum. 12 months is the academic standard."
        )
        skip = st.slider(
            "Skip Period (months)",
            min_value=0, max_value=3, value=1,
            help="Skip the most recent month to avoid short-term reversal effect."
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-title">📈 Portfolio Construction</div>', unsafe_allow_html=True)
        n_long = st.slider(
            "Long Positions (buy winners)",
            min_value=3, max_value=15, value=10,
            help="Number of top-performing stocks to buy"
        )
        n_short = st.slider(
            "Short Positions (sell losers)",
            min_value=0, max_value=15, value=10,
            help="Number of worst-performing stocks to short-sell. Set to 0 for long-only."
        )
        portfolio_type = "long_short" if n_short > 0 else "long_only"

        st.markdown("---")
        st.markdown('<div class="sidebar-title">💰 Transaction Costs</div>', unsafe_allow_html=True)
        transaction_cost = st.slider(
            "Cost (basis points)",
            min_value=0, max_value=50, value=10,
            help="Trading cost per transaction. 10 bps = 0.1% per trade. Realistic for large-cap stocks."
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-title">🛡️ Regime Filter</div>', unsafe_allow_html=True)
        st.caption("Reduce exposure during bear markets")
        regime_type = st.radio(
            "Filter Type",
            options=["None", "SMA 200 (Trend)", "Machine Learning"],
            index=1,
            help="When market is below its 200-day average, reduce exposure to protect capital."
        )

        if regime_type != "None":
            risk_off_exposure = st.slider(
                "Risk-Off Exposure",
                min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                help="Portfolio exposure during bearish regime"
            )
        else:
            risk_off_exposure = 0.5

        if regime_type == "Machine Learning":
            ml_train_end = st.text_input(
                "Training End Date",
                value="2018-12-31",
                help="End date for ML model training period"
            )

    # =========================================================================
    # DATA LOADING & PROCESSING
    # =========================================================================
    with st.spinner("Loading market data..."):
        try:
            prices, benchmark, benchmark_daily = load_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Please check your internet connection and try again.")
            return

    # Validate data
    if prices is None or prices.empty:
        st.error("Failed to load price data. Please try refreshing the page.")
        st.info("If the problem persists, the data source may be temporarily unavailable.")
        return

    if benchmark is None or benchmark.empty:
        st.warning("Benchmark data unavailable. Using equal-weight universe as benchmark.")
        benchmark = None

    if benchmark_daily is None or benchmark_daily.empty:
        st.warning("Daily benchmark data unavailable. Regime filter disabled.")
        benchmark_daily = None

    # Check minimum data requirement
    min_months_required = lookback + 2
    if len(prices) < min_months_required:
        st.error(f"Insufficient data: {len(prices)} months available, but {min_months_required} months required for lookback={lookback}.")
        st.info("Try reducing the lookback period in the sidebar.")
        return

    # Configuration
    config = Config()
    config.MOMENTUM_LOOKBACK = lookback
    config.MOMENTUM_SKIP = skip
    config.N_LONG = n_long
    config.N_SHORT = n_short
    config.PORTFOLIO_TYPE = portfolio_type
    config.TRANSACTION_COST_BPS = transaction_cost
    config.RISK_OFF_EXPOSURE = risk_off_exposure

    # Regime computation
    regime = None
    ml_model = None

    if regime_type != "None" and benchmark_daily is not None:
        if regime_type == "SMA 200 (Trend)":
            try:
                regime = compute_regime_filter(benchmark_daily, config)
            except Exception as e:
                st.warning(f"Regime filter failed: {str(e)}. Running without regime filter.")
        elif regime_type == "Machine Learning":
            with st.spinner("Training ML model..."):
                try:
                    regime, ml_model = compute_ml_regime_signal(benchmark_daily, train_end=ml_train_end)
                except Exception as e:
                    st.warning(f"ML model training failed: {str(e)}. Using SMA 200 as fallback.")
                    try:
                        regime = compute_regime_filter(benchmark_daily, config)
                    except Exception:
                        pass

    # Backtest
    try:
        results = run_backtest(prices, benchmark, config, regime_signal=regime)
        metrics = compute_metrics(
            results.strategy_returns,
            results.cumulative_returns,
            results.turnover,
            results.transaction_costs
        )
    except Exception as e:
        st.error(f"Backtest error: {str(e)}")
        return

    # =========================================================================
    # STRATEGY EXPLANATION
    # =========================================================================
    st.markdown("""
    <div class="info-box">
        <strong>What is this strategy?</strong> This is a <b>momentum strategy</b> that buys stocks
        that have performed well recently (winners) and sells stocks that have performed poorly (losers).
        The idea is simple: <i>trends tend to continue</i>. Stocks going up often keep going up,
        and stocks going down often keep going down. Adjust the parameters in the sidebar to see
        how different settings affect performance.
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # CURRENT RECOMMENDATIONS (ACTIONABLE!)
    # =========================================================================
    st.markdown('<div class="section-header">🎯 Current Recommendations</div>', unsafe_allow_html=True)

    # Calculate current momentum for recommendations
    from src.signal import compute_momentum_signal
    current_momentum = compute_momentum_signal(prices, config.MOMENTUM_LOOKBACK, config.MOMENTUM_SKIP)
    valid_momentum = current_momentum.dropna(how='all')

    if not valid_momentum.empty:
        last_date = valid_momentum.index[-1]
        latest_momentum = current_momentum.loc[last_date].dropna().sort_values(ascending=False)

        # Determine current regime
        current_regime = "Risk-On 🟢" if regime is None or regime.iloc[-1] == 1 else "Risk-Off 🔴"
        exposure_pct = 100 if regime is None or regime.iloc[-1] == 1 else int(risk_off_exposure * 100)

        # Display current status
        col_regime, col_buy, col_sell = st.columns([1, 2, 2])

        with col_regime:
            # Color based on regime
            if regime is None:
                regime_color = "#6b7280"  # Gray for no filter
                regime_text = "No Filter"
                regime_icon = "⚪"
            elif regime.iloc[-1] == 1:
                regime_color = "#22c55e"  # Green for risk-on
                regime_text = "Risk-On"
                regime_icon = "🟢"
            else:
                regime_color = "#ef4444"  # Red for risk-off
                regime_text = "Risk-Off"
                regime_icon = "🔴"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {regime_color}dd 0%, {regime_color}99 100%);
                        padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Market Regime</div>
                <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">{regime_text} {regime_icon}</div>
                <div style="font-size: 14px;">Exposure: {exposure_pct}%</div>
                <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">as of {last_date.strftime('%B %Y')}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("ℹ️ Regime depends on S&P 500 vs SMA200, not strategy params")

        with col_buy:
            st.markdown("#### 📈 BUY These Stocks")
            buy_stocks = latest_momentum.head(n_long)
            for i, (ticker, mom) in enumerate(buy_stocks.items(), 1):
                color = "#22c55e" if mom > 0 else "#ef4444"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 12px;
                            background: #f8fafc; border-radius: 6px; margin-bottom: 4px; border-left: 3px solid #22c55e;">
                    <span><b>{i}. {ticker}</b></span>
                    <span style="color: {color}; font-weight: 600;">{mom*100:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

        with col_sell:
            if n_short > 0:
                st.markdown("#### 📉 SHORT These Stocks")
                sell_stocks = latest_momentum.tail(n_short)
                for i, (ticker, mom) in enumerate(sell_stocks.items(), 1):
                    color = "#22c55e" if mom > 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 12px;
                                background: #f8fafc; border-radius: 6px; margin-bottom: 4px; border-left: 3px solid #ef4444;">
                        <span><b>{i}. {ticker}</b></span>
                        <span style="color: {color}; font-weight: 600;">{mom*100:+.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("#### ℹ️ Long-Only Mode")
                st.info("Short selling disabled. Only buying winners.")

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # KEY METRICS DISPLAY
    # =========================================================================
    st.markdown('<div class="section-header">Key Performance Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        color = "positive" if metrics.cagr > 0 else "negative"
        st.markdown(render_metric_card(metrics.cagr, "CAGR", "percentage", color), unsafe_allow_html=True)

    with col2:
        color = "positive" if metrics.sharpe_ratio > 1 else ("neutral" if metrics.sharpe_ratio > 0.5 else "negative")
        st.markdown(render_metric_card(metrics.sharpe_ratio, "Sharpe Ratio", "ratio", color), unsafe_allow_html=True)

    with col3:
        color = "negative" if metrics.max_drawdown < -0.3 else ("neutral" if metrics.max_drawdown < -0.15 else "positive")
        st.markdown(render_metric_card(metrics.max_drawdown, "Max Drawdown", "percentage", color), unsafe_allow_html=True)

    with col4:
        st.markdown(render_metric_card(metrics.volatility, "Volatility", "percentage", "neutral"), unsafe_allow_html=True)

    with col5:
        color = "positive" if metrics.sortino_ratio > 1.5 else "neutral"
        st.markdown(render_metric_card(metrics.sortino_ratio, "Sortino Ratio", "ratio", color), unsafe_allow_html=True)

    # Metrics explanation
    with st.expander("📚 What do these metrics mean? (Click to expand)"):
        st.markdown("""
        | Metric | What it measures | Good value |
        |--------|------------------|------------|
        | **CAGR** | Average annual return (Compound Annual Growth Rate) | > 10% |
        | **Sharpe Ratio** | Return per unit of risk. Higher = better risk-adjusted performance | > 1.0 |
        | **Max Drawdown** | Largest peak-to-trough decline. How much you could lose at worst | > -20% |
        | **Volatility** | How much returns fluctuate. Lower = more stable | < 15% |
        | **Sortino Ratio** | Like Sharpe, but only penalizes downside risk | > 1.5 |
        """)

    # =========================================================================
    # TABS
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)

    if ml_model is not None:
        tabs = st.tabs(["📈 Performance", "📊 Signal Analysis", "⚠️ Risk Metrics", "🤖 ML Model", "📖 Methodology"])
        tab_perf, tab_signal, tab_risk, tab_ml, tab_method = tabs
    else:
        tabs = st.tabs(["📈 Performance", "📊 Signal Analysis", "⚠️ Risk Metrics", "📖 Methodology"])
        tab_perf, tab_signal, tab_risk, tab_method = tabs
        tab_ml = None

    # =========================================================================
    # TAB 1: PERFORMANCE
    # =========================================================================
    with tab_perf:
        st.markdown("### Cumulative Performance")

        # Performance chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=results.cumulative_returns.index,
            y=results.cumulative_returns.values,
            name="Momentum Strategy",
            line=dict(width=2.5, color='#667eea'),
            hovertemplate='%{x}<br>Strategy: $%{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=results.cumulative_benchmark.index,
            y=results.cumulative_benchmark.values,
            name="S&P 500 Benchmark",
            line=dict(width=2, color='#9ca3af', dash='dot'),
            hovertemplate='%{x}<br>Benchmark: $%{y:.2f}<extra></extra>'
        ))

        fig.update_layout(
            yaxis_type="log",
            yaxis_title="Growth of $1 (Log Scale)",
            xaxis_title="",
            hovermode='x unified',
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=60, r=40, t=60, b=40),
            yaxis=dict(gridcolor='#f0f0f0', zerolinecolor='#e5e7eb'),
            xaxis=dict(gridcolor='#f0f0f0')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <strong>Reading the Chart:</strong> The logarithmic scale shows proportional returns over time.
            A value of 2 means your initial investment has doubled. The strategy is compared against
            a passive S&P 500 investment.
        </div>
        """, unsafe_allow_html=True)

        # Annual returns
        st.markdown("### Annual Returns Comparison")

        annual = compute_annual_returns(results.strategy_returns)
        annual_bench = compute_annual_returns(results.benchmark_returns)

        annual_df = pd.DataFrame({
            'Strategy': annual,
            'S&P 500': annual_bench
        })

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=annual_df.index.astype(str),
            y=annual_df['Strategy'],
            name='Momentum Strategy',
            marker_color='#667eea'
        ))

        fig2.add_trace(go.Bar(
            x=annual_df.index.astype(str),
            y=annual_df['S&P 500'],
            name='S&P 500',
            marker_color='#d1d5db'
        ))

        fig2.update_layout(
            barmode='group',
            yaxis_title="Annual Return",
            xaxis_title="Year",
            height=350,
            yaxis_tickformat='.0%',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            yaxis=dict(gridcolor='#f0f0f0'),
            margin=dict(l=60, r=40, t=60, b=40)
        )

        st.plotly_chart(fig2, use_container_width=True)

    # =========================================================================
    # TAB 2: SIGNAL ANALYSIS
    # =========================================================================
    with tab_signal:
        st.markdown("### Current Momentum Rankings")

        st.markdown("""
        <div class="info-box">
            <strong>Momentum Signal (12-1):</strong> Calculated as the 12-month return excluding
            the most recent month. This avoids short-term mean reversion effects that can
            contaminate the momentum signal.
        </div>
        """, unsafe_allow_html=True)

        # Calculate momentum
        momentum = compute_momentum_signal(prices, lookback, skip)

        # Get valid momentum data (rows where at least one stock has data)
        valid_momentum = momentum.dropna(how='all')

        if valid_momentum.empty:
            st.warning("Not enough data to compute momentum signal. Try reducing the lookback period.")
            last_momentum = pd.Series(dtype=float)
            last_date = None
        else:
            last_date = valid_momentum.index[-1]
            last_momentum = momentum.loc[last_date].dropna().sort_values(ascending=False)

        col1, col2 = st.columns(2)

        if not last_momentum.empty:
            with col1:
                st.markdown("#### Long Positions (Winners)")
                long_stocks = last_momentum.head(n_long)

                for ticker, mom in long_stocks.items():
                    color_class = "positive" if mom > 0 else "negative"
                    st.markdown(f"""
                    <div class="stock-item">
                        <span class="stock-ticker">{ticker}</span>
                        <span class="stock-momentum-{color_class}">{format_percentage(mom)}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                if n_short > 0:
                    st.markdown("#### Short Positions (Losers)")
                    short_stocks = last_momentum.tail(n_short)

                    for ticker, mom in short_stocks.items():
                        color_class = "positive" if mom > 0 else "negative"
                        st.markdown(f"""
                        <div class="stock-item">
                            <span class="stock-ticker">{ticker}</span>
                            <span class="stock-momentum-{color_class}">{format_percentage(mom)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("#### Long-Only Strategy")
                    st.info("Short selling is disabled. The strategy only takes long positions in high-momentum stocks.")

        # Distribution chart
        if not last_momentum.empty:
            st.markdown("### Momentum Distribution")

            fig3 = go.Figure()

            fig3.add_trace(go.Histogram(
                x=last_momentum.values,
                nbinsx=15,
                marker_color='#667eea',
                opacity=0.8,
                hovertemplate='Momentum: %{x:.1%}<br>Count: %{y}<extra></extra>'
            ))

            fig3.update_layout(
                xaxis_title="Momentum Score",
                yaxis_title="Number of Stocks",
                height=300,
                xaxis_tickformat='.0%',
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                yaxis=dict(gridcolor='#f0f0f0'),
                margin=dict(l=60, r=40, t=20, b=40)
            )

            st.plotly_chart(fig3, use_container_width=True)

            if last_date is not None:
                st.caption(f"Data as of {last_date.strftime('%B %Y')}")

    # =========================================================================
    # TAB 3: RISK METRICS
    # =========================================================================
    with tab_risk:
        st.markdown("### Drawdown Analysis")

        drawdowns = compute_drawdowns(results.cumulative_returns)

        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values,
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.2)',
            line=dict(color='#ef4444', width=1.5),
            name='Drawdown',
            hovertemplate='%{x}<br>Drawdown: %{y:.1%}<extra></extra>'
        ))

        fig4.add_hline(
            y=-0.2,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="Bear Market Threshold (-20%)",
            annotation_position="bottom right"
        )

        fig4.update_layout(
            yaxis_title="Drawdown from Peak",
            xaxis_title="",
            height=350,
            yaxis_tickformat='.0%',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            yaxis=dict(gridcolor='#f0f0f0'),
            margin=dict(l=60, r=40, t=20, b=40)
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <strong>Understanding Drawdowns:</strong> A drawdown measures the decline from a
            historical peak. It represents the maximum loss an investor would have experienced
            if they bought at the peak and sold at the trough.
        </div>
        """, unsafe_allow_html=True)

        # Risk metrics cards
        st.markdown("### Risk-Adjusted Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Max Drawdown", format_percentage(metrics.max_drawdown))
        with col2:
            st.metric("Sortino Ratio", format_ratio(metrics.sortino_ratio))
        with col3:
            st.metric("Calmar Ratio", format_ratio(metrics.calmar_ratio))
        with col4:
            st.metric("Hit Rate", format_percentage(metrics.hit_rate))

        # Turnover analysis
        st.markdown("### Portfolio Turnover & Costs")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Avg Monthly Turnover", format_percentage(metrics.avg_turnover, 1))
        with col2:
            st.metric("Total Transaction Costs", format_percentage(metrics.total_transaction_costs))
        with col3:
            annual_cost = metrics.total_transaction_costs / (len(results.strategy_returns) / 12)
            st.metric("Annualized Cost Impact", format_percentage(annual_cost))

    # =========================================================================
    # TAB 4: ML MODEL (if enabled)
    # =========================================================================
    if tab_ml is not None and ml_model is not None:
        with tab_ml:
            st.markdown("### Machine Learning Regime Filter")

            st.markdown("""
            <div class="info-box">
                <strong>About the ML Model:</strong> A Logistic Regression classifier trained to
                predict market regime (Risk-On vs Risk-Off) using momentum, volatility, and
                trend features. The model uses expanding window training to avoid look-ahead bias.
            </div>
            """, unsafe_allow_html=True)

            # Model metrics
            st.markdown("### Model Performance")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Train Accuracy", format_percentage(ml_model.metrics['train_accuracy'], 1))
            with col2:
                st.metric("Test Accuracy", format_percentage(ml_model.metrics['test_accuracy'], 1))
            with col3:
                st.metric("Train F1 Score", format_percentage(ml_model.metrics['train_f1'], 1))
            with col4:
                st.metric("Test F1 Score", format_percentage(ml_model.metrics['test_f1'], 1))

            # Feature importance
            st.markdown("### Feature Importance")

            importance = get_feature_importance(ml_model)

            fig_imp = go.Figure()

            fig_imp.add_trace(go.Bar(
                x=importance['importance'],
                y=importance['feature'],
                orientation='h',
                marker_color='#667eea'
            ))

            fig_imp.update_layout(
                xaxis_title="Absolute Coefficient",
                yaxis_title="",
                height=350,
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                margin=dict(l=120, r=40, t=20, b=40)
            )

            st.plotly_chart(fig_imp, use_container_width=True)

            # Current prediction
            st.markdown("### Current Regime Prediction")

            from src.ml_regime import compute_features, predict_regime

            try:
                features = compute_features(benchmark_daily.iloc[:, 0])
                predictions = predict_regime(features, ml_model)

                last_proba = predictions['probability'].iloc[-1]
                last_regime = "RISK-ON" if last_proba > 0.5 else "RISK-OFF"

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Risk-On Probability", format_percentage(last_proba, 1))
                with col2:
                    if last_regime == "RISK-ON":
                        st.success(f"Current Regime: {last_regime}")
                    else:
                        st.error(f"Current Regime: {last_regime}")

                # Probability chart
                fig_proba = go.Figure()

                fig_proba.add_trace(go.Scatter(
                    x=predictions.index,
                    y=predictions['probability'],
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)',
                    line=dict(color='#667eea', width=1.5),
                    name='Risk-On Probability',
                    hovertemplate='%{x}<br>Probability: %{y:.1%}<extra></extra>'
                ))

                fig_proba.add_hline(
                    y=0.5,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="Decision Threshold"
                )

                fig_proba.update_layout(
                    yaxis_title="Risk-On Probability",
                    xaxis_title="",
                    height=300,
                    yaxis_tickformat='.0%',
                    yaxis_range=[0, 1],
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    yaxis=dict(gridcolor='#f0f0f0'),
                    margin=dict(l=60, r=40, t=20, b=40)
                )

                st.plotly_chart(fig_proba, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not generate current predictions: {str(e)}")

    # =========================================================================
    # TAB: METHODOLOGY
    # =========================================================================
    with tab_method:
        st.markdown("### Research Methodology")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            #### The Momentum Effect

            Cross-sectional momentum is one of the most robust anomalies in financial markets.
            First documented by **Jegadeesh & Titman (1993)**, the effect shows that stocks
            with strong recent performance tend to continue outperforming over the medium term.

            This strategy implements a systematic approach to capturing this premium.

            #### Signal Construction

            The **12-1 momentum signal** measures the return over the past 12 months,
            excluding the most recent month:

            ```
            Momentum(t) = Price(t-1) / Price(t-12) - 1
            ```

            **Why skip the last month?** Short-term returns often exhibit mean reversion
            (the "reversal effect"), which can contaminate the momentum signal.

            #### Portfolio Construction

            Each month, the strategy:
            1. Computes momentum for all 30 Dow Jones stocks
            2. Ranks stocks by momentum score
            3. Goes **LONG** the top performers
            4. Goes **SHORT** the bottom performers (optional)
            5. Equal-weights all positions

            #### Regime Filter

            The SMA 200 filter is a classic trend-following rule:
            - **Risk-On**: S&P 500 price > 200-day moving average
            - **Risk-Off**: S&P 500 price < 200-day moving average

            During Risk-Off periods, the strategy reduces exposure to mitigate drawdowns.
            """)

        with col2:
            st.markdown("""
            #### Key Parameters

            | Parameter | Value |
            |-----------|-------|
            | Universe | Dow Jones 30 |
            | Lookback | 12 months |
            | Skip Period | 1 month |
            | Rebalancing | Monthly |
            | Weighting | Equal-weight |
            | Costs | 10 bps/trade |

            #### Limitations

            - **Survivorship bias**: Uses current index constituents
            - **Short costs**: Borrowing fees not modeled
            - **Slippage**: Execution costs not included
            - **Capacity**: Limited by small universe

            #### References

            - Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*
            - Faber, M. (2007). *A Quantitative Approach to Tactical Asset Allocation*
            - Asness, C. et al. (2013). *Value and Momentum Everywhere*
            """)

    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("""
    <div class="footer">
        <strong>Cross-Sectional Momentum Strategy</strong><br>
        Quantitative Research Project | Data: Yahoo Finance<br>
        <a href="https://github.com/alexcanc/Cross-Sectional-Momentum-actions-US" target="_blank">View Source Code</a>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
