"""
Configuration parameters for the Cross-Sectional Momentum Strategy.

All strategy parameters are centralized here for transparency and reproducibility.
No parameter optimization is performed - we use standard academic values.
"""

from dataclasses import dataclass
from typing import List
import datetime


@dataclass
class Config:
    """
    Strategy configuration.

    These parameters follow academic standards (Jegadeesh & Titman, 1993).
    We explicitly avoid optimization to prevent overfitting.
    """

    # ==========================================================================
    # UNIVERSE DEFINITION
    # ==========================================================================

    # Dow Jones 30 constituents (as of 2024)
    # Note: Using current constituents introduces survivorship bias
    # This is acknowledged as a limitation
    DOW_JONES_30: List[str] = None

    # S&P 100 alternative (optional, for future expansion)
    USE_SP100: bool = False

    # Benchmark
    BENCHMARK_TICKER: str = "SPY"  # S&P 500 ETF

    # ==========================================================================
    # DATA PARAMETERS
    # ==========================================================================

    # Date range for backtest
    START_DATE: str = "2005-01-01"
    END_DATE: str = "2024-12-31"

    # Data frequency
    FREQUENCY: str = "monthly"  # 'daily' or 'monthly'

    # ==========================================================================
    # MOMENTUM SIGNAL PARAMETERS
    # ==========================================================================

    # Lookback period for momentum calculation (in months)
    MOMENTUM_LOOKBACK: int = 12

    # Skip period (most recent month to exclude)
    # Rationale: Short-term mean reversion contaminates momentum signal
    MOMENTUM_SKIP: int = 1

    # Effective momentum window: t-12 to t-1 (12-1 momentum)

    # ==========================================================================
    # PORTFOLIO CONSTRUCTION
    # ==========================================================================

    # Number of stocks in long portfolio
    N_LONG: int = 10

    # Number of stocks in short portfolio
    N_SHORT: int = 10

    # Portfolio type: 'long_short' or 'long_only'
    PORTFOLIO_TYPE: str = "long_short"

    # Weighting scheme: 'equal_weight' or 'rank_weight'
    WEIGHTING: str = "equal_weight"

    # ==========================================================================
    # REBALANCING
    # ==========================================================================

    # Rebalancing frequency
    REBALANCE_FREQ: str = "monthly"  # 'monthly', 'quarterly'

    # ==========================================================================
    # TRANSACTION COSTS
    # ==========================================================================

    # Transaction cost per trade (one-way), in basis points
    # 10 bps is conservative for large-cap US equities
    TRANSACTION_COST_BPS: float = 10.0

    # Convert to decimal
    @property
    def transaction_cost(self) -> float:
        return self.TRANSACTION_COST_BPS / 10000

    # ==========================================================================
    # REGIME FILTER PARAMETERS
    # ==========================================================================

    # Moving average period for regime detection (in days)
    REGIME_MA_PERIOD: int = 200

    # Risk-off exposure multiplier (1.0 = full exposure, 0.0 = all cash)
    RISK_OFF_EXPOSURE: float = 0.5

    # ==========================================================================
    # RISK MANAGEMENT
    # ==========================================================================

    # Maximum position size (as fraction of portfolio)
    MAX_POSITION_SIZE: float = 0.20  # 20%

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __post_init__(self):
        """Initialize DOW_JONES_30 list after dataclass creation."""
        if self.DOW_JONES_30 is None:
            self.DOW_JONES_30 = [
                "AAPL",   # Apple
                "AMGN",   # Amgen
                "AXP",    # American Express
                "BA",     # Boeing
                "CAT",    # Caterpillar
                "CRM",    # Salesforce
                "CSCO",   # Cisco
                "CVX",    # Chevron
                "DIS",    # Disney
                "DOW",    # Dow Inc
                "GS",     # Goldman Sachs
                "HD",     # Home Depot
                "HON",    # Honeywell
                "IBM",    # IBM
                "INTC",   # Intel
                "JNJ",    # Johnson & Johnson
                "JPM",    # JPMorgan Chase
                "KO",     # Coca-Cola
                "MCD",    # McDonald's
                "MMM",    # 3M
                "MRK",    # Merck
                "MSFT",   # Microsoft
                "NKE",    # Nike
                "PG",     # Procter & Gamble
                "TRV",    # Travelers
                "UNH",    # UnitedHealth
                "V",      # Visa
                "VZ",     # Verizon
                "WBA",    # Walgreens
                "WMT",    # Walmart
            ]


def get_config() -> Config:
    """Return a new configuration instance."""
    return Config()
