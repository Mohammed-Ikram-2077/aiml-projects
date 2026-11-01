"""Central project configuration and paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODEL_NAME = "ProsusAI/finbert"

TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple", "Apple Inc", "iPhone maker"),
    "TSLA": ("Tesla", "Tesla Inc", "Musk's automaker"),
    "MSFT": ("Microsoft", "Microsoft Corp"),
    "NVDA": ("Nvidia", "NVIDIA Corp"),
    "AMZN": ("Amazon", "Amazon.com"),
    "RELIANCE.NS": ("Reliance Industries", "Reliance"),
    "JPL.NS": ("Jio Platforms", "Jio", "Reliance Jio"),
    "TCS.NS": ("Tata Consultancy Services", "TCS"),
    "INFY.NS": ("Infosys", "Infosys Limited"),
    "HDFCBANK.NS": ("HDFC Bank", "HDFC"),
}


@dataclass(frozen=True)
class ProjectConfig:
    """Settings shared by the CLI pipeline and modules."""

    tickers: tuple[str, ...] = ("AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "RELIANCE.NS", "JPL.NS")
    # Stay comfortably within GDELT DOC's rolling three-month search window.
    months: int = 2
    batch_size: int = 16
    request_timeout: int = 20
    transaction_cost_bps: float = 10.0
    sentiment_threshold: float = 0.10
    random_seed: int = 42

    def date_range(self, end: date | None = None) -> tuple[date, date]:
        """Return a calendar date range of the configured approximate month length."""
        last = end or date.today()
        first = last - timedelta(days=round(self.months * 30.4375))
        return first, last


DEFAULT_CONFIG = ProjectConfig()
