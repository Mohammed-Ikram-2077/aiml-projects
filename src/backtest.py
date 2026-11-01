"""Simple long/flat sentiment backtest with next-session execution and costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _metrics(returns: pd.Series) -> dict[str, Any]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    std = returns.std(ddof=1)
    return {
        "cumulative_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "annualized_sharpe": float(np.sqrt(252) * returns.mean() / std) if len(returns) > 1 and std > 0 else None,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def run_backtest(aligned: pd.DataFrame, *, threshold: float = 0.10, transaction_cost_bps: float = 10.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply sentiment known at t to return t+1, charging costs when position changes."""
    if aligned.empty or "mean_sentiment" not in aligned:
        return pd.DataFrame(), {"status": "insufficient data", "reason": "no aligned sentiment and prices"}
    frame = aligned.sort_values(["ticker", "date"]).copy()
    frame["position"] = (pd.to_numeric(frame["mean_sentiment"], errors="coerce") >= threshold).astype(float)
    frame["previous_position"] = frame.groupby("ticker")["position"].shift(1).fillna(0.0)
    frame["turnover"] = (frame["position"] - frame["previous_position"]).abs()
    frame["transaction_cost"] = frame["turnover"] * (transaction_cost_bps / 10_000.0)
    frame["strategy_return"] = frame["position"] * pd.to_numeric(frame["return_t_plus_1"], errors="coerce") - frame["transaction_cost"]
    frame["buy_hold_return"] = pd.to_numeric(frame["return_t_plus_1"], errors="coerce")
    per_ticker: dict[str, Any] = {}
    for ticker, group in frame.groupby("ticker"):
        per_ticker[str(ticker)] = {"strategy": _metrics(group["strategy_return"].dropna()), "buy_and_hold": _metrics(group["buy_hold_return"].dropna())}
    return frame, {"status": "ok", "transaction_cost_bps_per_position_change": transaction_cost_bps, "threshold": threshold, "by_ticker": per_ticker}
