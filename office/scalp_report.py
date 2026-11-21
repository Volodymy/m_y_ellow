"""Звіт для високоризикової стратегії: агенти vs випадкові входи, сітка плеча й цілей."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .patterns import DoubleTopBottomAgent, FibonacciAgent, combine
from .scalper import ScalpParams, load_ohlc, random_signals, simulate, synthetic_ohlc

BAR_HOURS = {"5m": 5 / 60, "15m": 0.25, "1h": 1.0, "4h": 4.0}


def agent_signals(df: pd.DataFrame, mode: str) -> np.ndarray:
    fib = FibonacciAgent().signals(df)
    pat = DoubleTopBottomAgent().signals(df)
    return {"fib": fib, "pattern": pat, "any": combine([fib, pat], "any"),
            "both": combine([fib, pat], "both")}[mode]


def run(symbols: list[str], interval: str, days: int, mode: str, leverages: list[float],
        targets: list[float], stop_ratio: float, synthetic: bool, seeds: int = 20,
        oos_fraction: float = 0.3) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df = synthetic_ohlc(seed=hash(sym) % 1000) if synthetic else load_ohlc(sym, interval, days)
        split = int(len(df) * (1 - oos_fraction))
        sig_all = agent_signals(df, mode)
        for part, sl in (("in-sample", slice(0, split)), ("out-of-sample", slice(split, None))):
            d, s = df.iloc[sl], sig_all[sl]
            for lev in leverages:
                for tp in targets:
                    p = ScalpParams(leverage=lev, take_profit=tp, stop_loss=tp * stop_ratio)
                    res = simulate(d, s, p, bar_hours=BAR_HOURS[interval])
                    n = res.stats.get("trades", 0)
                    rnd = [simulate(d, random_signals(len(d), max(n, 1), k), p,
                                    bar_hours=BAR_HOURS[interval]).stats for k in range(seeds)]
                    rows.append({
                        "symbol": sym, "period": part, "leverage": lev, "tp_%margin": tp,
                        "trades": n,
                        "win_rate": res.stats.get("win_rate", np.nan),
                        "need_win_rate": res.stats.get("breakeven_win_rate", np.nan),
                        "return": res.stats.get("total_return", 0.0),
                        "max_dd": res.stats.get("max_drawdown", 0.0),
                        "liquidations": res.stats.get("liquidations", 0),
                        "fees": res.stats.get("fees_paid", 0.0),
                        "random_median_return": float(np.median([r.get("total_return", 0) for r in rnd])),
                        "random_win_rate": float(np.nanmedian([r.get("win_rate", np.nan) for r in rnd])),
                    })
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("win_rate", "need_win_rate", "return", "max_dd", "random_median_return", "random_win_rate"):
        out[c] = out[c].map(lambda v: "—" if pd.isna(v) else f"{v:+.1%}" if c in ("return", "max_dd", "random_median_return") else f"{v:.0%}")
    out["tp_%margin"] = out["tp_%margin"].map(lambda v: f"{v:.0%}")
    out["leverage"] = out["leverage"].map(lambda v: f"x{v:g}")
    out["fees"] = out["fees"].map(lambda v: f"${v:,.0f}")
    out = out.rename(columns={
        "symbol": "актив", "period": "період", "leverage": "плече", "tp_%margin": "ціль",
        "trades": "угод", "win_rate": "виграшів", "need_win_rate": "треба для 0",
        "return": "дохід", "max_dd": "просадка", "liquidations": "ліквід.", "fees": "комісії",
        "random_median_return": "дохід випадк.", "random_win_rate": "виграшів випадк.",
    })
    return out.to_string(index=False)
