"""Бектест офісу: рішення за закриттям дня t виконуються за закриттям дня t+1.

Так модель не "бачить майбутнього": сигнал рахується з учорашніх даних,
а угода проходить за сьогоднішньою ціною з комісією й проковзуванням.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agents import Analyst, PaperBroker, RiskManager

ANNUAL_DAYS = 365


def metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(ANNUAL_DAYS)
    sharpe = rets.mean() / rets.std() * np.sqrt(ANNUAL_DAYS) if rets.std() > 0 else np.nan
    dd = equity / equity.cummax() - 1
    return {
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
    }


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: int
    fees: float
    in_sample: dict
    out_of_sample: dict
    full: dict
    split_date: pd.Timestamp
    benchmarks: dict
    halted_on: str | None


def run_backtest(prices: pd.DataFrame, analyst: Analyst, risk: RiskManager, costs: dict,
                 capital: float, oos_fraction: float = 0.3, warmup: int = 365) -> BacktestResult:
    targets, _ = analyst.target_history(prices)
    targets = targets.shift(1).fillna(0.0)  # рішення вчора → виконання сьогодні
    prices = prices[targets.columns]

    broker = PaperBroker(cash=capital, costs=costs)
    risk.reset()
    equity, halted_on = {}, None
    dates = prices.index[warmup:]
    for d in dates:
        px = prices.loc[d]
        if px.isna().any():
            px = px.fillna(0.0)
        eq = broker.equity(px)
        tgt = risk.check(targets.loc[d].where(px > 0, 0.0), eq)
        if risk.halted and halted_on is None:
            halted_on = str(d.date())
        broker.rebalance(tgt, px, d, risk)
        equity[d] = broker.equity(px)

    eq = pd.Series(equity)
    split = eq.index[int(len(eq) * (1 - oos_fraction))]
    bench = {}
    for sym in ("SPY", "BTCUSDT"):
        if sym in prices.columns:
            b = prices.loc[dates, sym]
            bench[f"buy&hold {sym}"] = metrics(b / b.iloc[0] * capital)
    return BacktestResult(
        equity=eq,
        trades=len(broker.trades),
        fees=sum(t.fee for t in broker.trades),
        in_sample=metrics(eq[:split]),
        out_of_sample=metrics(eq[split:]),
        full=metrics(eq),
        split_date=split,
        benchmarks=bench,
        halted_on=halted_on,
    )


def format_result(r: BacktestResult, title: str = "Бектест") -> str:
    def row(name, m):
        if not m:
            return f"{name:<28} немає даних"
        return (f"{name:<28} дохід {m['total_return']:+8.1%} | CAGR {m['cagr']:+7.1%} | "
                f"Sharpe {m['sharpe']:5.2f} | макс. просадка {m['max_drawdown']:7.1%}")

    out = [f"=== {title}: {r.equity.index[0].date()} → {r.equity.index[-1].date()} ===",
           row("Увесь період", r.full),
           row(f"До {r.split_date.date()} (in-sample)", r.in_sample),
           row("Після (out-of-sample)", r.out_of_sample)]
    for k, m in r.benchmarks.items():
        out.append(row(k, m))
    out.append(f"Угод: {r.trades}, комісії: ${r.fees:,.2f}")
    if r.halted_on:
        out.append(f"⚠️ Аварійна зупинка спрацювала {r.halted_on}")
    return "\n".join(out)
