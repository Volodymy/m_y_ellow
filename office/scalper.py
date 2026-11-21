"""Високоризиковий короткостроковий трейдинг з плечем (тільки бектест і папір).

Модель ф'ючерсу USDⓈ-M з ізольованою маржею:
- вхід на відкритті свічки після сигналу, ринковим ордером (комісія taker);
- тейк-профіт і стоп-лос задаються у % від МАРЖІ, тобто рух ціни = ціль / плече;
- ліквідація, якщо ціна пройшла проти позиції (1/плече − підтримуюча маржа);
- якщо в одній свічці досягнуто і стопу, і тейку, рахуємо стоп (песимістично);
- фандинг списується як витрата кожні 8 годин утримання.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import USER_AGENT

CACHE_DIR = Path("data/cache")
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


def load_ohlc(symbol: str, interval: str = "15m", days: int = 365) -> pd.DataFrame:
    """Свічки зі спотового ринку Binance (ціни perp-ф'ючерсів дуже близькі до споту)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}_{interval}_{days}d.csv"
    if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    start = int((time.time() - days * 86400) * 1000)
    rows = []
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "startTime": start, "limit": 1000},
                         headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        start = batch[-1][0] + INTERVAL_MS[interval]
    df = pd.DataFrame([[float(x) for x in row[1:5]] for row in rows],
                      columns=["open", "high", "low", "close"],
                      index=pd.to_datetime([row[0] for row in rows], unit="ms"))
    df.to_csv(path)
    return df


def synthetic_ohlc(n: int = 20_000, seed: int = 0, vol: float = 0.003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vol, n)) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close},
                        index=pd.date_range("2025-01-01", periods=n, freq="15min"))


@dataclass
class ScalpParams:
    leverage: float = 20
    take_profit: float = 0.10     # +10% від маржі
    stop_loss: float = 0.10       # −10% від маржі
    margin_fraction: float = 0.10 # скільки капіталу ставимо маржею в одну угоду
    taker_fee: float = 0.0005     # 0.05% від номіналу за кожну сторону (Binance Futures)
    funding_8h: float = 0.0001    # 0.01% номіналу кожні 8 годин
    maintenance_margin: float = 0.005
    max_hold_bars: int = 96       # примусове закриття через 24 год на 15m


@dataclass
class ScalpResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def simulate(df: pd.DataFrame, signals: np.ndarray, p: ScalpParams, capital: float = 1000.0,
             bar_hours: float = 0.25) -> ScalpResult:
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(df)
    tp_move = p.take_profit / p.leverage
    sl_move = p.stop_loss / p.leverage
    liq_move = 1 / p.leverage - p.maintenance_margin
    stop_move = min(sl_move, liq_move)
    stop_is_liq = liq_move <= sl_move

    equity = capital
    curve = np.full(n, np.nan)
    trades = []
    t = 0
    while t < n - 1:
        curve[t] = equity
        if signals[t] == 0 or equity <= capital * 0.01:
            t += 1
            continue
        side = int(signals[t])
        entry_i = t + 1
        entry = o[entry_i]
        margin = equity * p.margin_fraction
        notional = margin * p.leverage
        tp_px = entry * (1 + side * tp_move)
        sl_px = entry * (1 - side * stop_move)

        exit_px, reason, j = None, "time", entry_i
        for j in range(entry_i, min(n, entry_i + p.max_hold_bars)):
            hit_sl = l[j] <= sl_px if side == 1 else h[j] >= sl_px
            hit_tp = h[j] >= tp_px if side == 1 else l[j] <= tp_px
            if hit_sl:
                exit_px, reason = sl_px, ("liquidation" if stop_is_liq else "stop")
                break
            if hit_tp:
                exit_px, reason = tp_px, "take_profit"
                break
        if exit_px is None:
            exit_px = c[j]

        move = side * (exit_px / entry - 1)
        held_h = (j - entry_i + 1) * bar_hours
        fees = notional * p.taker_fee * 2
        funding = notional * p.funding_8h * (held_h // 8)
        pnl = -margin if reason == "liquidation" else notional * move
        pnl -= fees + funding
        pnl = max(pnl, -margin - fees)  # ізольована маржа: втрачаємо не більше маржі (+ комісії)
        equity += pnl
        trades.append({"entry_time": df.index[entry_i], "exit_time": df.index[j], "side": "LONG" if side == 1 else "SHORT",
                       "entry": entry, "exit": exit_px, "reason": reason, "pnl": pnl,
                       "pnl_pct_margin": pnl / margin, "fees": fees + funding})
        for k in range(t + 1, j + 1):
            curve[k] = curve[t]
        t = j + 1
    curve[n - 1] = equity
    eq = pd.Series(curve, index=df.index).ffill()
    tr = pd.DataFrame(trades)
    return ScalpResult(eq, tr, stats(eq, tr, capital))


def stats(eq: pd.Series, tr: pd.DataFrame, capital: float) -> dict:
    if tr.empty:
        return {"trades": 0, "total_return": 0.0}
    dd = (eq / eq.cummax() - 1).min()
    wins = tr["pnl"] > 0
    return {
        "trades": len(tr),
        "win_rate": wins.mean(),
        "avg_win_pct_margin": tr.loc[wins, "pnl_pct_margin"].mean() if wins.any() else 0.0,
        "avg_loss_pct_margin": tr.loc[~wins, "pnl_pct_margin"].mean() if (~wins).any() else 0.0,
        "liquidations": int((tr["reason"] == "liquidation").sum()),
        "fees_paid": tr["fees"].sum(),
        "total_return": eq.iloc[-1] / capital - 1,
        "max_drawdown": dd,
        "breakeven_win_rate": breakeven(tr),
    }


def breakeven(tr: pd.DataFrame) -> float:
    w = tr.loc[tr["pnl"] > 0, "pnl_pct_margin"].mean()
    lo = -tr.loc[tr["pnl"] <= 0, "pnl_pct_margin"].mean()
    if not np.isfinite(w) or not np.isfinite(lo) or (w + lo) == 0:
        return float("nan")
    return lo / (w + lo)


def random_signals(n: int, count: int, seed: int) -> np.ndarray:
    """Контрольна група: стільки ж входів, але в випадкові моменти й випадковий бік."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    idx = rng.choice(np.arange(n - 1), size=min(count, n - 1), replace=False)
    s[idx] = rng.choice([-1, 1], size=len(idx))
    return s
