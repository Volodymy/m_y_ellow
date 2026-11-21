"""Джерела ринкових даних: щоденні ціни закриття.

- BinanceProvider: крипта, публічний API без ключів (data-api.binance.vision).
- YahooProvider: акції та ETF США.
- SyntheticProvider: згенеровані ціни для тестів і роботи без інтернету.

Усі провайдери повертають DataFrame: індекс — дата (UTC, без часу), колонки — тікери.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path("data/cache")
USER_AGENT = "ai-trading-office/0.1 (research; paper trading)"


def is_crypto(symbol: str) -> bool:
    return symbol.endswith("USDT")


class Provider:
    def history(self, symbols: list[str], start: str) -> pd.DataFrame:
        raise NotImplementedError


def _cached(name: str, fetch, max_age_hours: float = 12) -> pd.Series:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = name
        return s
    s = fetch()
    s.to_frame(name).to_csv(path)
    return s


class BinanceProvider(Provider):
    BASE = "https://data-api.binance.vision/api/v3/klines"

    def _fetch(self, symbol: str, start: str) -> pd.Series:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        rows: list[list] = []
        while True:
            r = requests.get(
                self.BASE,
                params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 1000},
                headers={"User-Agent": USER_AGENT},
                timeout=20,
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            start_ms = batch[-1][0] + 86_400_000
        idx = pd.to_datetime([row[0] for row in rows], unit="ms").normalize()
        return pd.Series([float(row[4]) for row in rows], index=idx, name=symbol)

    def history(self, symbols, start):
        return pd.concat([_cached(s, lambda s=s: self._fetch(s, start)) for s in symbols], axis=1)


class YahooProvider(Provider):
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def _fetch(self, symbol: str, start: str) -> pd.Series:
        p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        r = requests.get(
            self.BASE.format(symbol=symbol),
            params={"period1": p1, "period2": int(time.time()), "interval": "1d", "events": "div,split"},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        closes = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or res["indicators"]["quote"][0]["close"]
        idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
        return pd.Series(closes, index=idx, name=symbol, dtype=float).dropna()

    def history(self, symbols, start):
        return pd.concat([_cached(s, lambda s=s: self._fetch(s, start)) for s in symbols], axis=1)


class SyntheticProvider(Provider):
    """Геометричне броунівське блукання зі зміною режимів (тренд / флет / падіння).

    Потрібне лише для тестів і перевірки механіки. Результати на синтетиці
    нічого не кажуть про прибутковість на реальному ринку.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def history(self, symbols, start):
        idx = pd.date_range(start, pd.Timestamp.today().normalize(), freq="D")
        self.days = len(idx)
        out = {}
        for i, s in enumerate(symbols):
            rng = np.random.default_rng(self.seed + i)
            vol = 0.035 if is_crypto(s) else 0.012
            regimes = rng.choice([0.0015, 0.0, -0.001], size=self.days // 100 + 1, p=[0.4, 0.4, 0.2])
            drift = np.repeat(regimes, 100)[: self.days]
            rets = drift + rng.normal(0, vol, self.days)
            out[s] = 100 * np.exp(np.cumsum(rets))
        return pd.DataFrame(out, index=idx)


def load_prices(symbols: list[str], start: str, synthetic: bool = False) -> pd.DataFrame:
    """Завантажує ціни для змішаного списку (крипта + акції).

    Календарі різні (крипта торгується 7 днів на тиждень), тому дані
    об'єднуються за датами, а пропуски акцій на вихідних заповнюються
    останньою відомою ціною.
    """
    symbols = list(dict.fromkeys(symbols))
    if synthetic:
        prices = SyntheticProvider().history(symbols, start)
    else:
        crypto = [s for s in symbols if is_crypto(s)]
        stocks = [s for s in symbols if not is_crypto(s)]
        frames = []
        if crypto:
            frames.append(BinanceProvider().history(crypto, start))
        if stocks:
            frames.append(YahooProvider().history(stocks, start))
        prices = pd.concat(frames, axis=1)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    # Єдиний календар по днях: усі "lookback" у стратегіях — це календарні дні.
    full = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    return prices.reindex(full).ffill()
