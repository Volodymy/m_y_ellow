"""Стратегії, змодельовані на підходах реальних трейдерів і фондів.

Кожна стратегія повертає історію цільових ваг усередині свого "рукава":
DataFrame (дата × тікер), рядок за дату t рахується лише з даних до t включно.
Сума ваг у рядку ≤ 1, решта — кеш.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ANNUAL_DAYS = 365  # календар у даних — щоденний (див. data.load_prices)


class Strategy:
    name: str = "base"
    backtestable: bool = True
    description: str = ""

    def symbols(self) -> list[str]:
        raise NotImplementedError

    def target_history(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class CryptoTrend(Strategy):
    """Трендова стратегія в стилі Turtle / CTA (Dunn, Winton, AHL).

    Вхід — пробій максимуму за entry_lookback днів, вихід — пробій мінімуму за
    exit_lookback днів. Розмір позиції обернено пропорційний волатильності,
    щоб кожен актив давав приблизно однаковий ризик.
    """

    universe: list[str]
    entry_lookback: int = 55
    exit_lookback: int = 20
    vol_lookback: int = 20
    target_vol: float = 0.40
    name: str = "crypto_trend"
    description: str = "Пробій каналу Дончіана + розмір позиції за волатильністю"

    def symbols(self):
        return list(self.universe)

    def target_history(self, prices):
        px = prices[self.universe]
        upper = px.rolling(self.entry_lookback).max().shift(1)
        lower = px.rolling(self.exit_lookback).min().shift(1)
        entry = (px > upper).to_numpy()
        exit_ = (px < lower).to_numpy()

        in_pos = np.zeros(px.shape, dtype=bool)
        state = np.zeros(px.shape[1], dtype=bool)
        for t in range(px.shape[0]):
            state = np.where(entry[t], True, np.where(exit_[t], False, state))
            in_pos[t] = state

        vol = np.log(px).diff().rolling(self.vol_lookback).std() * np.sqrt(ANNUAL_DAYS)
        size = (self.target_vol / vol).clip(upper=1.0).fillna(0.0)
        w = size.where(in_pos, 0.0) / len(self.universe)
        return w.fillna(0.0)


@dataclass
class DualMomentum(Strategy):
    """Подвійний моментум (Gary Antonacci) — ротація ETF раз на місяць.

    Купуємо top_n найсильніших ETF за 12 місяців, але лише якщо їхня
    дохідність вища за безпечний актив (короткі облігації). Інакше — у безпечний актив.
    """

    risky: list[str]
    safe: str = "SHY"
    lookback_days: int = 365
    top_n: int = 2
    name: str = "etf_dual_momentum"
    description: str = "Відносний + абсолютний моментум, щомісячна ребалансування"

    def symbols(self):
        return list(self.risky) + [self.safe]

    def target_history(self, prices):
        cols = self.symbols()
        px = prices[cols]
        mom = px / px.shift(self.lookback_days) - 1
        month_ends = px.groupby(px.index.to_period("M")).tail(1).index

        w = pd.DataFrame(np.nan, index=px.index, columns=cols)
        for d in month_ends:
            m = mom.loc[d]
            if m[self.risky].isna().all():
                continue
            row = pd.Series(0.0, index=cols)
            ranked = m[self.risky].dropna().sort_values(ascending=False).head(self.top_n)
            safe_ret = m[self.safe] if not np.isnan(m[self.safe]) else 0.0
            for sym, r in ranked.items():
                row[sym if r > safe_ret else self.safe] += 1.0 / self.top_n
            w.loc[d] = row
        return w.ffill().fillna(0.0)


@dataclass
class SuperinvestorClone(Strategy):
    """Копіювання портфелів відомих фондів зі звітів 13F.

    Бектестити не можна: у нас є лише поточний знімок портфеля, а прогін
    на минулих даних із сьогоднішнім портфелем дав би "заглядання в майбутнє".
    Для чесного бектесту потрібні історичні 13F за кожен квартал.
    """

    holdings_file: str
    funds: list[str]
    top_n: int = 10
    name: str = "superinvestor_clone"
    backtestable: bool = False
    description: str = "Консенсус-портфель фондів із 13F (затримка до 45 днів)"

    def weights(self) -> pd.Series:
        data = yaml.safe_load(Path(self.holdings_file).read_text())
        parts = []
        for fund in self.funds:
            h = {k: v for k, v in (data[fund]["holdings"] or {}).items() if v is not None}
            s = pd.Series(h, dtype=float)
            if s.sum() > 0:
                parts.append(s / s.sum())
        combined = pd.concat(parts, axis=1).fillna(0.0).mean(axis=1)
        top = combined.sort_values(ascending=False).head(self.top_n)
        return top / top.sum()

    def symbols(self):
        return list(self.weights().index)

    def target_history(self, prices):
        w = self.weights()
        return pd.DataFrame([w.values] * len(prices), index=prices.index, columns=w.index)


def build_strategies(cfg: dict) -> dict[str, Strategy]:
    s = cfg["strategies"]
    built: dict[str, Strategy] = {}
    if "crypto_trend" in cfg["allocation"]:
        built["crypto_trend"] = CryptoTrend(**s["crypto_trend"])
    if "etf_dual_momentum" in cfg["allocation"]:
        built["etf_dual_momentum"] = DualMomentum(**s["etf_dual_momentum"])
    if "superinvestor_clone" in cfg["allocation"]:
        built["superinvestor_clone"] = SuperinvestorClone(**s["superinvestor_clone"])
    return built
