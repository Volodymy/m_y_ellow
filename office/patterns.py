"""Агенти технічного аналізу: шукають рівні Фібоначчі та фігури на графіку.

Усі сигнали без заглядання в майбутнє: локальний максимум/мінімум (pivot)
на свічці i вважається відомим лише на свічці i + k, коли праворуч уже є k свічок.

Сигнал: +1 — купівля (long), -1 — продаж (short), 0 — нічого.
Сигнал на свічці t означає вхід на відкритті свічки t + 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Pivot:
    idx: int
    price: float
    kind: str  # "H" або "L"


def _pivot_stream(high, low, k):
    """Генератор: для кожної t повертає (t, pivots_known_so_far) без копіювання."""
    n = len(high)
    found: list[Pivot] = []
    for t in range(n):
        i = t - k
        if i >= k:
            if high[i] == high[i - k: i + k + 1].max():
                found.append(Pivot(i, float(high[i]), "H"))
            if low[i] == low[i - k: i + k + 1].min():
                found.append(Pivot(i, float(low[i]), "L"))
        yield t, found


def _last_swing(found: list[Pivot]):
    """Останні два pivot протилежного типу: (попередній, останній)."""
    if len(found) < 2:
        return None
    last = found[-1]
    for p in reversed(found[:-1]):
        if p.kind != last.kind:
            return p, last
    return None


@dataclass
class FibonacciAgent:
    """Відкат до «золотої зони» Фібоначчі 0.5–0.618 з відбоєм.

    Long: після хвилі вгору (мінімум → максимум) ціна відкочується в зону
    0.5–0.618, не пробиває 0.786, і свічка закривається зеленою вище 0.618.
    Short — дзеркально після хвилі вниз.
    """

    pivot_k: int = 5
    min_swing: float = 0.015   # мінімальний розмір хвилі, 1.5%
    zone_top: float = 0.5
    zone_bottom: float = 0.618
    invalid: float = 0.786
    name: str = "fibonacci"

    def signals(self, df: pd.DataFrame) -> np.ndarray:
        o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
        sig = np.zeros(len(df), dtype=int)
        used: set[tuple[int, int]] = set()
        for t, found in _pivot_stream(h, l, self.pivot_k):
            sw = _last_swing(found)
            if sw is None:
                continue
            a, b = sw
            key = (a.idx, b.idx)
            if key in used or t <= b.idx:
                continue
            rng = abs(b.price - a.price)
            if rng / min(a.price, b.price) < self.min_swing:
                continue
            if a.kind == "L" and b.kind == "H":  # хвиля вгору → шукаємо long
                top = b.price - self.zone_top * rng
                bottom = b.price - self.zone_bottom * rng
                inv = b.price - self.invalid * rng
                if l[t] < inv:
                    used.add(key)
                elif l[t] <= top and c[t] >= bottom and c[t] > o[t]:
                    sig[t] = 1
                    used.add(key)
            elif a.kind == "H" and b.kind == "L":  # хвиля вниз → шукаємо short
                top = b.price + self.zone_top * rng
                bottom = b.price + self.zone_bottom * rng
                inv = b.price + self.invalid * rng
                if h[t] > inv:
                    used.add(key)
                elif h[t] >= top and c[t] <= bottom and c[t] < o[t]:
                    sig[t] = -1
                    used.add(key)
        return sig


@dataclass
class DoubleTopBottomAgent:
    """Фігури «подвійне дно» і «подвійна вершина».

    Подвійне дно: два мінімуми на одному рівні (± tolerance) з максимумом між
    ними (лінія шиї). Сигнал long — закриття вище лінії шиї.
    Подвійна вершина — дзеркально, сигнал short.
    """

    pivot_k: int = 5
    tolerance: float = 0.003   # мінімуми/максимуми відрізняються не більше ніж на 0.3%
    max_age: int = 60          # фігура має сформуватися не довше ніж за 60 свічок
    name: str = "double_top_bottom"

    def signals(self, df: pd.DataFrame) -> np.ndarray:
        h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
        sig = np.zeros(len(df), dtype=int)
        used: set[int] = set()
        for t, found in _pivot_stream(h, l, self.pivot_k):
            if len(found) < 3 or t == 0:
                continue
            p1, p2, p3 = found[-3], found[-2], found[-1]
            if p3.idx in used or t - p1.idx > self.max_age:
                continue
            if p1.kind == "L" and p2.kind == "H" and p3.kind == "L":
                if abs(p1.price - p3.price) / p1.price <= self.tolerance:
                    if c[t] > p2.price and c[t - 1] <= p2.price:
                        sig[t] = 1
                        used.add(p3.idx)
            elif p1.kind == "H" and p2.kind == "L" and p3.kind == "H":
                if abs(p1.price - p3.price) / p1.price <= self.tolerance:
                    if c[t] < p2.price and c[t - 1] >= p2.price:
                        sig[t] = -1
                        used.add(p3.idx)
        return sig


def combine(signals: list[np.ndarray], mode: str, window: int = 3) -> np.ndarray:
    """any — будь-який агент дав сигнал; both — обидва в межах window свічок в один бік."""
    if mode == "any":
        out = np.zeros_like(signals[0])
        for s in signals:
            out = np.where(out == 0, s, out)
        return out
    if mode == "both":
        a, b = signals
        out = np.zeros_like(a)
        for t in range(len(a)):
            lo = max(0, t - window)
            if a[t] != 0 and (b[lo:t + 1] == a[t]).any():
                out[t] = a[t]
            elif b[t] != 0 and (a[lo:t + 1] == b[t]).any():
                out[t] = b[t]
        return out
    raise ValueError(mode)
