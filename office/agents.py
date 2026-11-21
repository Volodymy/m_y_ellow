"""«Офіс» агентів. Кожен агент відповідає за одну роль:

  Analyst      — запускає стратегії й зводить їхні ваги в один цільовий портфель.
  RiskManager  — ріже ліміти, тримає кеш-буфер, вмикає аварійну зупинку.
  Executor     — паперовий брокер: виконує угоди з комісіями й проковзуванням.
  Reporter     — формує звіт (опційно з коментарем Claude і відправкою в Telegram).

Торгові рішення ухвалюються лише за правилами в коді. LLM тільки пояснює, що сталося.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import is_crypto
from .strategies import Strategy


# ---------------------------------------------------------------- Analyst
@dataclass
class Analyst:
    strategies: dict[str, Strategy]
    allocation: dict[str, float]
    backtest_only: bool = False  # True — пропускати стратегії, які не можна чесно бектестити

    def active(self) -> dict[str, Strategy]:
        return {
            k: s for k, s in self.strategies.items()
            if not (self.backtest_only and not s.backtestable)
        }

    def target_history(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        active = self.active()
        alloc_total = sum(self.allocation[k] for k in active) or 1.0
        sleeves = {k: s.target_history(prices) for k, s in active.items()}
        combined = pd.DataFrame(0.0, index=prices.index, columns=sorted({c for w in sleeves.values() for c in w.columns}))
        for k, w in sleeves.items():
            share = self.allocation[k] / alloc_total
            combined = combined.add(w.reindex(prices.index).fillna(0.0) * share, fill_value=0.0)
        return combined, sleeves


# ------------------------------------------------------------ RiskManager
@dataclass
class RiskManager:
    max_position_weight: float = 0.15
    max_gross_exposure: float = 1.0
    cash_buffer: float = 0.02
    max_drawdown_kill: float = 0.25
    rebalance_threshold: float = 0.02
    min_trade_value: float = 10.0
    halted: bool = False
    peak_equity: float = 0.0
    notes: list[str] = field(default_factory=list)

    def check(self, targets: pd.Series, equity: float) -> pd.Series:
        self.notes = []
        self.peak_equity = max(self.peak_equity, equity)
        dd = 1 - equity / self.peak_equity if self.peak_equity else 0.0
        if dd >= self.max_drawdown_kill and not self.halted:
            self.halted = True
            self.notes.append(f"АВАРІЙНА ЗУПИНКА: просадка {dd:.1%} ≥ {self.max_drawdown_kill:.0%}. Усе в кеш.")
        if self.halted:
            return targets * 0.0

        clipped = targets.clip(lower=0.0, upper=self.max_position_weight)
        for sym in targets.index[(targets > self.max_position_weight + 1e-12)]:
            self.notes.append(f"{sym}: вагу {targets[sym]:.1%} обрізано до {self.max_position_weight:.0%}")
        limit = self.max_gross_exposure - self.cash_buffer
        gross = clipped.sum()
        if gross > limit:
            clipped *= limit / gross
            self.notes.append(f"Загальну експозицію {gross:.1%} зменшено до {limit:.0%}")
        return clipped

    def reset(self):
        self.halted = False
        self.peak_equity = 0.0


# ---------------------------------------------------------- Executor
@dataclass
class Trade:
    date: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: float


@dataclass
class PaperBroker:
    cash: float
    costs: dict
    positions: dict[str, float] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    def _cost(self, symbol):
        return self.costs["crypto" if is_crypto(symbol) else "stock"]

    def equity(self, prices: pd.Series) -> float:
        return self.cash + sum(q * float(prices[s]) for s, q in self.positions.items() if q)

    def weights(self, prices: pd.Series) -> pd.Series:
        eq = self.equity(prices)
        return pd.Series({s: q * float(prices[s]) / eq for s, q in self.positions.items() if q}, dtype=float)

    def rebalance(self, targets: pd.Series, prices: pd.Series, date, risk: RiskManager) -> list[Trade]:
        eq = self.equity(prices)
        current = self.weights(prices)
        symbols = sorted(set(targets.index[targets > 0]) | set(current.index))
        orders = []
        for s in symbols:
            tw = float(targets.get(s, 0.0))
            cw = float(current.get(s, 0.0))
            full_exit = tw == 0 and cw > 0
            if not full_exit and abs(tw - cw) < risk.rebalance_threshold:
                continue
            value = (tw - cw) * eq
            if abs(value) < risk.min_trade_value and not full_exit:
                continue
            orders.append((s, value, full_exit))

        done = []
        # спершу продажі, щоб звільнити кеш під покупки
        for s, value, full_exit in sorted(orders, key=lambda o: o[1]):
            px = float(prices[s])
            c = self._cost(s)
            if value < 0:
                exec_px = px * (1 - c["slippage"])
                qty = self.positions.get(s, 0.0) if full_exit else min(-value / px, self.positions.get(s, 0.0))
                notional = qty * exec_px
                fee = notional * c["fee"]
                self.cash += notional - fee
                self.positions[s] = self.positions.get(s, 0.0) - qty
                if abs(self.positions[s]) < 1e-12:
                    self.positions.pop(s)
                side = "SELL"
            else:
                exec_px = px * (1 + c["slippage"])
                spend = min(value, self.cash / (1 + c["fee"]))
                if spend < risk.min_trade_value:
                    continue
                qty = spend / exec_px
                fee = spend * c["fee"]
                self.cash -= spend + fee
                self.positions[s] = self.positions.get(s, 0.0) + qty
                side = "BUY"
            t = Trade(str(pd.Timestamp(date).date()), s, side, qty, exec_px, fee)
            self.trades.append(t)
            done.append(t)
        return done

    # --- збереження стану між запусками (паперовий рахунок)
    def save(self, path: Path, risk: RiskManager):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "cash": self.cash,
            "positions": self.positions,
            "trades": [t.__dict__ for t in self.trades],
            "risk": {"halted": risk.halted, "peak_equity": risk.peak_equity},
        }, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path, costs: dict, starting_capital: float, risk: RiskManager) -> "PaperBroker":
        if not path.exists():
            return cls(cash=starting_capital, costs=costs)
        st = json.loads(path.read_text())
        risk.halted = st["risk"]["halted"]
        risk.peak_equity = st["risk"]["peak_equity"]
        return cls(cash=st["cash"], costs=costs, positions=st["positions"],
                   trades=[Trade(**t) for t in st["trades"]])


# ---------------------------------------------------------- Reporter
@dataclass
class Reporter:
    llm_commentary: bool = False
    telegram: bool = False

    def build(self, date, broker: PaperBroker, prices: pd.Series, new_trades: list[Trade],
              risk: RiskManager, starting_capital: float) -> str:
        eq = broker.equity(prices)
        lines = [
            f"📊 Звіт офісу за {pd.Timestamp(date).date()} (паперова торгівля)",
            f"Капітал: ${eq:,.2f} ({eq / starting_capital - 1:+.2%} від старту), кеш: ${broker.cash:,.2f}",
        ]
        w = broker.weights(prices).sort_values(ascending=False)
        if len(w):
            lines.append("Позиції: " + ", ".join(f"{s} {v:.1%}" for s, v in w.items()))
        else:
            lines.append("Позицій немає, усе в кеші.")
        if new_trades:
            lines.append("Нові угоди:")
            lines += [f"  {t.side} {t.symbol} {t.qty:.6g} @ {t.price:,.4g} (комісія ${t.fee:.2f})" for t in new_trades]
        else:
            lines.append("Нових угод немає.")
        lines += [f"⚠️ {n}" for n in risk.notes]
        text = "\n".join(lines)

        if self.llm_commentary:
            from .llm import commentary
            note = commentary(text)
            if note:
                text += "\n\n🧠 Коментар аналітика (Claude):\n" + note
        if self.telegram:
            from .notify import send_telegram
            send_telegram(text)
        return text
