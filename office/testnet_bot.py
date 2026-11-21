"""Бот для тестової мережі Binance Futures: Фібоначчі + фігури, плече, TP/SL на біржі.

Один цикл (`python -m office testnet --once`):
  1. Ризик-менеджер: рахує капітал, просадку, денний збиток.
  2. Якщо позиція відкрита — перевіряє час утримання і закриття перед вихідними.
  3. Якщо позиції немає — агенти аналізують останню ЗАКРИТУ свічку; при сигналі
     відкривається ринкова угода з ізольованою маржею, і одразу ставляться
     стоп-лос і тейк-профіт на сервері Binance.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .binance_futures import BinanceError, FuturesTestnet, qty_for, tp_sl_prices
from .patterns import DoubleTopBottomAgent, FibonacciAgent, combine

STATE = Path("data/testnet_state.json")
TRADES = Path("data/testnet_trades.csv")
HARD_MAX_LEVERAGE = 10


@dataclass
class BotConfig:
    symbol: str = "XAUUSDT"
    interval: str = "1h"
    leverage: int = 10
    take_profit: float = 0.30      # +30% від маржі
    stop_loss: float = 0.30        # −30% від маржі
    margin_fraction: float = 0.10  # 10% капіталу як маржа однієї угоди
    min_swing: float = 0.01
    max_hold_hours: int = 24
    daily_loss_limit: float = 0.05
    kill_drawdown: float = 0.20
    friday_close_utc_hour: int = 20


def _load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save_state(st: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2, ensure_ascii=False))


def _log_trade(row: dict):
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    new = not TRADES.exists()
    with TRADES.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def latest_signal(df, cfg: BotConfig) -> int:
    sig = combine([FibonacciAgent(min_swing=cfg.min_swing).signals(df), DoubleTopBottomAgent().signals(df)], "any")
    return int(sig[-1])


def run_once(cfg: BotConfig, client: FuturesTestnet | None = None) -> list[str]:
    log: list[str] = []
    c = client or FuturesTestnet()
    c.sync_time()
    now = datetime.now(timezone.utc)
    st = _load_state()
    lev = min(cfg.leverage, HARD_MAX_LEVERAGE)

    equity = c.wallet_usdt()
    today = now.date().isoformat()
    if st.get("day") != today:
        st["day"], st["day_start_equity"] = today, equity
    st["peak_equity"] = max(st.get("peak_equity", equity), equity)
    dd = 1 - equity / st["peak_equity"]
    day_pnl = equity / st["day_start_equity"] - 1
    log.append(f"Капітал {equity:,.2f} USDT | від піку {-dd:+.1%} | за день {day_pnl:+.1%}")
    if dd >= cfg.kill_drawdown and not st.get("halted"):
        st["halted"] = True
        log.append(f"⛔ АВАРІЙНА ЗУПИНКА: просадка {dd:.1%}. Скидання: python -m office testnet --reset-halt")

    pos = c.position(cfg.symbol)
    open_trade = st.get("open_trade")

    # Позицію закрив стоп або тейк на біржі
    if open_trade and not pos:
        c.cancel_all(cfg.symbol)
        pnl = equity - open_trade["equity_before"]
        _log_trade({**open_trade, "closed_at": now.isoformat(timespec="seconds"), "reason": "tp/sl на біржі",
                    "pnl_usdt": round(pnl, 4), "pnl_pct_margin": round(pnl / open_trade["margin"], 4)})
        log.append(f"✅ Угоду закрито біржею (TP/SL): {pnl:+.2f} USDT ({pnl / open_trade['margin']:+.1%} маржі)")
        st["open_trade"], open_trade = None, None

    if pos and open_trade:
        opened = datetime.fromisoformat(open_trade["opened_at"])
        held_h = (now - opened).total_seconds() / 3600
        friday = now.weekday() == 4 and now.hour >= cfg.friday_close_utc_hour
        if held_h >= cfg.max_hold_hours or friday or st.get("halted"):
            c.close_position(cfg.symbol)
            time.sleep(1)
            pnl = c.wallet_usdt() - open_trade["equity_before"]
            reason = "аварійна зупинка" if st.get("halted") else ("закриття перед вихідними" if friday else f"час ({held_h:.0f} год)")
            _log_trade({**open_trade, "closed_at": now.isoformat(timespec="seconds"), "reason": reason,
                        "pnl_usdt": round(pnl, 4), "pnl_pct_margin": round(pnl / open_trade["margin"], 4)})
            log.append(f"Закрито вручну ботом ({reason}): {pnl:+.2f} USDT")
            st["open_trade"] = None
        else:
            log.append(f"Позиція відкрита {open_trade['side']} {open_trade['qty']} @ {open_trade['entry']}, "
                       f"тримаємо {held_h:.1f} год. TP {open_trade['tp']} / SL {open_trade['sl']}")
        _save_state(st)
        return log

    if pos and not open_trade:
        log.append("⚠️ На біржі є позиція, якої бот не відкривав. Нічого не роблю.")
        _save_state(st)
        return log

    if st.get("halted"):
        _save_state(st)
        return log
    if day_pnl <= -cfg.daily_loss_limit:
        log.append(f"Денний ліміт збитку {cfg.daily_loss_limit:.0%} досягнуто — нових угод сьогодні не буде.")
        _save_state(st)
        return log

    df = c.klines(cfg.symbol, cfg.interval, limit=500)
    sig = latest_signal(df, cfg)
    log.append(f"Остання закрита свічка {df.index[-1]} close {df['close'].iloc[-1]}: сигнал "
               f"{ {1: 'LONG', -1: 'SHORT', 0: 'немає'}[sig] }")
    if sig == 0:
        _save_state(st)
        return log

    f = c.filters(cfg.symbol)
    price = c.mark_price(cfg.symbol)
    margin = equity * cfg.margin_fraction
    qty = qty_for(margin, lev, price, f["step"])
    if float(qty) < f["min_qty"] or float(qty) * price < f["min_notional"]:
        log.append(f"Обсяг {qty} замалий для біржі (мін. {f['min_qty']}, {f['min_notional']} USDT).")
        _save_state(st)
        return log

    c.set_isolated_leverage(cfg.symbol, lev)
    side, close_side = ("BUY", "SELL") if sig > 0 else ("SELL", "BUY")
    res = c.market_order(cfg.symbol, side, qty)
    entry = float(res.get("avgPrice") or 0) or price
    tp, sl = tp_sl_prices(entry, sig, lev, cfg.take_profit, cfg.stop_loss, f["tick"])
    try:
        c.protective_order(cfg.symbol, close_side, "STOP_MARKET", sl)
        c.protective_order(cfg.symbol, close_side, "TAKE_PROFIT_MARKET", tp)
    except BinanceError as e:
        c.close_position(cfg.symbol)  # без захисних ордерів позицію не тримаємо
        log.append(f"⚠️ Не вдалося поставити TP/SL ({e}). Позицію одразу закрито.")
        _save_state(st)
        return log

    st["open_trade"] = {"opened_at": now.isoformat(timespec="seconds"), "symbol": cfg.symbol,
                        "side": "LONG" if sig > 0 else "SHORT", "qty": qty, "entry": entry, "leverage": lev,
                        "margin": round(margin, 4), "tp": tp, "sl": sl, "equity_before": equity}
    log.append(f"🚀 Відкрито {st['open_trade']['side']} {qty} {cfg.symbol} @ {entry} (x{lev}, маржа {margin:.2f} USDT). "
               f"TP {tp} / SL {sl}")
    _save_state(st)
    return log


def run_loop(cfg: BotConfig):
    """Запуск щогодини через 30 с після закриття свічки."""
    while True:
        try:
            for line in run_once(cfg):
                print(line, flush=True)
        except Exception as e:  # продовжуємо роботу, але показуємо помилку
            print(f"Помилка циклу: {e}", flush=True)
        now = time.time()
        time.sleep(3600 - now % 3600 + 30)


def config_from(settings: dict) -> BotConfig:
    return BotConfig(**settings.get("testnet_bot", {}))


def reset_halt():
    st = _load_state()
    st.pop("halted", None)
    st.pop("peak_equity", None)
    _save_state(st)
