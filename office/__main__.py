"""Командний рядок офісу.

  python -m office backtest [--synthetic]   прогін на історії
  python -m office run [--synthetic]        один робочий цикл паперового рахунку
  python -m office status                   стан паперового рахунку
  python -m office reset                    скинути паперовий рахунок і аварійну зупинку
  python -m office fetch-13f berkshire_hathaway   оновити портфель фонду з SEC EDGAR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .agents import Analyst, PaperBroker, Reporter, RiskManager
from .backtest import format_result, run_backtest
from .data import load_prices
from .strategies import build_strategies

STATE = Path("data/paper_account.json")
STATE_SYNTHETIC = Path("data/paper_account_synthetic.json")  # окремий рахунок, щоб не змішувати з реальним


def state_path(args) -> Path:
    return STATE_SYNTHETIC if getattr(args, "synthetic", False) else STATE


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def make_office(cfg: dict, backtest_only: bool = False):
    strategies = build_strategies(cfg)
    analyst = Analyst(strategies, cfg["allocation"], backtest_only=backtest_only)
    risk = RiskManager(**cfg["risk"])
    return analyst, risk


def all_symbols(analyst: Analyst) -> list[str]:
    return [s for st in analyst.active().values() for s in st.symbols()]


def cmd_backtest(cfg, args):
    analyst, risk = make_office(cfg, backtest_only=True)
    skipped = [k for k, s in analyst.strategies.items() if not s.backtestable]
    syms = all_symbols(analyst) + ["SPY", "BTCUSDT"]
    prices = load_prices(syms, start=cfg["backtest"]["start"], synthetic=args.synthetic)
    title = "Бектест (СИНТЕТИЧНІ дані — лише перевірка механіки)" if args.synthetic else "Бектест"
    if skipped:
        print(f"Не бектестуються (немає історичних 13F, був би lookahead): {', '.join(skipped)}")
    res = run_backtest(prices, analyst, risk, cfg["costs"], cfg["starting_capital"],
                       cfg["backtest"]["out_of_sample_fraction"])
    print(format_result(res, title))
    out = Path("reports")
    out.mkdir(exist_ok=True)
    res.equity.to_csv(out / "backtest_equity.csv", header=["equity"])
    print(f"Крива капіталу: {out / 'backtest_equity.csv'}")


def cmd_run(cfg, args):
    analyst, risk = make_office(cfg)
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    prices = load_prices(all_symbols(analyst), start=start, synthetic=args.synthetic)
    targets, _ = analyst.target_history(prices)
    today = prices.index[-1]
    px = prices.loc[today, targets.columns]

    broker = PaperBroker.load(state_path(args), cfg["costs"], cfg["starting_capital"], risk)
    tgt = risk.check(targets.loc[today], broker.equity(px))
    trades = broker.rebalance(tgt, px, today, risk)
    broker.save(state_path(args), risk)
    rep = Reporter(**cfg["reporter"])
    print(rep.build(today, broker, px, trades, risk, cfg["starting_capital"]))


def cmd_status(cfg, args):
    if not STATE.exists():
        print("Паперовий рахунок ще не створено. Запустіть: python -m office run")
        return
    print(STATE.read_text())


def cmd_reset(cfg, args):
    STATE.unlink(missing_ok=True)
    STATE_SYNTHETIC.unlink(missing_ok=True)
    print("Паперовий рахунок скинуто.")


def cmd_fetch_13f(cfg, args):
    from .edgar import update_holdings_file
    df = update_holdings_file(args.fund, cfg["strategies"]["superinvestor_clone"]["holdings_file"])
    print(df.head(20).to_string(index=False))
    unmapped = df[df["ticker"].isna()]
    if len(unmapped):
        print(f"\nБез тікера ({len(unmapped)}): додайте CUSIP у office/edgar.py → CUSIP_TO_TICKER")


def main():
    p = argparse.ArgumentParser(prog="office", description="Офіс AI-асистентів для паперової торгівлі")
    p.add_argument("--config", default="config/settings.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--synthetic", action="store_true", help="синтетичні дані без інтернету")
    sub.add_parser("status")
    sub.add_parser("reset")
    f = sub.add_parser("fetch-13f")
    f.add_argument("fund", help="ключ фонду з config/superinvestors.yaml")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    {"backtest": cmd_backtest, "run": cmd_run, "status": cmd_status,
     "reset": cmd_reset, "fetch-13f": cmd_fetch_13f}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
