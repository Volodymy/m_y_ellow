import numpy as np
import pandas as pd
import pytest

from office.agents import Analyst, PaperBroker, RiskManager
from office.backtest import metrics, run_backtest
from office.strategies import CryptoTrend, DualMomentum, SuperinvestorClone

COSTS = {"crypto": {"fee": 0.001, "slippage": 0.0005}, "stock": {"fee": 0.0, "slippage": 0.0}}


def trend_prices():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    up = np.concatenate([np.full(100, 100.0), np.linspace(100, 200, 150), np.linspace(200, 120, 150)])
    return pd.DataFrame({"BTCUSDT": up, "ETHUSDT": np.full(400, 50.0)}, index=idx)


def test_trend_enters_on_breakout_and_exits_on_breakdown():
    w = CryptoTrend(universe=["BTCUSDT", "ETHUSDT"]).target_history(trend_prices())
    assert w["BTCUSDT"].iloc[200] > 0          # у середині росту — в позиції
    assert w["BTCUSDT"].iloc[-1] == 0          # після падіння — вийшли
    assert (w["ETHUSDT"] == 0).all()           # флет — жодного входу


def test_trend_has_no_lookahead():
    px = trend_prices()
    full = CryptoTrend(universe=["BTCUSDT", "ETHUSDT"]).target_history(px)
    cut = CryptoTrend(universe=["BTCUSDT", "ETHUSDT"]).target_history(px.iloc[:250])
    pd.testing.assert_frame_equal(full.iloc[:250], cut)


def test_dual_momentum_goes_to_safe_asset_in_bear_market():
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    falling = np.linspace(100, 50, 800)
    px = pd.DataFrame({"SPY": falling, "QQQ": falling * 1.1, "SHY": np.linspace(100, 102, 800)}, index=idx)
    w = DualMomentum(risky=["SPY", "QQQ"], safe="SHY", top_n=2).target_history(px)
    assert w["SHY"].iloc[-1] == pytest.approx(1.0)


def test_superinvestor_weights_skip_null_and_normalize(tmp_path):
    f = tmp_path / "h.yaml"
    f.write_text("a:\n  holdings: {AAPL: 60, KO: null, AXP: 40}\nb:\n  holdings: {MSFT: 1}\n")
    w = SuperinvestorClone(holdings_file=str(f), funds=["a", "b"]).weights()
    assert "KO" not in w
    assert w.sum() == pytest.approx(1.0)
    assert w["MSFT"] == pytest.approx(0.5)


def test_risk_manager_caps_and_kill_switch():
    rm = RiskManager(max_position_weight=0.15, cash_buffer=0.02, max_drawdown_kill=0.25)
    out = rm.check(pd.Series({"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5, "F": 0.5, "G": 0.5}), 10_000)
    assert out.max() <= 0.15 + 1e-12
    assert out.sum() <= 0.98 + 1e-9
    rm.check(pd.Series({"A": 0.1}), 7_000)  # -30% від піку
    assert rm.halted
    assert rm.check(pd.Series({"A": 0.1}), 7_500).sum() == 0


def test_broker_charges_fees_and_never_goes_negative_cash():
    b = PaperBroker(cash=1000, costs=COSTS)
    px = pd.Series({"BTCUSDT": 100.0})
    rm = RiskManager()
    b.rebalance(pd.Series({"BTCUSDT": 1.0}), px, "2024-01-01", rm)
    assert b.cash >= -1e-9
    assert b.trades[0].fee > 0
    b.rebalance(pd.Series({"BTCUSDT": 0.0}), px, "2024-01-02", rm)
    assert "BTCUSDT" not in b.positions
    assert b.cash < 1000  # комісії й проковзування з'їли частину


def test_backtest_runs_and_reports_metrics():
    px = trend_prices()
    analyst = Analyst({"crypto_trend": CryptoTrend(universe=["BTCUSDT", "ETHUSDT"])}, {"crypto_trend": 1.0})
    res = run_backtest(px, analyst, RiskManager(), COSTS, 10_000, warmup=60)
    assert res.trades > 0
    assert {"cagr", "sharpe", "max_drawdown"} <= set(res.full)


def test_metrics_drawdown():
    eq = pd.Series([100, 120, 60, 90], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    assert metrics(eq)["max_drawdown"] == pytest.approx(-0.5)
