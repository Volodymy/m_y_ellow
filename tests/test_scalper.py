import numpy as np
import pandas as pd
import pytest

from office.patterns import DoubleTopBottomAgent, FibonacciAgent, combine
from office.scalper import ScalpParams, simulate, synthetic_ohlc


def bars(closes, wick=0.0):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + wick), "low": np.minimum(o, c) * (1 - wick),
                         "close": c}, index=pd.date_range("2025-01-01", periods=len(c), freq="15min"))


def test_take_profit_is_percent_of_margin():
    df = bars([100, 100, 100.6, 101, 101])
    sig = np.array([1, 0, 0, 0, 0])
    p = ScalpParams(leverage=20, take_profit=0.10, stop_loss=0.10, taker_fee=0.0, funding_8h=0.0)
    r = simulate(df, sig, p, capital=1000)
    t = r.trades.iloc[0]
    assert t["reason"] == "take_profit"
    assert t["pnl_pct_margin"] == pytest.approx(0.10)       # +10% від маржі = +0.5% ціни при x20


def test_fees_turn_tiny_target_into_loss_at_high_leverage():
    df = bars([100, 100, 100.2, 100.2])
    p = ScalpParams(leverage=50, take_profit=0.05, stop_loss=0.05)  # ціль = 0.1% ціни, комісії = 0.1%
    r = simulate(df, np.array([1, 0, 0, 0]), p, capital=1000)
    assert r.trades.iloc[0]["reason"] == "take_profit"
    assert r.trades.iloc[0]["pnl"] <= 0


def test_liquidation_loses_whole_margin_when_stop_is_too_far():
    df = bars([100, 100, 90, 90])
    p = ScalpParams(leverage=20, take_profit=0.5, stop_loss=2.0, taker_fee=0.0, funding_8h=0.0)
    r = simulate(df, np.array([1, 0, 0, 0]), p, capital=1000)
    t = r.trades.iloc[0]
    assert t["reason"] == "liquidation"
    assert t["pnl"] == pytest.approx(-100.0)  # маржа = 10% від 1000


def test_short_side_profits_on_fall():
    df = bars([100, 100, 99.4, 99])
    p = ScalpParams(leverage=20, take_profit=0.10, stop_loss=0.10, taker_fee=0.0, funding_8h=0.0)
    r = simulate(df, np.array([-1, 0, 0, 0]), p)
    assert r.trades.iloc[0]["pnl"] > 0


@pytest.mark.parametrize("agent", [FibonacciAgent(), DoubleTopBottomAgent()])
def test_pattern_agents_have_no_lookahead(agent):
    df = synthetic_ohlc(n=3000, seed=3)
    full = agent.signals(df)
    cut = agent.signals(df.iloc[:2000])
    assert (full[:2000] == cut).all()
    assert (full != 0).sum() > 0


def test_combine_both_requires_agreement():
    a = np.array([0, 1, 0, 0, -1])
    b = np.array([0, 0, 1, 0, 1])
    assert list(combine([a, b], "both", window=1)) == [0, 0, 1, 0, 0]
