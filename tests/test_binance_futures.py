import hashlib
import hmac

from office.binance_futures import TESTNET_URL, qty_for, round_step, sign, tp_sl_prices


def test_only_testnet_url():
    assert TESTNET_URL == "https://testnet.binancefuture.com"


def test_signature_matches_hmac_sha256():
    q = "symbol=XAUUSDT&timestamp=1"
    assert sign("secret", q) == hmac.new(b"secret", q.encode(), hashlib.sha256).hexdigest()


def test_round_step_floors_to_exchange_step():
    assert round_step(1.23456, "0.001") == "1.234"
    assert round_step(0.0009, "0.001") == "0"


def test_qty_for_leverage():
    assert qty_for(100, 10, 4000, "0.001") == "0.25"   # маржа 100 × x10 / 4000


def test_tp_sl_for_long_and_short():
    tp, sl = tp_sl_prices(4000, 1, 10, 0.30, 0.30, "0.01")
    assert (float(tp), float(sl)) == (4120.0, 3880.0)  # ±3% ціни = ±30% маржі при x10
    tp, sl = tp_sl_prices(4000, -1, 10, 0.30, 0.30, "0.01")
    assert (float(tp), float(sl)) == (3880.0, 4120.0)


def test_no_real_binance_endpoint():
    import pytest
    from office.binance_futures import ENDPOINTS, FuturesTestnet
    assert all("fapi.binance.com" != u.split("//")[1] for u in ENDPOINTS.values())
    with pytest.raises(ValueError):
        FuturesTestnet(env="live")
