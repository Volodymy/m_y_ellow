"""Клієнт Binance USDⓈ-M Futures — ТІЛЬКИ ТЕСТОВА МЕРЕЖА.

Адреса жорстко прописана на testnet.binancefuture.com: реальних грошей цей
модуль не торкається. Ключі читаються з файлу .env (він у .gitignore):

    BINANCE_TESTNET_API_KEY=...
    BINANCE_TESTNET_API_SECRET=...
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode

import requests

TESTNET_URL = "https://testnet.binancefuture.com"


class BinanceError(RuntimeError):
    def __init__(self, status: int, code: int | None, msg: str):
        super().__init__(f"HTTP {status}, code {code}: {msg}")
        self.status, self.code, self.msg = status, code, msg


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def round_step(value: float, step: str) -> str:
    """Округлення вниз до кроку біржі (кількість / ціна) у вигляді рядка без хвостів."""
    st = Decimal(step)
    q = (Decimal(str(value)) / st).to_integral_value(rounding="ROUND_FLOOR") * st
    return format(q.normalize(), "f")


class FuturesTestnet:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None, session=None):
        load_env()
        self.key = api_key or os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self.secret = api_secret or os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        self.base = TESTNET_URL
        self.s = session or requests.Session()
        self.time_offset = 0
        self._filters: dict[str, dict] = {}

    # ------------------------------------------------------------ низький рівень
    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {}
        if signed:
            if not self.key or not self.secret:
                raise BinanceError(0, None, "Немає ключів у .env (BINANCE_TESTNET_API_KEY / _SECRET)")
            params["timestamp"] = int(time.time() * 1000) + self.time_offset
            params["recvWindow"] = 5000
            q = urlencode(params)
            params_str = f"{q}&signature={sign(self.secret, q)}"
            headers["X-MBX-APIKEY"] = self.key
        else:
            params_str = urlencode(params)
        url = f"{self.base}{path}" + (f"?{params_str}" if params_str else "")
        r = self.s.request(method, url, headers=headers, timeout=20)
        try:
            data = r.json()
        except ValueError:
            data = {"msg": r.text[:200]}
        if r.status_code >= 400 or (isinstance(data, dict) and data.get("code", 0) < 0):
            raise BinanceError(r.status_code, data.get("code") if isinstance(data, dict) else None,
                               data.get("msg", "") if isinstance(data, dict) else str(data))
        return data

    def sync_time(self):
        server = self._request("GET", "/fapi/v1/time")["serverTime"]
        self.time_offset = server - int(time.time() * 1000)

    # ------------------------------------------------------------ ринкові дані
    def klines(self, symbol: str, interval: str = "1h", limit: int = 500):
        import pandas as pd
        rows = self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame([[float(x) for x in r[1:5]] + [r[6]] for r in rows],
                          columns=["open", "high", "low", "close", "close_time"],
                          index=pd.to_datetime([r[0] for r in rows], unit="ms"))
        # остання свічка ще не закрита — відкидаємо, щоб не торгувати на неповних даних
        now_ms = int(time.time() * 1000) + self.time_offset
        return df[df["close_time"] < now_ms].drop(columns="close_time")

    def mark_price(self, symbol: str) -> float:
        return float(self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})["markPrice"])

    def filters(self, symbol: str) -> dict:
        if symbol not in self._filters:
            info = self._request("GET", "/fapi/v1/exchangeInfo")
            sym = next(s for s in info["symbols"] if s["symbol"] == symbol)
            f = {x["filterType"]: x for x in sym["filters"]}
            self._filters[symbol] = {
                "tick": f["PRICE_FILTER"]["tickSize"],
                "step": f["LOT_SIZE"]["stepSize"],
                "min_qty": float(f["LOT_SIZE"]["minQty"]),
                "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", 0) or 0),
            }
        return self._filters[symbol]

    # ------------------------------------------------------------ рахунок
    def balance_usdt(self) -> float:
        for b in self._request("GET", "/fapi/v2/balance", signed=True):
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0

    def wallet_usdt(self) -> float:
        acc = self._request("GET", "/fapi/v2/account", signed=True)
        return float(acc["totalWalletBalance"]) + float(acc.get("totalUnrealizedProfit", 0))

    def position(self, symbol: str) -> dict | None:
        for p in self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True):
            if float(p["positionAmt"]) != 0:
                return p
        return None

    def set_isolated_leverage(self, symbol: str, leverage: int):
        try:
            self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)
        except BinanceError as e:
            if e.code != -4046:  # -4046: вже ISOLATED
                raise
        self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    # ------------------------------------------------------------ ордери
    def market_order(self, symbol: str, side: str, qty: str) -> dict:
        return self._request("POST", "/fapi/v1/order",
                             {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty,
                              "newOrderRespType": "RESULT"}, signed=True)

    def protective_order(self, symbol: str, side: str, kind: str, trigger: str) -> dict:
        """Стоп-лос (STOP_MARKET) або тейк-профіт (TAKE_PROFIT_MARKET), що закриває всю позицію.

        З кінця 2025 Binance переносить умовні ордери в Algo Order API, тому
        при помилці -4120 повторюємо запит через /fapi/v1/algoOrder.
        """
        try:
            return self._request("POST", "/fapi/v1/order",
                                 {"symbol": symbol, "side": side, "type": kind, "stopPrice": trigger,
                                  "closePosition": "true", "workingType": "MARK_PRICE"}, signed=True)
        except BinanceError as e:
            if e.code != -4120:
                raise
        return self._request("POST", "/fapi/v1/algoOrder",
                             {"algoType": "CONDITIONAL", "symbol": symbol, "side": side, "type": kind,
                              "triggerPrice": trigger, "closePosition": "true", "workingType": "MARK_PRICE"},
                             signed=True)

    def cancel_all(self, symbol: str):
        self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        try:
            self._request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol}, signed=True)
        except BinanceError:
            pass

    def close_position(self, symbol: str):
        p = self.position(symbol)
        if p:
            amt = float(p["positionAmt"])
            self._request("POST", "/fapi/v1/order",
                          {"symbol": symbol, "side": "SELL" if amt > 0 else "BUY", "type": "MARKET",
                           "quantity": format(abs(Decimal(p["positionAmt"])).normalize(), "f"),
                           "reduceOnly": "true"}, signed=True)
        self.cancel_all(symbol)


def qty_for(margin_usdt: float, leverage: int, price: float, step: str) -> str:
    return round_step(margin_usdt * leverage / price, step)


def tp_sl_prices(entry: float, side: int, leverage: int, take_profit: float, stop_loss: float,
                 tick: str) -> tuple[str, str]:
    tp = entry * (1 + side * take_profit / leverage)
    sl = entry * (1 - side * stop_loss / leverage)
    # округлюємо в бік, безпечний для позиції: тейк ближче, стоп ближче
    t = Decimal(tick)

    def r(x, up):
        q = Decimal(str(x)) / t
        q = q.to_integral_value(rounding="ROUND_CEILING" if up else "ROUND_FLOOR")
        return format((q * t).normalize(), "f")

    return r(tp, side < 0), r(sl, side > 0)
