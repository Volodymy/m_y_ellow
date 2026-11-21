"""Завантаження останнього звіту 13F фонду напряму з SEC EDGAR.

SEC просить вказувати в User-Agent контакт; задайте свій через змінну
SEC_USER_AGENT, наприклад "MyName my@email.com".
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
import yaml

# CUSIP → тікер для найпоширеніших позицій. Невідомі CUSIP показуються як є.
CUSIP_TO_TICKER = {
    "037833100": "AAPL", "025816109": "AXP", "191216100": "KO", "060505104": "BAC",
    "166764100": "CVX", "674599105": "OXY", "615369105": "MCO", "H1467J104": "CB",
    "02079K305": "GOOGL", "02079K107": "GOOG", "500754106": "KHC", "594918104": "MSFT",
    "023135106": "AMZN", "67066G104": "NVDA", "30303M102": "META", "92826C839": "V",
    "57636Q104": "MA", "14040H105": "COF", "92343E102": "VRSN", "526057104": "LEN",
    "247361702": "DAL", "084670702": "BRK.B", "46625H100": "JPM", "88160R101": "TSLA",
}


def _ua() -> dict:
    return {"User-Agent": os.environ.get("SEC_USER_AGENT", "ai-trading-office research")}


def latest_13f(cik: str) -> tuple[pd.DataFrame, dict]:
    cik10 = cik.zfill(10)
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik10}.json", headers=_ua(), timeout=30)
    sub.raise_for_status()
    recent = sub.json()["filings"]["recent"]
    i = next(k for k, f in enumerate(recent["form"]) if f == "13F-HR")
    acc = recent["accessionNumber"][i].replace("-", "")
    meta = {"filed": recent["filingDate"][i], "as_of": recent["reportDate"][i], "accession": recent["accessionNumber"][i]}

    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"
    idx = requests.get(f"{base}/index.json", headers=_ua(), timeout=30)
    idx.raise_for_status()
    names = [it["name"] for it in idx.json()["directory"]["item"]]
    xml_name = next(n for n in names if n.lower().endswith(".xml") and "primary_doc" not in n.lower())
    xml = requests.get(f"{base}/{xml_name}", headers=_ua(), timeout=60)
    xml.raise_for_status()

    rows = []
    for node in ET.fromstring(xml.content).iter():
        if node.tag.split("}")[-1] != "infoTable":
            continue
        get = {c.tag.split("}")[-1]: (c.text or "").strip() for c in node.iter()}
        rows.append({"issuer": get.get("nameOfIssuer"), "cusip": get.get("cusip"),
                     "value": float(get.get("value") or 0), "shares": float(get.get("sshPrnamt") or 0)})
    df = pd.DataFrame(rows).groupby(["cusip", "issuer"], as_index=False)[["value", "shares"]].sum()
    df["ticker"] = df["cusip"].map(CUSIP_TO_TICKER)
    df["weight_pct"] = df["value"] / df["value"].sum() * 100
    return df.sort_values("weight_pct", ascending=False), meta


def update_holdings_file(fund_key: str, holdings_file: str = "config/superinvestors.yaml", top: int = 15) -> pd.DataFrame:
    path = Path(holdings_file)
    data = yaml.safe_load(path.read_text())
    fund = data[fund_key]
    df, meta = latest_13f(fund["cik"])
    mapped = df.dropna(subset=["ticker"]).head(top)
    fund["holdings"] = {r.ticker: round(r.weight_pct, 2) for r in mapped.itertuples()}
    fund.update({"as_of": meta["as_of"], "filed": meta["filed"],
                 "source": [f"SEC EDGAR 13F-HR {meta['accession']}"]})
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return df
