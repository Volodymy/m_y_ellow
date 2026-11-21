"""Відправка звіту в Telegram. Потрібні змінні TELEGRAM_BOT_TOKEN і TELEGRAM_CHAT_ID."""
from __future__ import annotations

import os

import requests


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram: не задано TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, пропускаю.")
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text[:4000]}, timeout=20)
    return r.ok
