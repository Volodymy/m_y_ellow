#!/usr/bin/env bash
# Встановлення бота на VPS (Ubuntu 22.04/24.04) у Європі. Запуск:
#   bash install.sh
# Бот працює з Binance Demo Trading (віртуальні гроші) як системна служба 24/7.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/Volodymy/m_y_ellow.git}"
BRANCH="${BRANCH:-claude/market-analysis-monetization-5yfdr4}"
DIR="$HOME/m_y_ellow"

sudo apt-get update -y && sudo apt-get install -y python3 python3-venv git
[ -d "$DIR" ] || git clone -b "$BRANCH" "$REPO_URL" "$DIR"
cd "$DIR" && git pull --ff-only || true
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

if [ ! -f .env ]; then
  read -rp "Demo API Key: " K
  read -rsp "Demo Secret Key: " S; echo
  printf 'BINANCE_ENV=demo\nBINANCE_TESTNET_API_KEY=%s\nBINANCE_TESTNET_API_SECRET=%s\n' "$K" "$S" > .env
  chmod 600 .env
fi

.venv/bin/python -m office testnet --check

sudo tee /etc/systemd/system/gold-bot.service >/dev/null <<UNIT
[Unit]
Description=Gold demo trading bot
After=network-online.target
[Service]
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python -m office testnet --loop
Restart=always
User=$USER
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now gold-bot
echo "Готово. Журнал: journalctl -u gold-bot -f   Угоди: $DIR/data/testnet_trades.csv"
