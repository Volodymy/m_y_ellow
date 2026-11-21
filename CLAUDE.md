# CLAUDE.md

Python project: a paper-trading "office" of rule-based agents (`office/`). Docs and user-facing text are in Ukrainian.

- Run tests: `pytest -q`. Offline check: `python -m office backtest --synthetic`.
- Paper trading only. Do not add live order execution without the owner's explicit request.
- Strategies must not look ahead: row t uses data ≤ t, and backtests execute at t+1 (`targets.shift(1)`). Add a no-lookahead test for any new strategy.
- An LLM (Claude, `office/llm.py`) only comments on reports; it never sets weights or orders.
- Strategies that can't be honestly backtested set `backtestable = False`.
- Config lives in `config/settings.yaml`; fund holdings in `config/superinvestors.yaml` (with sources and dates).
- Network hosts needed: data-api.binance.vision, query1.finance.yahoo.com, data.sec.gov, www.sec.gov.
