"""Коментар Claude до щоденного звіту.

Claude лише пояснює звіт людською мовою. Він не змінює ваги і не відправляє
ордери: усі торгові рішення ухвалюють правила стратегій і ризик-менеджера.
"""
from __future__ import annotations

SYSTEM = (
    "Ти аналітик у команді паперової торгівлі. Тобі дають щоденний звіт робота. "
    "Поясни українською в 3–5 реченнях, що змінилося в портфелі й чому це логічно "
    "з погляду правил стратегій (трендова на крипті, моментум ETF, копія 13F). "
    "Не давай інвестиційних порад і не прогнозуй ціни. Якщо бачиш ризики "
    "(концентрація, аварійна зупинка, багато угод), назви їх."
)


def commentary(report: str) -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model="claude-opus-5-5",
            max_tokens=2000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            messages=[{"role": "user", "content": report}],
        )
    except anthropic.AuthenticationError:
        return "(Коментар недоступний: не налаштовано ключ Anthropic API.)"
    except anthropic.RateLimitError:
        return "(Коментар недоступний: перевищено ліміт запитів, спробуйте пізніше.)"
    except anthropic.APIStatusError as e:
        return f"(Коментар недоступний: помилка API {e.status_code}.)"
    except anthropic.APIConnectionError:
        return "(Коментар недоступний: немає з'єднання з API.)"
    if response.stop_reason == "refusal":
        return None
    return "".join(b.text for b in response.content if b.type == "text").strip() or None
