"""Rule-based suspicious-content signals.

These are transparent, deterministic checks computed from the submitted text.
They are shown *alongside* (never instead of) the machine-learning verdict and
do not change the model's score.
"""
from __future__ import annotations

import re

_URL = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|net|org|info|biz|ru|xyz|top|click|link|cc|tk)\b\S*)", re.I)
_PHONE = re.compile(r"(\+?\d[\d\s().-]{8,}\d|\b\d{5,6}\b)")
_MONEY = re.compile(r"([£$€₹]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|gbp|eur|inr|rs)\b|\b(?:free|prize|winner|won|cash|claim|voucher|jackpot|reward|bonus|lottery|gift)\b)", re.I)
_URGENCY = re.compile(r"\b(urgent(?:ly)?|immediately|act now|hurry|expires?|limited time|last chance|final notice|today only|asap|right away|call now|reply now|don'?t miss)\b", re.I)
_PHISH = re.compile(r"\b(verify|password|passcode|otp|pin|account (?:is |has been )?(?:suspended|locked|blocked|compromised)|bank|login|log in|sign in|confirm your|click here|security alert|ssn|credit card|kyc)\b", re.I)


def _uniq(matches, limit=5):
    seen, out = set(), []
    for m in matches:
        s = m.strip(" .,;:!?()\"'")
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s[:60])
        if len(out) >= limit:
            break
    return out


def _find(pattern, text):
    return [m.group(0) for m in pattern.finditer(text)]


def analyze_signals(text: str) -> list[dict]:
    """Return a checklist of suspicious-content indicators for ``text``."""
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    exclaims = text.count("!")

    def entry(key, label, found, detail, severity):
        return {"key": key, "label": label, "found": bool(found),
                "detail": detail, "severity": severity if found else "none"}

    urls = _uniq(_find(_URL, text))
    phones = _uniq(_find(_PHONE, text))
    money = _uniq(_find(_MONEY, text))
    urgency = _uniq(_find(_URGENCY, text))
    phish = _uniq(_find(_PHISH, text))

    return [
        entry("links", "Links or web addresses", urls,
              ", ".join(urls) if urls else "No links found", "high"),
        entry("phish", "Credential / account wording", phish,
              ", ".join(phish) if phish else "None detected", "high"),
        entry("money", "Prize, money or reward wording", money,
              ", ".join(money) if money else "None detected", "medium"),
        entry("urgency", "Urgency or pressure language", urgency,
              ", ".join(urgency) if urgency else "None detected", "medium"),
        entry("phone", "Phone numbers or short codes", phones,
              ", ".join(phones) if phones else "None detected", "medium"),
        entry("caps", "Excessive capital letters",
              len(letters) >= 10 and caps_ratio > 0.4,
              f"{caps_ratio * 100:.0f}% of letters are uppercase", "low"),
        entry("exclaim", "Repeated exclamation marks", exclaims >= 3,
              f"{exclaims} exclamation mark{'s' if exclaims != 1 else ''}", "low"),
    ]
