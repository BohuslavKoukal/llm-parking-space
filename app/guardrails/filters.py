"""Guardrail filters based on Presidio and regex pattern checks."""

import re

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

FORBIDDEN_PATTERNS = [
    r"\bapi[_-]?key\b",
    r"\bweaviate\b",
    r"\bdatabase\b",
    r"\bsql\b",
    r"\bschema\b",
    r"\bvector\b",
    r"\bpassword\b",
    r"\bsecret\b",
    r"\btoken\b",
]

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    return _analyzer


def _get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


def analyze_text(text: str) -> list[str]:
    """Return detected PII entity types from input text."""
    try:
        results = _get_analyzer().analyze(text=text, language="en")
        return sorted({result.entity_type for result in results})
    except Exception:
        return []


def anonymize_text(text: str) -> str:
    """Anonymize detected PII entities and return sanitized text."""
    try:
        analyzer_results = _get_analyzer().analyze(text=text, language="en")
    except Exception:
        return text

    if not analyzer_results:
        return text
    return _get_anonymizer().anonymize(text=text, analyzer_results=analyzer_results).text


def is_sensitive(text: str) -> bool:
    """Detect sensitive input via Presidio PII entities or forbidden regex patterns."""
    if analyze_text(text):
        return True
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in FORBIDDEN_PATTERNS)
