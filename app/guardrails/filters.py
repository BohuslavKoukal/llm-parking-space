"""Guardrail filters using Presidio and regex pattern checks."""

import re
from typing import Any, cast

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

PII_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "UK_NHS",
    "NRP",
]

FORBIDDEN_PATTERNS = [
    # Internal system keywords
    r"\bweaviate\b",
    r"\bvector\b",
    r"\bembedding\b",
    r"\bdatabase\b",
    r"\bschema\b",
    r"\bsql\b",
    r"\btable\b",
    r"\bcolumn\b",
    # Security probing
    r"api[._-]?key",
    r"\bapi_key\b",
    r"\bsecret\b",
    r"\bpassword\b",
    r"\btoken\b",
    # Prompt injection attempts
    r"ignore.*instructions",
    r"forget.*instructions",
    r"disregard.*instructions",
    r"you are now",
    r"act as if",
    r"pretend you are",
    r"ignore previous",
    r"new persona",
    # Data extraction attempts
    r"show.*all.*reservations",
    r"list.*all.*users",
    r"dump.*data",
    r"show.*database",
]

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def get_analyzer() -> AnalyzerEngine:
    """Return a cached Presidio AnalyzerEngine configured with spaCy en_core_web_lg."""
    global _analyzer
    if _analyzer is None:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    """Return a cached Presidio AnonymizerEngine instance."""
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


def analyze_text(text: str) -> list[RecognizerResult]:
    """Detect configured PII entities in English text and return recognizer results."""
    if not text:
        return []
    try:
        return get_analyzer().analyze(text=text, entities=PII_ENTITIES, language="en")
    except Exception:
        return []


def anonymize_text(text: str) -> str:
    """Replace detected PII spans with entity placeholders like <PERSON>."""
    analyzer_results = analyze_text(text)
    if not analyzer_results:
        return text

    operators = {
        result.entity_type: OperatorConfig("replace", {"new_value": f"<{result.entity_type}>"})
        for result in analyzer_results
    }
    operators["DEFAULT"] = OperatorConfig("replace", {"new_value": "<PII>"})

    return get_anonymizer().anonymize(
        text=text,
        analyzer_results=cast(list[Any], analyzer_results),
        operators=operators,
    ).text


def contains_pii(text: str) -> bool:
    """Return True when Presidio detects at least one configured PII entity."""
    return bool(analyze_text(text))


def contains_forbidden_patterns(text: str) -> bool:
    """Return True when text matches any forbidden regex pattern (case-insensitive)."""
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS)


def is_sensitive(text: str) -> bool:
    """Return True when text contains PII or forbidden guardrail patterns."""
    return contains_pii(text) or contains_forbidden_patterns(text)


def get_block_reason(text: str) -> str:
    """Return block reason: pii_detected, forbidden_pattern, or safe."""
    if contains_pii(text):
        return "pii_detected"
    if contains_forbidden_patterns(text):
        return "forbidden_pattern"
    return "safe"
