"""Placeholder tests for guardrail filters."""

import pytest

from app.guardrails import filters


@pytest.mark.skip(reason="not yet implemented")
def test_pii_detection_placeholder():
    """Will validate that PII entities are detected from user input."""
    assert filters is not None


@pytest.mark.skip(reason="not yet implemented")
def test_forbidden_pattern_detection_placeholder():
    """Will validate non-PII forbidden term detection logic."""
    assert filters.FORBIDDEN_PATTERNS
