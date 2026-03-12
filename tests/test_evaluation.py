"""Placeholder tests for evaluation package."""

import pytest

from app.evaluation import ragas_eval, report


@pytest.mark.skip(reason="not yet implemented")
def test_ragas_execution_placeholder():
    """Will validate end-to-end RAGAS metric computation pipeline."""
    assert ragas_eval is not None


@pytest.mark.skip(reason="not yet implemented")
def test_report_generation_placeholder():
    """Will validate evaluation report generation from metric outputs."""
    assert report is not None
