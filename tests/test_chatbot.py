"""Placeholder tests for chatbot graph and chains."""

import pytest

from app.chatbot import chains, graph, prompts


@pytest.mark.skip(reason="not yet implemented")
def test_graph_routing_placeholder():
    """Will validate intent-based routing in the LangGraph workflow."""
    assert graph.chatbot_graph is not None


@pytest.mark.skip(reason="not yet implemented")
def test_prompt_and_chain_composition_placeholder():
    """Will validate prompt templates and chain wiring for GPT-4o calls."""
    assert prompts is not None and chains is not None
