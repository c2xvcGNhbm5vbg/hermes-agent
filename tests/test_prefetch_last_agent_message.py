"""Tests for ``memory.prefetch_include_last_agent_message`` (opt-in, default off).

When enabled, the per-turn memory prefetch query is augmented with the last
assistant message (bounded to 500 chars, truncated with ``…``) so anaphoric
follow-ups like "implement that plan" retrieve relevant memory. Provider-agnostic
core code — no lancedb changes.
"""

import types

import pytest

import agent.turn_context as tc
from agent.turn_context import _memory_turn_start_and_prefetch
from hermes_cli.config_defaults import DEFAULT_CONFIG


class _MemoryManagerStub:
    """Captures the query passed to ``prefetch_all``; no-ops everything else."""

    def __init__(self):
        self.prefetch_calls = []

    def on_turn_start(self, *args, **kwargs):
        return None

    def prefetch_all(self, query, **kwargs):
        self.prefetch_calls.append(query)
        return ""

    def describe_recall(self):
        return None


def _make_agent():
    return types.SimpleNamespace(
        _memory_manager=_MemoryManagerStub(),
        _user_turn_count=1,
        session_id="sess-test",
        _emit_status=lambda *a: None,
    )


def _prefetch_query(agent, user_message, last_assistant_message=None):
    _memory_turn_start_and_prefetch(
        agent, user_message, None, last_assistant_message=last_assistant_message
    )
    assert len(agent._memory_manager.prefetch_calls) == 1
    return agent._memory_manager.prefetch_calls[0]


def test_default_off_no_append(monkeypatch):
    """Flag False: the last assistant message is NOT appended to the prefetch query."""
    monkeypatch.setattr(tc, "_prefetch_include_last_agent_message", lambda: False)
    agent = _make_agent()
    query = _prefetch_query(agent, "implement that plan", "we planned the API last turn")
    assert "we planned the API last turn" not in query
    assert query == "implement that plan"


def test_flag_on_appends_last_assistant(monkeypatch):
    """Flag on + last assistant present: the captured query contains the assistant text."""
    monkeypatch.setattr(tc, "_prefetch_include_last_agent_message", lambda: True)
    agent = _make_agent()
    query = _prefetch_query(agent, "implement that plan", "we planned the API last turn")
    assert "we planned the API last turn" in query
    assert "implement that plan" in query


def test_flag_on_no_assistant_message_unchanged(monkeypatch):
    """Flag on + no assistant message (None): the query is unchanged."""
    monkeypatch.setattr(tc, "_prefetch_include_last_agent_message", lambda: True)
    agent = _make_agent()
    query = _prefetch_query(agent, "implement that plan", None)
    assert query == "implement that plan"


def test_flag_on_long_assistant_message_bounded(monkeypatch):
    """Flag on + 2000-char assistant message: bounded to 500 chars, truncated with '…'."""
    monkeypatch.setattr(tc, "_prefetch_include_last_agent_message", lambda: True)
    agent = _make_agent()
    long_text = "a" * 2000
    query = _prefetch_query(agent, "implement that plan", long_text)
    assert query.endswith("…")
    # The assistant part is the flattened 500-char bound: 499 chars + '…'.
    assert query == "implement that plan\n" + "a" * 499 + "…"
    assert len(query) <= len("implement that plan\n") + 500


def test_default_config_flag_off():
    """Config default: memory.prefetch_include_last_agent_message is False."""
    assert DEFAULT_CONFIG["memory"]["prefetch_include_last_agent_message"] is False
