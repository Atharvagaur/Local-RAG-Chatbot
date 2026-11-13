"""Tests for chat_manager.py."""

from __future__ import annotations

from backend.chat_manager import ChatManager


def test_add_and_trim_history():
    cm = ChatManager(max_messages=4)
    for i in range(6):
        cm.add_user_message(f"q{i}")
        cm.add_ai_message(f"a{i}")
    assert len(cm.history) <= 4
    # most recent messages retained
    assert cm.history[-1].content == "a5"


def test_contextualize_short_question_returns_standalone():
    cm = ChatManager()
    cm.add_user_message("Tell me about renewable energy.")
    cm.add_ai_message("Renewable energy comes from natural sources.")
    result = cm.contextualize_question("What are its benefits?")
    assert "benefits" in result.lower()
    assert "renewable energy" in result.lower()


def test_contextualize_independent_question_unchanged():
    cm = ChatManager()
    cm.add_user_message("Previous topic.")
    cm.add_ai_message("Previous answer.")
    result = cm.contextualize_question("What is the capital of France?")
    assert result == "What is the capital of France?"


def test_clear():
    cm = ChatManager()
    cm.add_user_message("q")
    cm.add_ai_message("a")
    cm.clear()
    assert cm.history == []
