"""Unit tests for Agent guardrails and response telemetry.

Tests empty query handling, error message formatting, and telemetry logging
without triggering live Gemini API calls.
"""

from __future__ import annotations

import pytest
from agent import AgentResponse, ToolExecutionStep, AutonomousAgent


class TestAgentInputGuards:
    """Tests for agent input validation and error handling."""

    def test_empty_query_string_handling(self) -> None:
        agent = AutonomousAgent.__new__(AutonomousAgent)
        agent.max_steps = 15

        res = agent.run("")
        assert res.success is False
        assert "Please enter a valid, non-empty query." in res.final_answer
        assert res.error_message == "Empty query provided."
        assert len(res.steps) == 0

    def test_whitespace_query_handling(self) -> None:
        agent = AutonomousAgent.__new__(AutonomousAgent)
        agent.max_steps = 15

        res = agent.run("     \n\t  ")
        assert res.success is False
        assert "Please enter a valid, non-empty query." in res.final_answer

    def test_error_message_translation_api_key(self) -> None:
        agent = AutonomousAgent.__new__(AutonomousAgent)
        agent.max_steps = 15
        agent.model_name = "gemini-3.8-flash"

        msg = agent._format_error_message("Error: API_KEY is invalid or not found")
        assert "Authentication Error" in msg

    def test_error_message_translation_rate_limit(self) -> None:
        agent = AutonomousAgent.__new__(AutonomousAgent)
        agent.max_steps = 15
        agent.model_name = "gemini-3.8-flash"

        msg = agent._format_error_message("ResourceExhausted: 429 quota reached")
        assert "Rate Limit Exceeded" in msg

    def test_error_message_translation_recursion_limit(self) -> None:
        agent = AutonomousAgent.__new__(AutonomousAgent)
        agent.max_steps = 15
        agent.model_name = "gemini-3.8-flash"

        msg = agent._format_error_message("GraphRecursionError: recursion_limit exceeded")
        assert "Execution Limit Reached" in msg

    def test_tool_execution_step_logging_format(self) -> None:
        step = ToolExecutionStep(
            step_number=1,
            tool_name="calculator_tool",
            tool_input={"expression": "1250 * 18 + 500"},
            tool_output_summary="23000",
        )
        log_str = step.to_log_string()
        assert "[Step 1]" in log_str
        assert "calculator_tool" in log_str
        assert "1250 * 18 + 500" in log_str
        assert "23000" in log_str
