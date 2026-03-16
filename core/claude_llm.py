"""
Claude API provider for Tali voice assistant.
Uses Claude Haiku 3.5 for fast, high-quality responses with tool support.
Includes cost tracking with configurable monthly spending limit.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import anthropic

from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS,
    CLAUDE_MONTHLY_LIMIT, GRAY, CYAN, GREEN, YELLOW, RESET,
)

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Pricing per million tokens (as of 2026)
_PRICING = {
    "claude-haiku-4-20250414": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Fallback
    "default": {"input": 0.80, "output": 4.00},
}

_USAGE_FILE = Path(__file__).parent.parent / "data" / "claude_usage.json"


def _load_usage() -> dict:
    """Load monthly usage data from disk."""
    if _USAGE_FILE.exists():
        try:
            with open(_USAGE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"month": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "requests": 0}


def _save_usage(data: dict):
    """Persist usage data."""
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_USAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single request."""
    prices = _PRICING.get(model, _PRICING["default"])
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


class CostTracker:
    """Tracks Claude API spending and enforces monthly limit."""

    def __init__(self, monthly_limit: float):
        self.monthly_limit = monthly_limit
        self._usage = _load_usage()
        # Reset if new month
        if self._usage.get("month") != _current_month():
            self._usage = {
                "month": _current_month(),
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            }
            _save_usage(self._usage)

    @property
    def cost_usd(self) -> float:
        # Reset if month rolled over
        if self._usage.get("month") != _current_month():
            self._usage = {
                "month": _current_month(),
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            }
            _save_usage(self._usage)
        return self._usage["cost_usd"]

    @property
    def remaining_usd(self) -> float:
        return max(0, self.monthly_limit - self.cost_usd)

    @property
    def limit_reached(self) -> bool:
        return self.cost_usd >= self.monthly_limit

    def record(self, model: str, input_tokens: int, output_tokens: int):
        """Record usage for a single request."""
        cost = _calculate_cost(model, input_tokens, output_tokens)
        self._usage["input_tokens"] += input_tokens
        self._usage["output_tokens"] += output_tokens
        self._usage["cost_usd"] = round(self._usage["cost_usd"] + cost, 6)
        self._usage["requests"] += 1
        self._usage["month"] = _current_month()
        _save_usage(self._usage)
        print(f"{GRAY}[Claude] Request cost: ${cost:.4f} | Month total: ${self._usage['cost_usd']:.4f} / ${self.monthly_limit}{RESET}")

    def summary(self) -> str:
        return (
            f"Claude usage this month: ${self.cost_usd:.4f} / ${self.monthly_limit:.2f} "
            f"({self._usage['requests']} requests, "
            f"{self._usage['input_tokens']}+{self._usage['output_tokens']} tokens)"
        )


# ---------------------------------------------------------------------------
# Claude LLM Client
# ---------------------------------------------------------------------------

# System prompt for Tali
TALI_SYSTEM_PROMPT = (
    "You are Tali, a voice assistant. Your responses will be spoken aloud via TTS. "
    "CRITICAL RULES: "
    "1) LANGUAGE: Detect which language the user wrote in and ALWAYS reply in that SAME language. "
    "English input = English reply. Portuguese input = Portuguese reply. "
    "NEVER mix languages in a single response. If unsure, default to English. "
    "2) Keep responses SHORT — 1-2 sentences for greetings, 2-3 sentences max for questions. "
    "3) NEVER use emojis, markdown, bullet points, numbered lists, asterisks, or special characters. "
    "Write plain spoken text only — no formatting whatsoever. "
    "4) Be direct, warm, and conversational. "
    "5) Never say 'as an AI' or 'I don't have feelings'. Just be natural."
)


class ClaudeLLM:
    """Claude API client with streaming, tool support, and cost tracking."""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL
        self.max_tokens = CLAUDE_MAX_TOKENS
        self.cost_tracker = CostTracker(CLAUDE_MONTHLY_LIMIT)
        print(f"{GREEN}[Claude] ✓ Initialized ({self.model}){RESET}")
        print(f"{CYAN}[Claude] {self.cost_tracker.summary()}{RESET}")

    def stream_response(
        self,
        messages: list[dict],
        system_prompt: str = None,
        stop_event=None,
    ) -> Generator[dict, None, None]:
        """Stream a Claude response, yielding chunks.

        Yields dicts with keys:
            {"type": "text", "content": "..."}
            {"type": "thinking", "content": "..."}
            {"type": "done", "full_response": "...", "input_tokens": N, "output_tokens": N}
            {"type": "error", "message": "..."}
            {"type": "limit_reached", "message": "..."}
        """
        # Check cost limit
        if self.cost_tracker.limit_reached:
            yield {
                "type": "limit_reached",
                "message": f"Monthly Claude API limit of ${self.cost_tracker.monthly_limit:.2f} reached. "
                           f"Current spend: ${self.cost_tracker.cost_usd:.2f}. "
                           f"Limit resets on the 1st of next month.",
            }
            return

        system = system_prompt or TALI_SYSTEM_PROMPT

        # Convert messages from Ollama format to Claude format
        claude_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue  # System is passed separately
            if role in ("user", "assistant"):
                claude_messages.append({"role": role, "content": content})

        if not claude_messages:
            yield {"type": "error", "message": "No messages to send"}
            return

        try:
            full_response = ""
            input_tokens = 0
            output_tokens = 0

            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=claude_messages,
            ) as stream:
                for event in stream:
                    if stop_event and stop_event.is_set():
                        break

                    if hasattr(event, 'type'):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, 'text'):
                                text = event.delta.text
                                full_response += text
                                yield {"type": "text", "content": text}
                            elif hasattr(event.delta, 'thinking'):
                                yield {"type": "thinking", "content": event.delta.thinking}

                # Get final usage from the stream
                final_message = stream.get_final_message()
                if final_message and final_message.usage:
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens

            # Record cost
            self.cost_tracker.record(self.model, input_tokens, output_tokens)

            yield {
                "type": "done",
                "full_response": full_response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except anthropic.APIError as e:
            yield {"type": "error", "message": f"Claude API error: {e}"}
        except Exception as e:
            yield {"type": "error", "message": f"Claude error: {e}"}


# Global instance (lazy — created on first use)
_claude_instance = None


def get_claude() -> ClaudeLLM:
    """Get or create the global Claude LLM instance."""
    global _claude_instance
    if _claude_instance is None:
        _claude_instance = ClaudeLLM()
    return _claude_instance
