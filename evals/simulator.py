"""Customer simulator for parley evaluations using an LLM-played customer."""

from __future__ import annotations

import re

from evals.scenario import Scenario
from parley.core.brain.state import Turn
from parley.core.providers.llm_base import LLM, ChatMessage, TextDelta


class CustomerSimulator:
    """Simulates a customer on a support call following a scenario."""

    def __init__(self, scenario: Scenario, llm: LLM, temperature: float = 0.7) -> None:
        self.scenario = scenario
        self.llm = llm
        self.temperature = temperature
        self._first_turn = True
        self._hung_up = False

    def _build_system_prompt(self) -> str:
        prompt_lines = [
            "You are a CUSTOMER on a phone call with Kettle & Co's customer support line. "
            "You are NOT the support agent.",
            f"Persona: {self.scenario.persona}",
            f"Goal: {self.scenario.goal}",
        ]

        if self.scenario.behavior:
            prompt_lines.append("Behaviors to exhibit:")
            for b in self.scenario.behavior:
                prompt_lines.append(f"- {b}")

        if self.scenario.knows:
            prompt_lines.append("Facts you know (only reveal when asked or natural):")
            for k, v in self.scenario.knows.items():
                prompt_lines.append(f"- {k}: {v}")
        else:
            prompt_lines.append(
                "You do not know any specific order or account details. "
                "If asked for an order ID or account info, say you do not have it."
            )

        prompt_lines.extend(
            [
                "Phone conversation rules:",
                "1. Speak like a real person on a phone call: 1 to 2 short sentences at a time.",
                "2. Do not use bullet points, numbered lists, markdown, quotes, or stage directions.",
                "3. Never invent order numbers, emails, or phone numbers beyond what is listed in your facts.",
                "   If asked for something you do not know, say you don't have it.",
                "4. When your goal is achieved, clearly refused with no further options, or the agent "
                "transfers you to a human or ends the call, say goodbye and end with [HANGUP] "
                "(e.g. 'Thank you, goodbye [HANGUP]' or simply '[HANGUP]').",
            ]
        )
        return "\n".join(prompt_lines)

    @staticmethod
    def _strip_prefixes_and_quotes(text: str) -> str:
        cleaned = re.sub(
            r"^(?:Customer|User)\s*:\s*", "", text.strip(), flags=re.IGNORECASE
        ).strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    async def next_utterance(self, transcript: list[Turn]) -> str | None:
        """Generate the next customer utterance, or None if the customer hangs up."""
        if self._hung_up:
            return None

        if self._first_turn and self.scenario.opening is not None:
            self._first_turn = False
            return self.scenario.opening.strip()

        self._first_turn = False

        system_prompt = self._build_system_prompt()
        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

        for turn in transcript:
            if turn.role == "assistant":
                messages.append(ChatMessage(role="user", content=turn.text))
            elif turn.role == "user":
                messages.append(ChatMessage(role="assistant", content=turn.text))

        chunks: list[str] = []
        async for event in self.llm.stream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=200,
        ):
            if isinstance(event, TextDelta):
                chunks.append(event.text)

        raw = "".join(chunks).strip()

        if "[HANGUP]" in raw:
            idx = raw.index("[HANGUP]")
            before = raw[:idx].strip()
            before = self._strip_prefixes_and_quotes(before)
            self._hung_up = True
            if before:
                return before
            return None

        final_text = self._strip_prefixes_and_quotes(raw)
        return final_text if final_text else None
