"""Interactive terminal REPL for testing the agent brain in text mode."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from parley.config import get_settings, load_business
from parley.core.brain.events import Control, Filler, SpeakDelta, ToolFinished, TurnDone
from parley.core.brain.session import Agent

if TYPE_CHECKING:
    from parley.core.brain.state import CallState
    from parley.core.providers.llm_base import LLM


async def run_repl(
    business_path: str | None = None,
    caller_number: str | None = None,
    llm: LLM | None = None,
    input_fn: Callable[[str], str] = input,
) -> CallState:
    """Run an interactive text-mode call session in the terminal."""
    load_dotenv()
    settings = get_settings()
    business = load_business(business_path or settings.business_config)
    agent = await Agent.create(settings, business, llm=llm)
    session = agent.new_session(channel="text", caller_number=caller_number)

    print("agent> ", end="", flush=True)
    async for ev in session.start():
        if isinstance(ev, SpeakDelta):
            print(ev.text, end="", flush=True)
    print(flush=True)

    while not session.state.ended:
        try:
            line = await asyncio.to_thread(input_fn, "you> ")
        except (EOFError, KeyboardInterrupt):
            break

        cmd = line.strip()
        if not cmd:
            continue
        if cmd == "/quit":
            break
        if cmd == "/state":
            print(session.state.model_dump_json(exclude={"transcript"}, indent=2), flush=True)
            continue

        printing_speech = False
        stop_loop = False
        async for ev in session.respond(line):
            if isinstance(ev, Filler):
                if printing_speech:
                    print(flush=True)
                    printing_speech = False
                print(f"(filler: {ev.text})", flush=True)
            elif isinstance(ev, ToolFinished):
                if printing_speech:
                    print(flush=True)
                    printing_speech = False
                print(
                    f"\x1b[2m  [tool {ev.event.name} -> {ev.event.status}]\x1b[0m",
                    flush=True,
                )
            elif isinstance(ev, SpeakDelta):
                if not printing_speech:
                    print("agent> ", end="", flush=True)
                    printing_speech = True
                print(ev.text, end="", flush=True)
            elif isinstance(ev, Control):
                if printing_speech:
                    print(flush=True)
                    printing_speech = False
                tag = "[transfer]" if ev.action == "transfer" else "[end call]"
                print(tag, flush=True)
                stop_loop = True
            elif isinstance(ev, TurnDone):
                if printing_speech:
                    print(flush=True)
                    printing_speech = False

        if stop_loop or session.state.ended:
            break

    state = await session.end()
    print(f"Summary:\n{state.summary}\nDisposition: {state.disposition}", flush=True)
    return state
