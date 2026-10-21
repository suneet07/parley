from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    EndTaskFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_transport import BaseTransport

from parley.config import BusinessConfig, Settings, get_settings, load_business
from parley.core.brain.state import CallState
from parley.core.pipeline.brain_processor import BrainProcessor, Control
from parley.core.pipeline.latency import LatencyTracker
from parley.core.pipeline.services import build_stt, build_tts

if TYPE_CHECKING:
    from parley.api.store import CallStore
    from parley.core.brain import Agent

logger = structlog.get_logger(__name__)

_TRANSFER_FALLBACK_TEXT = (
    "I'm unable to connect you to a person right now, "
    "so I've created a ticket and someone will call you back."
)
_TIME_LIMIT_TEXT = "We've reached the time limit for this call. Goodbye."


@dataclass
class CallControl:
    """Telephony actions the pipeline can request.

    Phone calls pass a Twilio-backed implementation; WebRTC passes None
    (transfer just ends the demo call with a message).
    """

    transfer: Callable[[str, str | None], Awaitable[bool]]


async def run_call(
    transport: BaseTransport,
    *,
    agent: Agent,
    call_id: str,
    channel: Literal["phone", "webrtc"],
    direction: Literal["inbound", "outbound"] = "inbound",
    caller_number: str | None = None,
    audio_sample_rate: int | None = None,
    control: CallControl | None = None,
    store: CallStore | None = None,
    outbound_goal: str | None = None,
) -> CallState:
    """Run a single real-time voice call through the Pipecat 1.13 pipeline."""
    settings: Settings = getattr(agent, "settings", None) or get_settings()
    business: BusinessConfig = getattr(agent, "business", None) or load_business(
        settings.business_config
    )

    session = agent.new_session(
        call_id,
        direction=direction,
        channel=channel,
        caller_number=caller_number or "",
        outbound_goal=outbound_goal,
    )

    stt = build_stt(settings)
    tts = build_tts(settings, business)

    context = LLMContext()
    # In Pipecat 1.13, VAD lives on LLMUserAggregatorParams (not TransportParams),
    # and UserTurnStrategies defaults to Smart Turn v3 so VAD stop_secs=0.2 hands off quickly.
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_idle_timeout=6.0,
        ),
    )

    latency = LatencyTracker()
    end_reason = "hangup"
    task_ref: dict[str, PipelineTask] = {}

    async def on_control(ctrl: Control) -> None:
        nonlocal end_reason
        pipeline_task = task_ref.get("task")
        if ctrl.action == "end_call":
            end_reason = "agent_end"
            # Push EndTaskFrame upstream so the source queues an EndFrame behind in-flight speech.
            await brain.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
            if pipeline_task is not None:
                await pipeline_task.queue_frame(EndFrame())
        elif ctrl.action == "transfer":
            end_reason = "transferred"
            transferred = False
            if control is not None:
                try:
                    transferred = await control.transfer(call_id, ctrl.summary)
                except Exception:
                    logger.exception("call_transfer_error", call_id=call_id)
                    transferred = False
            if not transferred:
                await brain.push_frame(LLMFullResponseStartFrame())
                await brain.push_frame(LLMTextFrame(_TRANSFER_FALLBACK_TEXT))
                await brain.push_frame(LLMFullResponseEndFrame())
                await brain.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                if pipeline_task is not None:
                    await pipeline_task.queue_frame(EndFrame())

    brain = BrainProcessor(session, on_control=on_control, latency=latency)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            pair.user(),
            brain,
            tts,
            transport.output(),
            pair.assistant(),
        ]
    )

    # In Pipecat 1.13, PipelineParams audio_in_sample_rate / audio_out_sample_rate are non-null ints.
    params_kwargs: dict[str, Any] = {
        "enable_metrics": True,
        "enable_usage_metrics": True,
    }
    if audio_sample_rate is not None:
        params_kwargs["audio_in_sample_rate"] = audio_sample_rate
        params_kwargs["audio_out_sample_rate"] = audio_sample_rate

    task = PipelineTask(
        pipeline,
        params=PipelineParams(**params_kwargs),
        observers=[latency],
        idle_timeout_secs=business.max_call_seconds + 30,
    )
    task_ref["task"] = task

    silence_count = 0

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_obj: Any, client: Any) -> None:
        logger.info("client_connected", call_id=call_id, channel=channel)
        await brain.greet()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_obj: Any, client: Any) -> None:
        logger.info("client_disconnected", call_id=call_id, channel=channel)
        await task.cancel()

    @pair.user().event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator: Any, strategy: Any, message: UserTurnStoppedMessage
    ) -> None:
        nonlocal silence_count
        silence_count = 0

    @pair.user().event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator: Any) -> None:
        nonlocal silence_count
        silence_count += 1
        await brain.handle_idle(silence_count)

    @pair.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(
        aggregator: Any, message: AssistantTurnStoppedMessage
    ) -> None:
        if message.interrupted:
            session.on_interrupted(message.content)
            latency.mark_interrupted()

    async def _max_duration_watchdog() -> None:
        nonlocal end_reason
        try:
            await asyncio.sleep(business.max_call_seconds)
            end_reason = "timeout"
            logger.info("call_max_duration_reached", call_id=call_id)
            await brain.push_frame(LLMFullResponseStartFrame())
            await brain.push_frame(LLMTextFrame(_TIME_LIMIT_TEXT))
            await brain.push_frame(LLMFullResponseEndFrame())
            await task.queue_frame(EndFrame())
        except asyncio.CancelledError:
            return

    watchdog_task = asyncio.create_task(_max_duration_watchdog())
    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    except Exception:
        logger.exception("pipeline_runner_error", call_id=call_id)
    finally:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

        state: CallState = session.state
        try:
            state = await session.end(reason=end_reason)
        except Exception:
            logger.exception("session_end_error", call_id=call_id)

        metrics = latency.summary()
        if store is not None:
            try:
                await store.save(state, metrics)
            except Exception:
                logger.exception("store_save_error", call_id=call_id)

    return state
