from __future__ import annotations

from typing import Any

import structlog
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyFailover
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transcriptions.language import Language

from parley.config import BusinessConfig, Settings

logger = structlog.get_logger(__name__)

_DEFAULT_ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"
_DEFAULT_DEEPGRAM_VOICE = "aura-2-thalia-en"


def build_stt(settings: Settings) -> FrameProcessor:
    """Build STT processor (Deepgram Nova-3 primary, Groq Whisper fallback via ServiceSwitcher)."""
    services: list[Any] = []
    active_names: list[str] = []

    if settings.deepgram_api_key:
        # In Pipecat 1.13, Deepgram live_options is deprecated in favor of DeepgramSTTService.Settings;
        # Nova-3 uses language="multi" rather than "en-IN" for full feature support.
        services.append(
            DeepgramSTTService(
                api_key=settings.deepgram_api_key,
                settings=DeepgramSTTService.Settings(
                    model="nova-3",
                    language="multi",
                    interim_results=True,
                    smart_format=True,
                    punctuate=True,
                ),
            )
        )
        active_names.append("deepgram:nova-3")

    if settings.groq_api_key:
        # In Pipecat 1.13, GroqSTTService takes GroqSTTService.Settings for model/language.
        services.append(
            GroqSTTService(
                api_key=settings.groq_api_key,
                settings=GroqSTTService.Settings(
                    model="whisper-large-v3-turbo",
                    language=Language.EN,
                ),
            )
        )
        active_names.append("groq:whisper-large-v3-turbo")

    if not services:
        raise RuntimeError("No STT provider configured: set DEEPGRAM_API_KEY or GROQ_API_KEY.")

    logger.info("stt_configured", providers=active_names)
    if len(services) == 1:
        return services[0]
    # ServiceSwitcher with ServiceSwitcherStrategyFailover switches automatically on provider error.
    return ServiceSwitcher(services=services, strategy_type=ServiceSwitcherStrategyFailover)


def build_tts(settings: Settings, business: BusinessConfig) -> FrameProcessor:
    """Build TTS processor ordered by business.voice.provider with failover across available keys."""
    preferred = business.voice.provider
    ordered_providers: list[str] = []
    for candidate in (preferred, "elevenlabs", "cartesia", "deepgram"):
        if (
            candidate in ("elevenlabs", "cartesia", "deepgram")
            and candidate not in ordered_providers
        ):
            ordered_providers.append(candidate)

    services: list[Any] = []
    active_names: list[str] = []

    for provider in ordered_providers:
        if provider == "elevenlabs" and settings.elevenlabs_api_key:
            voice_id = business.voice.voice_id or _DEFAULT_ELEVENLABS_VOICE
            speed = min(1.2, max(0.7, float(business.voice.speaking_rate or 1.0)))
            # In Pipecat 1.13, ElevenLabsTTSService uses ElevenLabsTTSService.Settings.
            services.append(
                ElevenLabsTTSService(
                    api_key=settings.elevenlabs_api_key,
                    settings=ElevenLabsTTSService.Settings(
                        model="eleven_flash_v2_5",
                        voice=voice_id,
                        speed=speed,
                    ),
                )
            )
            active_names.append("elevenlabs:eleven_flash_v2_5")
        elif provider == "cartesia" and settings.cartesia_api_key:
            cartesia_voice = settings.cartesia_voice_id or (
                business.voice.voice_id if preferred == "cartesia" else ""
            )
            services.append(
                CartesiaTTSService(
                    api_key=settings.cartesia_api_key,
                    settings=CartesiaTTSService.Settings(voice=cartesia_voice),
                )
            )
            active_names.append("cartesia")
        elif provider == "deepgram" and settings.deepgram_api_key:
            services.append(
                DeepgramTTSService(
                    api_key=settings.deepgram_api_key,
                    settings=DeepgramTTSService.Settings(voice=_DEFAULT_DEEPGRAM_VOICE),
                )
            )
            active_names.append(f"deepgram:{_DEFAULT_DEEPGRAM_VOICE}")

    if not services:
        raise RuntimeError(
            "No TTS provider configured: set ELEVENLABS_API_KEY, CARTESIA_API_KEY, or DEEPGRAM_API_KEY."
        )

    logger.info("tts_configured", providers=active_names)
    if len(services) == 1:
        return services[0]
    return ServiceSwitcher(services=services, strategy_type=ServiceSwitcherStrategyFailover)
