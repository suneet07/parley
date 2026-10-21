"""Business configuration: the YAML file that turns the generic agent into a specific business.

A new business needs a `business.yaml`, its docs, and a tools module. No core code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceConfig(BaseModel):
    provider: Literal["elevenlabs", "cartesia", "deepgram", "piper"] = "elevenlabs"
    voice_id: str = ""
    speaking_rate: float = 1.0


class HoursConfig(BaseModel):
    tz: str = "UTC"
    open: str = "09:00"  # HH:MM local time
    close: str = "18:00"
    days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # 0 = Monday


class PoliciesConfig(BaseModel):
    """Generic policy knobs. Businesses may add extra keys; custom policy checks read them."""

    model_config = ConfigDict(extra="allow")

    currency: str = "INR"
    refund_max: float | None = None
    return_window_days: int | None = None


EscalationTrigger = Literal[
    "human_requested", "anger", "policy_exception", "repeated_failure", "low_confidence"
]


class EscalationConfig(BaseModel):
    transfer_to: str | None = None  # E.164 number for warm transfer
    on: list[EscalationTrigger] = Field(
        default_factory=lambda: ["human_requested", "policy_exception", "repeated_failure"]
    )


class BusinessConfig(BaseModel):
    business: str
    persona: str
    greeting: str | None = None  # spoken first; AI disclosure is prepended automatically
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    hours: HoursConfig = Field(default_factory=HoursConfig)
    knowledge: list[str] = Field(default_factory=list)  # paths relative to the yaml file
    tools_module: str = "tools.py"  # path relative to the yaml file
    tools: list[str] = Field(default_factory=list)  # enabled tool names (built-ins + module)
    policies: PoliciesConfig = Field(default_factory=PoliciesConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    max_call_seconds: int = 600

    # Set by the loader, not by the YAML.
    base_dir: Path = Field(default=Path("."), exclude=True)

    def resolve(self, rel: str) -> Path:
        return (self.base_dir / rel).resolve()


def load_business(path: str | Path) -> BusinessConfig:
    p = Path(path).resolve()
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = BusinessConfig.model_validate(data)
    cfg.base_dir = p.parent
    return cfg


class Settings(BaseSettings):
    """Process-wide settings from environment / .env. API keys live here, never in YAML."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    business_config: str = "examples/kettle-and-co/business.yaml"
    public_base_url: str = "http://localhost:8000"  # used for Twilio webhooks (ngrok URL in dev)

    # LLM fallback chain, comma-separated provider names, tried in order.
    llm_chain: str = "groq,cerebras,gemini,openai"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    cerebras_api_key: str = ""
    cerebras_model: str = "llama-3.3-70b"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""
    cartesia_api_key: str = ""
    cartesia_voice_id: str = ""

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    database_url: str = "sqlite+aiosqlite:///./parley.db"
    redis_url: str = "redis://localhost:6379"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_exporter_otlp_endpoint: str = ""

    tenant_id: str = "default"


def get_settings() -> Settings:
    return Settings()
