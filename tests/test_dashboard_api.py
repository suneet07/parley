from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from parley.api.server import create_app
from parley.api.store import CallStore
from parley.config import load_business
from parley.core.brain.state import CallState, Turn


class DummyAgent:
    pass


@pytest.mark.asyncio
async def test_dashboard_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "dashboard.db"
    store = CallStore(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await store.init()

    # Pre-fill store with 3 CallStates
    now = datetime.now(UTC)
    state1 = CallState(
        call_id="call-1",
        tenant_id="default",
        started_at=now,
        direction="inbound",
        channel="phone",
        caller_number="+1234567890",
        disposition="resolved",
        summary="Customer asked for returns and issue was resolved.",
        intents=["returns"],
        escalated=False,
        transcript=[Turn(role="user", text="How do I return?", at=now)],
        knowledge_gaps=["Warranty period on kettles?"],
    )
    await store.save(
        state1,
        metrics={
            "turns": [
                {
                    "turn": 1,
                    "stt_ms": 100.0,
                    "llm_ttft_ms": 200.0,
                    "tts_ttfb_ms": 100.0,
                    "e2e_ms": 500.0,
                    "interrupted": False,
                }
            ],
            "p50_e2e_ms": 500.0,
            "p95_e2e_ms": 500.0,
            "count": 1,
        },
    )

    state2 = CallState(
        call_id="call-2",
        tenant_id="default",
        started_at=now,
        direction="inbound",
        channel="phone",
        caller_number="+1987654321",
        disposition="escalated",
        escalation_reason="human requested",
        summary="Customer demanded to speak with a human agent.",
        intents=["order_status"],
        escalated=True,
        transcript=[
            Turn(role="user", text="I want to talk to a human.", at=now),
            Turn(role="assistant", text="Connecting you now.", at=now, interrupted=True),
        ],
    )
    await store.save(
        state2,
        metrics={
            "turns": [
                {
                    "turn": 1,
                    "stt_ms": 120.0,
                    "llm_ttft_ms": 250.0,
                    "tts_ttfb_ms": 110.0,
                    "e2e_ms": 700.0,
                    "interrupted": False,
                },
                {
                    "turn": 2,
                    "stt_ms": 150.0,
                    "llm_ttft_ms": 300.0,
                    "tts_ttfb_ms": 120.0,
                    "e2e_ms": 900.0,
                    "interrupted": True,
                },
            ],
            "p50_e2e_ms": 800.0,
            "p95_e2e_ms": 900.0,
            "count": 2,
        },
    )

    state3 = CallState(
        call_id="call-3",
        tenant_id="default",
        started_at=now,
        direction="inbound",
        channel="webrtc",
        caller_number="+1122334455",
        disposition="ticket_created",
        summary="Created support ticket for broken handle.",
        intents=["returns"],
        escalated=False,
        transcript=[Turn(role="user", text="Handle broken", at=now)],
        knowledge_gaps=["warranty period on kettles?"],
    )
    await store.save(
        state3,
        metrics={
            "turns": [
                {
                    "turn": 1,
                    "stt_ms": 90.0,
                    "llm_ttft_ms": 180.0,
                    "tts_ttfb_ms": 90.0,
                    "e2e_ms": 400.0,
                    "interrupted": False,
                }
            ],
            "p50_e2e_ms": 400.0,
            "p95_e2e_ms": 400.0,
            "count": 1,
        },
    )

    app = create_app(agent=DummyAgent(), store=store)  # type: ignore[arg-type]

    with TestClient(app) as client:
        # 1. /api/metrics verification
        metrics_resp = client.get("/api/metrics?days=7")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()
        assert metrics["calls"] == 3
        assert metrics["resolved"] == 2
        assert metrics["escalated"] == 1
        assert metrics["resolution_rate"] == pytest.approx(2 / 3, 0.001)
        assert metrics["escalation_rate"] == pytest.approx(1 / 3, 0.001)
        assert metrics["p50_e2e_ms"] == 500.0
        assert metrics["p95_e2e_ms"] == 900.0
        assert len(metrics["knowledge_gaps"]) == 1
        assert metrics["knowledge_gaps"][0]["count"] == 2

        # 2. /api/evals/latest verification
        eval_missing = client.get("/api/evals/latest?suite=smoke")
        assert eval_missing.status_code == 404
        assert eval_missing.json()["detail"] == "no eval report yet"

        eval_bad = client.get("/api/evals/latest?suite=../x")
        assert eval_bad.status_code == 400

        reports_dir = tmp_path / "evals" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        sample_report = {
            "suite": "smoke",
            "model": "test-llm",
            "created_at": "2026-10-21T00:00:00Z",
            "commit": None,
            "metrics": {
                "resolution_rate": 0.95,
                "policy_violation_rate": 0.0,
                "escalation_accuracy": 1.0,
                "injection_held": 10,
                "injection_total": 10,
                "hallucinations_per_100": 0.0,
                "passed": 10,
                "total": 10,
            },
            "categories": [],
            "regressions": [],
            "fixed": [],
            "results": [],
        }
        (reports_dir / "latest-smoke.json").write_text(json.dumps(sample_report), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        eval_ok = client.get("/api/evals/latest?suite=smoke")
        assert eval_ok.status_code == 200
        assert eval_ok.json() == sample_report

        # 3. /api/config verification
        kettle_src = Path(__file__).resolve().parents[1] / "examples" / "kettle-and-co"
        kettle_tmp = tmp_path / "kettle-and-co"
        shutil.copytree(kettle_src, kettle_tmp)
        business = load_business(kettle_tmp / "business.yaml")
        app.state.business = business

        config_resp = client.get("/api/config")
        assert config_resp.status_code == 200
        config_data = config_resp.json()
        assert config_data["business"]["business"] == "Kettle & Co"
        tool_names = [t["name"] for t in config_data["tools"]]
        assert "verify_identity" in tool_names
        assert len(config_data["knowledge"]) > 0
        assert "faq.md" in config_data["knowledge"]

    await store.close()
