# parley

Open-source, self-hostable voice agent for customer support. It answers and makes phone calls for a
business, takes real actions through typed tools, hands off to a human when it should, and is tested
against simulated customers before it talks to a real one.

> Status: early development. APIs and config format may change.

## What it does

- **Real-time voice loop** — streaming STT → agent brain → streaming TTS, with barge-in (the caller can
  interrupt at any time) and semantic end-of-turn detection. Target: under 800 ms from the caller
  finishing a sentence to hearing the first word back.
- **Phone and browser** — Twilio Media Streams for phone calls, WebRTC for a browser demo.
- **Pluggable providers with fallback** — Deepgram / Groq Whisper for STT, Groq / Cerebras / Gemini /
  OpenAI for the LLM, ElevenLabs / Cartesia / Deepgram Aura for TTS. If one fails, the next takes over.
- **Answers from your docs** — hybrid retrieval (BM25 + embeddings) over the business's own markdown
  docs. It doesn't state prices, policies or order details that weren't retrieved or returned by a tool.
- **Real actions, safely** — tools are plain Python functions with Pydantic arguments. Policies (refund
  limits, return windows, verify-before-action) are enforced in code, not by the LLM. Write actions need
  the caller's spoken confirmation, and the agent can't confirm on its own behalf.
- **Human handoff** — warm transfer where the human hears a short summary before being connected.
- **Simulated-customer evals** — LLM-played customers (happy paths, policy edge cases, angry callers,
  prompt injection, social engineering) run in CI, with a scored report per release.
- **Outbound calls with compliance on by default** — webhook or CSV triggers, calling hours in the
  callee's time zone, consent evidence, do-not-call list (including SMS STOP), daily and weekly caps,
  retries, voicemail drop, and an audit log of every dial decision.
- **Ops console** — call log with transcripts, live calls with operator takeover, insights, per-stage
  latency, eval results, cost per call.
- **Configured per business with YAML** — persona, hours, knowledge docs, tools, policies, escalation.

## Quickstart

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> parley && cd parley
uv sync
cp .env.example .env   # add your API keys
```

Talk to the demo business ("Kettle & Co", a fictional kitchenware store) in the terminal:

```bash
uv run parley chat
```

Run the server with the browser voice demo at http://localhost:8000:

```bash
uv run parley serve
```

Run the eval smoke suite:

```bash
uv run parley eval --suite smoke
```

### Phone calls (Twilio)

1. Expose the server publicly (for example `cloudflared tunnel --url http://localhost:8000` or ngrok)
   and set `PUBLIC_BASE_URL` in `.env` to that URL.
2. In the Twilio console, set your number's "A call comes in" webhook to `POST <PUBLIC_BASE_URL>/twilio/voice`.
3. Call the number.

### Dashboard

```bash
cd dashboard && npm install && npm run dev
```

Opens at http://localhost:3000 and reads from the API on :8000. To preview it without real calls,
seed some fictional ones first: `uv run python scripts/seed_demo_calls.py`.

### Outbound calls

Set `OUTBOUND_WEBHOOK_SECRET`, run a worker (`uv run parley worker`, needs Redis), and POST a signed
job to `/api/outbound/trigger`:

```python
import hashlib, hmac, json, time, httpx

body = json.dumps({
    "contact": {"phone": "+919876543210", "name": "Aarav", "consent": True,
                "consent_source": "checkout_form", "consent_at": "2026-10-01T10:00:00Z"},
    "goal": "delivery issue",
    "idempotency_key": "order-10482-delivery",
}).encode()
ts = str(int(time.time()))
sig = hmac.new(SECRET.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
httpx.post("http://localhost:8000/api/outbound/trigger", content=body,
           headers={"X-Parley-Timestamp": ts, "X-Parley-Signature": f"sha256={sig}"})
```

Calls are only placed inside the callee's local calling window, with recorded consent, and never to
numbers on the do-not-call list. Numbers whose time zone can't be determined are not called unless
the contact carries an explicit `tz`.

### Security checklist for a public deployment

- Set `TWILIO_AUTH_TOKEN` (Twilio webhooks are rejected without a valid signature).
- Set `PARLEY_API_KEY`; the dashboard sends it server-side via `dashboard/.env`.
- Set `DASHBOARD_USER` / `DASHBOARD_PASSWORD` so the console asks for a login.
- For a public demo number, keep `MAX_CALLS_PER_CALLER_PER_DAY` low and know where
  `PARLEY_KILL_SWITCH` is.

### Docker

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Starts the API with Postgres (pgvector) and Redis.

## Configure your own business

Copy `examples/kettle-and-co/` and edit:

- `business.yaml` — name, persona, hours, voice, knowledge files, enabled tools, policies, escalation
- `docs/*.md` — the knowledge the agent answers from
- `tools.py` — your tools (`@tool`) and code-enforced policies (`@policy`), or use the generic webhook
  adapter in `parley/core/tools/webhook.py` to call your existing API

Then point `BUSINESS_CONFIG` in `.env` at your `business.yaml`.

## Project layout

```
parley/
  config.py         business YAML schema + environment settings
  core/
    pipeline/       Pipecat voice pipeline, barge-in, latency tracking
    brain/          call state, guard, prompts, LangGraph turn handler, sessions
    tools/          tool registry, policy engine, confirmation gateway, built-ins, webhook adapter
    knowledge/      chunking, embeddings, hybrid retrieval (in-memory or pgvector)
    providers/      LLM providers + fallback chain
    outbound/       dial queue, retries, compliance rules
    telemetry/      PII redaction, tracing
  api/              FastAPI server: WebRTC demo, Twilio webhooks, call log API
evals/              simulated customers, scenarios, scorers, reports
examples/           demo business
deploy/             Dockerfile, docker-compose
```

## Responsible use

Outbound calling is regulated (consent, calling hours, do-not-call lists) and the rules differ by
country. Parley ships a compliance module that is on by default, but **you are responsible for
complying with the laws that apply to you**. The agent always discloses that it is an AI at the start
of a call. Don't use this project for unsolicited or spam calls.

## License

Apache-2.0. See [LICENSE](LICENSE).
