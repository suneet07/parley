export type CallRow = {
  id: string;
  started_at: string;
  ended_at: string | null;
  direction: "inbound" | "outbound";
  channel: "phone" | "webrtc" | "text";
  caller_number: string | null;
  disposition: string;
  summary: string | null;
  intents: string[];
  escalated: boolean;
  duration_s: number | null;
};

export type ToolEvent = {
  name: string;
  args: Record<string, unknown>;
  status: "ok" | "error" | "blocked" | "confirmation_required";
  result: Record<string, unknown>;
  reason: string | null;
  turn: number;
  latency_ms: number;
};

export type Turn = {
  role: "user" | "assistant" | "system";
  text: string;
  at: string;
  interrupted: boolean;
  tool_events: ToolEvent[];
};

export type TurnLatency = {
  turn: number;
  stt_ms: number | null;
  llm_ttft_ms: number | null;
  tts_ttfb_ms: number | null;
  e2e_ms: number | null;
  interrupted: boolean;
};

export type CallDetail = CallRow & {
  state: {
    transcript: Turn[];
    tool_events: ToolEvent[];
    verified: boolean;
    escalation_reason: string | null;
    knowledge_gaps: string[];
    follow_ups: string[];
    summary: string | null;
    disposition: string;
    caller: { name?: string | null } | null;
  };
  metrics: {
    turns: TurnLatency[];
    p50_e2e_ms: number | null;
    p95_e2e_ms: number | null;
    count: number;
  } | null;
};

export type Metrics = {
  days: number;
  calls: number;
  resolved: number;
  escalated: number;
  resolution_rate: number | null;
  escalation_rate: number | null;
  avg_duration_s: number | null;
  p50_e2e_ms: number | null;
  p95_e2e_ms: number | null;
  stage_avg_ms: {
    stt: number | null;
    llm: number | null;
    tts: number | null;
  };
  intents: { name: string; count: number }[];
  dispositions: { name: string; count: number }[];
  knowledge_gaps: { question: string; count: number }[];
  by_day: {
    date: string;
    calls: number;
    resolved: number;
    escalated: number;
  }[];
  slowest_turns: (TurnLatency & { call_id: string })[];
};

export type EvalReport = {
  suite: string;
  model: string;
  created_at: string;
  commit: string | null;
  metrics: {
    resolution_rate: number | null;
    policy_violation_rate: number | null;
    escalation_accuracy: number | null;
    injection_held: number;
    injection_total: number;
    hallucinations_per_100: number | null;
    passed: number;
    total: number;
  };
  categories: { name: string; passed: number; total: number }[];
  regressions: string[];
  fixed: string[];
  results: {
    scenario_id: string;
    category: string;
    passed: boolean;
    turns: number;
    duration_s: number;
    policy_violation: boolean;
    escalated: boolean;
    resolution: string;
    error: string | null;
    hallucinations: string[];
    checks: {
      name: string;
      passed: boolean;
      detail: string;
      kind: "code" | "judge";
      severity: "critical" | "major" | "minor";
    }[];
    transcript: {
      role: "user" | "assistant";
      text: string;
      interrupted: boolean;
    }[];
    tool_events: {
      name: string;
      args: Record<string, unknown>;
      status: string;
      turn: number;
      reason: string | null;
    }[];
  }[];
};

export type ConfigView = {
  business: Record<string, unknown> & {
    business: string;
    persona: string;
    policies: Record<string, unknown>;
    hours: { tz: string; open: string; close: string };
    escalation: { transfer_to: string | null; on: string[] };
  };
  tools: {
    name: string;
    kind: string;
    description: string;
    requires_verification: boolean;
    requires_confirmation: boolean;
  }[];
  knowledge: string[];
};
