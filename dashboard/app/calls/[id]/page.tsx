"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { Badge } from "@/components/Badge";
import { LatencyRow } from "@/components/LatencyRow";
import { Empty, ErrorBox, Loading } from "@/components/States";
import { Transcript } from "@/components/Transcript";
import { useCall } from "@/lib/api";
import { dispositionTone, fmtDuration, fmtMs, fmtTime, fmtUsd } from "@/lib/format";

export default function CallDetailPage() {
  const params = useParams();
  const id = typeof params?.id === "string" ? params.id : "";
  const { data: call, error, isLoading } = useCall(id);

  if (isLoading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorBox message={error.message || "Failed to load call detail."} />;
  }

  if (!call) {
    return (
      <div>
        <div className="mb-4">
          <Link href="/calls" className="btn">
            Back to calls
          </Link>
        </div>
        <Empty>Call not found.</Empty>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4">
        <Link href="/calls" className="btn">
          Back to calls
        </Link>
      </div>

      <div className="page-header">
        <div>
          <h1 className="page-title mono">{call.id}</h1>
          <div className="flex items-center gap-2 mt-2">
            <Badge tone={dispositionTone(call.disposition)}>{call.disposition}</Badge>
            {call.state.verified ? <Badge tone="ok">verified</Badge> : null}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card">
          <div className="card-head">Transcript</div>
          <Transcript turns={call.state.transcript} />
        </div>

        <div className="flex flex-col gap-6">
          <div className="card card-pad">
            <h2 className="text-sm font-semibold mb-2">Summary</h2>
            <div className="whitespace-pre-line text-sm dim">
              {call.state.summary || call.summary || "No summary recorded."}
            </div>
          </div>

          <div className="card card-pad">
            <h2 className="text-sm font-semibold mb-3">Facts</h2>
            <dl className="grid grid-cols-2 gap-2 text-sm">
              <dt className="dim">Direction</dt>
              <dd className="capitalize">{call.direction}</dd>

              <dt className="dim">Channel</dt>
              <dd className="capitalize">{call.channel}</dd>

              <dt className="dim">Started</dt>
              <dd>{fmtTime(call.started_at)}</dd>

              <dt className="dim">Duration</dt>
              <dd className="num">{fmtDuration(call.duration_s)}</dd>

              {call.metrics?.cost?.total_usd !== undefined &&
              call.metrics?.cost?.total_usd !== null ? (
                <>
                  <dt className="dim">Cost</dt>
                  <dd className="num">{fmtUsd(call.metrics.cost.total_usd, 4)}</dd>
                </>
              ) : null}

              {call.state.escalation_reason ? (
                <>
                  <dt className="dim">Escalation reason</dt>
                  <dd className="text-warn">{call.state.escalation_reason}</dd>
                </>
              ) : null}

              {call.state.caller?.name ? (
                <>
                  <dt className="dim">Caller name</dt>
                  <dd>{call.state.caller.name}</dd>
                </>
              ) : null}
            </dl>

            {call.state.follow_ups && call.state.follow_ups.length > 0 ? (
              <div className="mt-4 pt-3 border-t border-line">
                <h3 className="text-xs font-semibold dim mb-1">Follow-ups</h3>
                <ul className="list-disc pl-5 space-y-1 text-sm">
                  {call.state.follow_ups.map((fu, i) => (
                    <li key={i}>{fu}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>

          <div className="card card-pad">
            <h2 className="text-sm font-semibold mb-2">Knowledge gaps</h2>
            {call.state.knowledge_gaps && call.state.knowledge_gaps.length > 0 ? (
              <ul className="list-disc pl-5 space-y-1 text-sm dim">
                {call.state.knowledge_gaps.map((gap, i) => (
                  <li key={i}>{gap}</li>
                ))}
              </ul>
            ) : (
              <p className="dim text-sm">None recorded</p>
            )}
          </div>

          <div className="card card-pad">
            <h2 className="text-sm font-semibold mb-2">Latency</h2>
            <div className="legend mb-3">
              <span>
                <span className="legend-swatch" data-stage="stt" />
                STT
              </span>
              <span>
                <span className="legend-swatch" data-stage="llm" />
                LLM
              </span>
              <span>
                <span className="legend-swatch" data-stage="tts" />
                TTS
              </span>
              <span>
                <span className="legend-swatch" data-stage="other" />
                Other
              </span>
            </div>

            {call.metrics && call.metrics.turns && call.metrics.turns.length > 0 ? (
              <div className="flex flex-col gap-1">
                {call.metrics.turns.map((t) => (
                  <LatencyRow key={t.turn} turn={t} />
                ))}
              </div>
            ) : (
              <p className="dim text-sm">No latency metrics recorded</p>
            )}

            {call.metrics ? (
              <div className="flex justify-between mt-3 pt-3 border-t border-line text-xs dim">
                <span>
                  p50: <strong className="num">{fmtMs(call.metrics.p50_e2e_ms)}</strong>
                </span>
                <span>
                  p95: <strong className="num">{fmtMs(call.metrics.p95_e2e_ms)}</strong>
                </span>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
