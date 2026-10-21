"use client";

import { useState } from "react";
import { Badge } from "@/components/Badge";
import { Empty, ErrorBox, Loading } from "@/components/States";
import { Transcript } from "@/components/Transcript";
import { useLiveCalls } from "@/lib/api";
import { fmtDuration } from "@/lib/format";

export default function LivePage() {
  const { data: calls, error, isLoading, mutate } = useLiveCalls();
  const [inFlight, setInFlight] = useState<Record<string, boolean>>({});
  const [takeoverErrors, setTakeoverErrors] = useState<Record<string, string>>({});

  if (isLoading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorBox message={error.message || "Failed to load live calls."} />;
  }

  const list = calls ?? [];

  async function handleTakeover(callId: string) {
    setInFlight((prev) => ({ ...prev, [callId]: true }));
    setTakeoverErrors((prev) => {
      const next = { ...prev };
      delete next[callId];
      return next;
    });
    try {
      const res = await fetch(`/api/live/${encodeURIComponent(callId)}/takeover`, {
        method: "POST",
      });
      if (!res.ok) {
        let detail = `Takeover failed (${res.status})`;
        try {
          const body = await res.json();
          if (body && typeof body.detail === "string") {
            detail = body.detail;
          }
        } catch {
          // ignore JSON parse errors
        }
        setTakeoverErrors((prev) => ({ ...prev, [callId]: detail }));
      } else {
        await mutate();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Takeover request failed";
      setTakeoverErrors((prev) => ({ ...prev, [callId]: msg }));
    } finally {
      setInFlight((prev) => ({ ...prev, [callId]: false }));
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Live calls</h1>
          <p className="page-sub">Active sessions updating every 2s</p>
        </div>
      </div>

      {list.length === 0 ? (
        <div className="card">
          <Empty>No calls in progress.</Empty>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {list.map((call) => {
            const startedMs = new Date(call.started_at).getTime();
            const elapsedSec = isNaN(startedMs)
              ? 0
              : Math.max(0, Math.floor((Date.now() - startedMs) / 1000));
            const busy = Boolean(inFlight[call.call_id]);
            const errMsg = takeoverErrors[call.call_id];

            return (
              <div key={call.call_id} className="card">
                <div className="card-head">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="mono">{call.call_id}</span>
                    <Badge>{call.channel}</Badge>
                    {call.intent ? <Badge tone="info">{call.intent}</Badge> : null}
                    {call.verified ? <Badge tone="ok">verified</Badge> : null}
                    <span className="dim num">{fmtDuration(elapsedSec)}</span>
                  </div>
                  <button
                    type="button"
                    className="btn"
                    data-variant="primary"
                    disabled={busy}
                    onClick={() => handleTakeover(call.call_id)}
                  >
                    Take over
                  </button>
                </div>
                {errMsg ? (
                  <div className="card-pad">
                    <div className="error-box">{errMsg}</div>
                  </div>
                ) : null}
                <Transcript turns={call.transcript} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
