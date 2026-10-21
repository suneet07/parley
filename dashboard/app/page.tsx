"use client";

import { CallsTable } from "@/components/CallsTable";
import { ErrorBox, Loading } from "@/components/States";
import { StatCard } from "@/components/StatCard";
import { useCalls, useMetrics } from "@/lib/api";
import { fmtMs, fmtPct, fmtUsd } from "@/lib/format";

export default function OverviewPage() {
  const { data: metrics, error: metricsError, isLoading: metricsLoading } = useMetrics(7);
  const { data: calls, error: callsError, isLoading: callsLoading } = useCalls(8, 0);

  if (metricsLoading || callsLoading) {
    return <Loading />;
  }

  if (metricsError || callsError) {
    return (
      <ErrorBox
        message={
          metricsError?.message || callsError?.message || "Failed to load overview data."
        }
      />
    );
  }

  const p95 = metrics?.p95_e2e_ms;
  const p95Tone = p95 !== null && p95 !== undefined ? (p95 < 1500 ? "ok" : "danger") : undefined;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Overview</h1>
          <p className="page-sub">Last 7 days</p>
        </div>
      </div>

      <div className="stat-grid">
        <StatCard label="Calls" value={metrics?.calls ?? 0} />
        <StatCard label="Resolution rate" value={fmtPct(metrics?.resolution_rate ?? null)} />
        <StatCard label="Escalation rate" value={fmtPct(metrics?.escalation_rate ?? null)} />
        <StatCard label="p50 response" value={fmtMs(metrics?.p50_e2e_ms ?? null)} />
        <StatCard
          label="p95 response"
          value={fmtMs(p95 ?? null)}
          note="target < 1500 ms"
          noteTone={p95Tone}
        />
        <StatCard
          label="Cost per call"
          value={fmtUsd(metrics?.calls ? metrics.cost?.per_call_avg_usd : null)}
        />
      </div>

      <h2 className="section-title">Recent calls</h2>
      <div className="card">
        <CallsTable calls={(calls || []).slice(0, 8)} />
      </div>
    </div>
  );
}
