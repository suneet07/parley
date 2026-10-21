"use client";

import Link from "next/link";
import { Badge } from "@/components/Badge";
import { Empty, ErrorBox, Loading } from "@/components/States";
import { StatCard } from "@/components/StatCard";
import { useMetrics } from "@/lib/api";
import { fmtMs } from "@/lib/format";

export default function LatencyPage() {
  const { data: metrics, error, isLoading } = useMetrics(7);

  if (isLoading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorBox message={error.message || "Failed to load latency data."} />;
  }

  if (!metrics) {
    return <Empty>No latency data available.</Empty>;
  }

  const stt = metrics.stage_avg_ms.stt;
  const llm = metrics.stage_avg_ms.llm;
  const tts = metrics.stage_avg_ms.tts;
  const p50 = metrics.p50_e2e_ms;

  const budgetItems = [
    { name: "End-of-turn", target: 300, targetLabel: "300 ms", measured: null },
    { name: "STT", target: 150, targetLabel: "150 ms", measured: stt },
    { name: "LLM first token", target: 200, targetLabel: "200 ms", measured: llm },
    { name: "TTS first audio", target: 100, targetLabel: "100 ms", measured: tts },
    { name: "Network", target: 50, targetLabel: "50 ms", measured: null },
    { name: "Total", target: 800, targetLabel: "≈ 800 ms", measured: p50 },
  ];

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Latency</h1>
          <p className="page-sub">Turn-by-turn latency budget and breakdown</p>
        </div>
      </div>

      <div className="stat-grid mb-6">
        <StatCard label="p50 response" value={fmtMs(metrics.p50_e2e_ms)} />
        <StatCard label="p95 response" value={fmtMs(metrics.p95_e2e_ms)} />
        <StatCard label="Avg STT" value={fmtMs(stt)} />
        <StatCard label="Avg LLM first token" value={fmtMs(llm)} />
        <StatCard label="Avg TTS first audio" value={fmtMs(tts)} />
      </div>

      <div className="card card-pad mb-6">
        <h2 className="text-sm font-semibold mb-3">Latency budget</h2>
        <div className="overflow-x-auto">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Stage</th>
                <th scope="col" className="text-right">
                  Target
                </th>
                <th scope="col" className="text-right">
                  Measured average
                </th>
              </tr>
            </thead>
            <tbody>
              {budgetItems.map((item) => {
                const isOver = item.measured !== null && item.measured > item.target;
                return (
                  <tr key={item.name}>
                    <td>{item.name}</td>
                    <td className="num text-right dim">{item.targetLabel}</td>
                    <td className={isOver ? "num text-right text-danger" : "num text-right"}>
                      {item.measured !== null ? fmtMs(item.measured) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card card-pad">
        <h2 className="text-sm font-semibold mb-3">Slowest turns</h2>
        {metrics.slowest_turns && metrics.slowest_turns.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Call ID</th>
                  <th scope="col">Turn</th>
                  <th scope="col" className="text-right">
                    STT
                  </th>
                  <th scope="col" className="text-right">
                    LLM
                  </th>
                  <th scope="col" className="text-right">
                    TTS
                  </th>
                  <th scope="col" className="text-right">
                    Total (E2E)
                  </th>
                </tr>
              </thead>
              <tbody>
                {metrics.slowest_turns.map((turn, i) => (
                  <tr key={`${turn.call_id}-${turn.turn}-${i}`}>
                    <td>
                      <Link
                        href={`/calls/${turn.call_id}`}
                        className="mono hover:underline"
                      >
                        {turn.call_id}
                      </Link>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span className="mono dim">T{turn.turn}</span>
                        {turn.interrupted ? (
                          <Badge tone="warn">barge-in</Badge>
                        ) : null}
                      </div>
                    </td>
                    <td className="num text-right">{fmtMs(turn.stt_ms)}</td>
                    <td className="num text-right">{fmtMs(turn.llm_ttft_ms)}</td>
                    <td className="num text-right">{fmtMs(turn.tts_ttfb_ms)}</td>
                    <td className="num text-right font-medium">{fmtMs(turn.e2e_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="dim text-sm">No turn latency data available.</p>
        )}
      </div>
    </div>
  );
}
