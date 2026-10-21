"use client";

import { useState } from "react";
import { Empty, ErrorBox, Loading } from "@/components/States";
import { useMetrics } from "@/lib/api";

export default function InsightsPage() {
  const [days, setDays] = useState<7 | 30>(7);
  const { data: metrics, error, isLoading } = useMetrics(days);

  if (isLoading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorBox message={error.message || "Failed to load insights."} />;
  }

  if (!metrics) {
    return <Empty>No insights data available.</Empty>;
  }

  const maxIntentCount = Math.max(...(metrics.intents.map((i) => i.count) || []), 1);
  const maxDispCount = Math.max(...(metrics.dispositions.map((d) => d.count) || []), 1);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Insights</h1>
          <p className="page-sub">Intent trends, gaps, and call patterns</p>
        </div>
        <div className="segmented" role="group" aria-label="Time period">
          <button
            type="button"
            aria-pressed={days === 7}
            onClick={() => setDays(7)}
          >
            7d
          </button>
          <button
            type="button"
            aria-pressed={days === 30}
            onClick={() => setDays(30)}
          >
            30d
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Top intents</h2>
          {metrics.intents && metrics.intents.length > 0 ? (
            <div className="flex flex-col gap-2">
              {metrics.intents.map((intent) => {
                const pct = (intent.count / maxIntentCount) * 100;
                return (
                  <div key={intent.name} className="hbar">
                    <span className="text-sm truncate">{intent.name}</span>
                    <div className="hbar-track">
                      <div className="hbar-fill" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="num text-right text-xs dim">{intent.count}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="dim text-sm">No intents recorded.</p>
          )}
        </div>

        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Dispositions</h2>
          {metrics.dispositions && metrics.dispositions.length > 0 ? (
            <div className="flex flex-col gap-2">
              {metrics.dispositions.map((disp) => {
                const pct = (disp.count / maxDispCount) * 100;
                return (
                  <div key={disp.name} className="hbar">
                    <span className="text-sm truncate">{disp.name}</span>
                    <div className="hbar-track">
                      <div className="hbar-fill" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="num text-right text-xs dim">{disp.count}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="dim text-sm">No dispositions recorded.</p>
          )}
        </div>

        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Knowledge gaps</h2>
          {metrics.knowledge_gaps && metrics.knowledge_gaps.length > 0 ? (
            <div>
              <div className="overflow-x-auto">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Question</th>
                      <th scope="col" className="text-right">
                        Count
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.knowledge_gaps.map((kg) => (
                      <tr key={kg.question}>
                        <td>{kg.question}</td>
                        <td className="num text-right">{kg.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="dim text-xs mt-3">Add answers for these to your docs</p>
            </div>
          ) : (
            <p className="dim text-sm">No knowledge gaps recorded.</p>
          )}
        </div>

        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Calls per day</h2>
          {metrics.by_day && metrics.by_day.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Date</th>
                    <th scope="col" className="text-right">
                      Calls
                    </th>
                    <th scope="col" className="text-right">
                      Resolved
                    </th>
                    <th scope="col" className="text-right">
                      Escalated
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.by_day.map((d) => (
                    <tr key={d.date}>
                      <td className="mono">{d.date}</td>
                      <td className="num text-right">{d.calls}</td>
                      <td className="num text-right">{d.resolved}</td>
                      <td className="num text-right">{d.escalated}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="dim text-sm">No daily records found.</p>
          )}
        </div>
      </div>
    </div>
  );
}
