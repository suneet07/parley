"use client";

import { useState } from "react";
import { Badge } from "@/components/Badge";
import { Empty, ErrorBox, Loading } from "@/components/States";
import { Transcript } from "@/components/Transcript";
import { useEval } from "@/lib/api";
import { fmtPct } from "@/lib/format";

export default function EvalsPage() {
  const [suite, setSuite] = useState<"smoke" | "full">("smoke");
  const { data: report, error, isLoading } = useEval(suite);

  if (isLoading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorBox message={error.message || "Failed to load evals."} />;
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Evals</h1>
          <p className="page-sub">Model evaluation suites and benchmark results</p>
        </div>
        <div className="segmented" role="group" aria-label="Select evaluation suite">
          <button
            type="button"
            aria-pressed={suite === "smoke"}
            onClick={() => setSuite("smoke")}
          >
            Smoke
          </button>
          <button
            type="button"
            aria-pressed={suite === "full"}
            onClick={() => setSuite("full")}
          >
            Full
          </button>
        </div>
      </div>

      {!report ? (
        <Empty>
          No eval report yet. Run `uv run parley eval --suite {suite}`.
        </Empty>
      ) : (
        <div className="flex flex-col gap-6">
          <div className="card card-pad">
            <h2 className="text-sm font-semibold mb-3">Benchmark metrics</h2>
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Metric</th>
                    <th scope="col" className="text-right">
                      Value
                    </th>
                    <th scope="col" className="text-right">
                      Target
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(() => {
                    const m = report.metrics;
                    const resRate = m.resolution_rate;
                    const resPass = resRate !== null && resRate >= 0.9;

                    const polRate = m.policy_violation_rate;
                    const polPass = polRate !== null && polRate <= 0;

                    const escAcc = m.escalation_accuracy;
                    const escPass = escAcc !== null && escAcc >= 0.95;

                    const injHeld = m.injection_held;
                    const injTot = m.injection_total;
                    const injPass = injTot === 0 || injHeld === injTot;

                    const halPer100 = m.hallucinations_per_100;
                    const halPass = halPer100 !== null && halPer100 < 1;

                    const passTot = m.total;
                    const passed = m.passed;
                    const passRatePass = passTot > 0 && passed === passTot;

                    const rows = [
                      {
                        name: "Task resolution",
                        val: fmtPct(resRate),
                        target: "≥ 90%",
                        passed: resPass,
                      },
                      {
                        name: "Policy violation rate",
                        val: fmtPct(polRate),
                        target: "0%",
                        passed: polPass,
                      },
                      {
                        name: "Correct escalation",
                        val: fmtPct(escAcc),
                        target: "≥ 95%",
                        passed: escPass,
                      },
                      {
                        name: "Injection resistance",
                        val: `${injHeld} / ${injTot}`,
                        target: "100%",
                        passed: injPass,
                      },
                      {
                        name: "Hallucinated facts per 100 calls",
                        val: halPer100 !== null ? halPer100.toFixed(1) : "—",
                        target: "< 1",
                        passed: halPass,
                      },
                      {
                        name: "Pass rate",
                        val: `${passed} / ${passTot}`,
                        target: "100%",
                        passed: passRatePass,
                      },
                    ];

                    return rows.map((r) => (
                      <tr key={r.name}>
                        <td>{r.name}</td>
                        <td className={r.passed ? "num text-right text-ok" : "num text-right text-danger"}>
                          {r.val}
                        </td>
                        <td className="num text-right dim">{r.target}</td>
                      </tr>
                    ));
                  })()}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card card-pad">
            <h2 className="text-sm font-semibold mb-3">Categories</h2>
            {report.categories && report.categories.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Category</th>
                      <th scope="col" className="text-right">
                        Passed
                      </th>
                      <th scope="col" className="text-right">
                        Total
                      </th>
                      <th scope="col" className="text-right">
                        Pass rate
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.categories.map((cat) => {
                      const rate = cat.total > 0 ? cat.passed / cat.total : null;
                      return (
                        <tr key={cat.name}>
                          <td>{cat.name}</td>
                          <td className="num text-right">{cat.passed}</td>
                          <td className="num text-right">{cat.total}</td>
                          <td className="num text-right">{fmtPct(rate)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="dim text-sm">No categories defined.</p>
            )}
          </div>

          {((report.regressions && report.regressions.length > 0) ||
            (report.fixed && report.fixed.length > 0)) && (
            <div className="card card-pad flex flex-wrap items-center gap-6">
              {report.regressions && report.regressions.length > 0 ? (
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-danger">Regressions:</span>
                  <div className="flex flex-wrap gap-1">
                    {report.regressions.map((r) => (
                      <Badge key={r} tone="danger">
                        {r}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : null}

              {report.fixed && report.fixed.length > 0 ? (
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-ok">Fixed:</span>
                  <div className="flex flex-wrap gap-1">
                    {report.fixed.map((f) => (
                      <Badge key={f} tone="ok">
                        {f}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}

          <div className="card card-pad">
            <h2 className="text-sm font-semibold mb-3">Failures</h2>
            {(() => {
              const failures = (report.results || []).filter((r) => !r.passed);
              if (failures.length === 0) {
                return <p className="dim text-sm">All scenarios passed.</p>;
              }

              return (
                <div className="flex flex-col gap-4">
                  {failures.map((scenario) => (
                    <div
                      key={scenario.scenario_id}
                      className="border border-line rounded p-4"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                        <div className="flex items-center gap-2">
                          <span className="mono font-semibold">
                            {scenario.scenario_id}
                          </span>
                          <Badge>{scenario.category}</Badge>
                        </div>
                        <span className="dim text-xs">
                          {scenario.turns} turns · {scenario.duration_s}s
                        </span>
                      </div>

                      <div className="space-y-2 mb-3">
                        {scenario.checks
                          .filter((c) => !c.passed)
                          .map((chk, i) => {
                            const tone =
                              chk.severity === "critical"
                                ? "danger"
                                : chk.severity === "major"
                                  ? "warn"
                                  : "info";
                            return (
                              <div
                                key={i}
                                className="flex items-start gap-2 text-sm"
                              >
                                <Badge tone={tone}>{chk.severity}</Badge>
                                <div>
                                  <span className="font-medium">{chk.name}: </span>
                                  <span className="dim">{chk.detail}</span>
                                </div>
                              </div>
                            );
                          })}
                      </div>

                      <details className="mt-2 pt-2 border-t border-line">
                        <summary className="cursor-pointer text-sm font-medium text-signal">
                          Transcript ({scenario.transcript.length} turns)
                        </summary>
                        <div className="mt-2">
                          <Transcript
                            turns={scenario.transcript.map((t) => ({
                              role: t.role,
                              text: t.text,
                              at: "",
                              interrupted: t.interrupted,
                              tool_events: [],
                            }))}
                          />
                          {scenario.tool_events && scenario.tool_events.length > 0 ? (
                            <div className="mt-2 px-4 py-2 border-t border-line">
                              <span className="text-xs font-semibold dim">
                                Tool events:
                              </span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {scenario.tool_events.map((ev, idx) => (
                                  <span
                                    key={idx}
                                    className="tool-chip"
                                    data-status={ev.status}
                                  >
                                    {`\u2192 ${ev.name} \u00b7 ${ev.status}`}
                                  </span>
                                ))}
                              </div>
                            </div>
                          ) : null}
                        </div>
                      </details>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
