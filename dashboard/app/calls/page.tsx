"use client";

import { useMemo, useState } from "react";
import { CallsTable } from "@/components/CallsTable";
import { ErrorBox, Loading } from "@/components/States";
import { useCalls } from "@/lib/api";
import type { CallRow } from "@/lib/types";

type FilterType = "All" | "Resolved" | "Escalated" | "Other";

const RESOLVED_DISPOSITIONS = new Set(["resolved", "ticket_created", "callback_booked"]);

export default function CallsPage() {
  const { data: calls, error, isLoading } = useCalls(100, 0);
  const [search, setSearch] = useState("");
  const [activeFilter, setActiveFilter] = useState<FilterType>("All");

  const filteredCalls = useMemo(() => {
    if (!calls) return [];

    return calls.filter((call: CallRow) => {
      // 1. Filter by category
      if (activeFilter === "Resolved") {
        if (!RESOLVED_DISPOSITIONS.has(call.disposition)) {
          return false;
        }
      } else if (activeFilter === "Escalated") {
        if (!call.escalated && call.disposition !== "escalated" && call.disposition !== "transferred") {
          return false;
        }
      } else if (activeFilter === "Other") {
        const isResolved = RESOLVED_DISPOSITIONS.has(call.disposition);
        const isEscalated =
          call.escalated ||
          call.disposition === "escalated" ||
          call.disposition === "transferred";
        if (isResolved || isEscalated) {
          return false;
        }
      }

      // 2. Filter by search query
      const q = search.trim().toLowerCase();
      if (!q) return true;

      const summaryMatch = (call.summary || "").toLowerCase().includes(q);
      const callerMatch = (call.caller_number || "").toLowerCase().includes(q);
      const idMatch = call.id.toLowerCase().includes(q);

      return summaryMatch || callerMatch || idMatch;
    });
  }, [calls, activeFilter, search]);

  if (isLoading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorBox message={error.message || "Failed to load calls."} />;
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Calls</h1>
          <p className="page-sub">Recent call logs and outcomes</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-4 mb-6">
        <input
          type="text"
          className="input"
          placeholder="Search summary, caller, or id..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="segmented" role="group" aria-label="Filter calls by outcome">
          {(["All", "Resolved", "Escalated", "Other"] as const).map((filter) => (
            <button
              key={filter}
              type="button"
              aria-pressed={activeFilter === filter}
              onClick={() => setActiveFilter(filter)}
            >
              {filter}
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <CallsTable calls={filteredCalls} />
      </div>
    </div>
  );
}
