"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/Badge";
import { Empty } from "@/components/States";
import { dispositionTone, fmtDuration, fmtTime } from "@/lib/format";
import type { CallRow } from "@/lib/types";

export function CallsTable({ calls }: { calls: CallRow[] }) {
  const router = useRouter();

  if (!calls || calls.length === 0) {
    return <Empty>No calls found.</Empty>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="table">
        <thead>
          <tr>
            <th scope="col">Time</th>
            <th scope="col">Caller</th>
            <th scope="col">Direction / channel</th>
            <th scope="col">Duration</th>
            <th scope="col">Disposition</th>
            <th scope="col">Summary</th>
          </tr>
        </thead>
        <tbody>
          {calls.map((call) => {
            const firstLine = (call.summary || "—").split("\n")[0];
            return (
              <tr
                key={call.id}
                data-href={call.id}
                onClick={() => router.push(`/calls/${call.id}`)}
              >
                <td>
                  <Link
                    href={`/calls/${call.id}`}
                    className="hover:underline"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {fmtTime(call.started_at)}
                  </Link>
                </td>
                <td>
                  <span className="mono">{call.caller_number || "—"}</span>
                </td>
                <td>
                  <Badge>
                    {call.direction} · {call.channel}
                  </Badge>
                </td>
                <td>
                  <span className="num">{fmtDuration(call.duration_s)}</span>
                </td>
                <td>
                  <Badge tone={dispositionTone(call.disposition)}>
                    {call.disposition}
                  </Badge>
                </td>
                <td>
                  <span className="dim truncate block max-w-md">{firstLine}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
