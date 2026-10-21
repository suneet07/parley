import { Badge } from "@/components/Badge";
import { fmtMs } from "@/lib/format";
import type { TurnLatency } from "@/lib/types";

export function LatencyRow({
  turn,
  maxE2E = 1500,
}: {
  turn: TurnLatency;
  maxE2E?: number;
}) {
  const scale = Math.max(1500, maxE2E);
  const stt = turn.stt_ms ?? 0;
  const llm = turn.llm_ttft_ms ?? 0;
  const tts = turn.tts_ttfb_ms ?? 0;
  const e2e = turn.e2e_ms ?? stt + llm + tts;
  const other = Math.max(0, e2e - (stt + llm + tts));

  const sttPct = scale > 0 ? (stt / scale) * 100 : 0;
  const llmPct = scale > 0 ? (llm / scale) * 100 : 0;
  const ttsPct = scale > 0 ? (tts / scale) * 100 : 0;
  const otherPct = scale > 0 ? (other / scale) * 100 : 0;

  return (
    <div className="lat-row">
      <div className="mono dim">T{turn.turn}</div>
      <div className="flex items-center gap-2 min-w-0">
        <div className="lat-track flex-1">
          {sttPct > 0 && <div className="lat-seg" data-stage="stt" style={{ width: `${sttPct}%` }} />}
          {llmPct > 0 && <div className="lat-seg" data-stage="llm" style={{ width: `${llmPct}%` }} />}
          {ttsPct > 0 && <div className="lat-seg" data-stage="tts" style={{ width: `${ttsPct}%` }} />}
          {otherPct > 0 && <div className="lat-seg" data-stage="other" style={{ width: `${otherPct}%` }} />}
        </div>
        {turn.interrupted ? <Badge tone="warn">barge-in</Badge> : null}
      </div>
      <div className="num text-right">{fmtMs(turn.e2e_ms)}</div>
    </div>
  );
}
