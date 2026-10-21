import type { Turn } from "@/lib/types";

export function Transcript({ turns }: { turns: Turn[] }) {
  if (!turns || turns.length === 0) {
    return <div className="dim p-4">No transcript available.</div>;
  }

  return (
    <div className="transcript">
      {turns.map((turn, idx) => {
        const who =
          turn.role === "assistant"
            ? "Agent"
            : turn.role === "user"
              ? "Caller"
              : "System";

        return (
          <div
            key={idx}
            className="msg"
            data-role={turn.role}
            data-interrupted={String(turn.interrupted)}
          >
            <div className="msg-who">{who}</div>
            <div>
              <div className="msg-text">{turn.text}</div>
              {turn.tool_events && turn.tool_events.length > 0 ? (
                <div className="flex flex-wrap">
                  {turn.tool_events.map((ev, i) => {
                    let argsStr = "{}";
                    try {
                      argsStr = JSON.stringify(ev.args ?? {});
                    } catch {
                      argsStr = "{}";
                    }
                    const shortArgs =
                      argsStr.length > 60 ? argsStr.slice(0, 57) + "..." : argsStr;
                    return (
                      <span key={i} className="tool-chip" data-status={ev.status}>
                        {`\u2192 ${ev.name}(${shortArgs}) \u00b7 ${ev.status}`}
                      </span>
                    );
                  })}
                </div>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}
