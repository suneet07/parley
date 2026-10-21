export function fmtMs(n: number | null | undefined): string {
  if (n === null || n === undefined || isNaN(n)) {
    return "—";
  }
  return `${Math.round(n)} ms`;
}

export function fmtPct(r: number | null | undefined): string {
  if (r === null || r === undefined || isNaN(r)) {
    return "—";
  }
  const pct = r <= 1 && r >= 0 ? Math.round(r * 100) : Math.round(r);
  return `${pct}%`;
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || isNaN(s)) {
    return "0:00";
  }
  const total = Math.max(0, Math.round(s));
  const m = Math.floor(total / 60);
  const rem = total % 60;
  return `${m}:${rem < 10 ? "0" : ""}${rem}`;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) {
    return "—";
  }
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) {
      return "—";
    }
    const day = d.getDate();
    const months = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    const month = months[d.getMonth()];
    const hours = String(d.getHours()).padStart(2, "0");
    const mins = String(d.getMinutes()).padStart(2, "0");
    return `${day} ${month}, ${hours}:${mins}`;
  } catch {
    return "—";
  }
}

export type Tone = "ok" | "warn" | "danger" | "info" | "signal";

export function dispositionTone(d: string | null | undefined): Tone | undefined {
  if (!d) {
    return undefined;
  }
  const val = d.toLowerCase();
  if (val === "resolved" || val === "ticket_created" || val === "callback_booked") {
    return "ok";
  }
  if (val === "escalated" || val === "transferred") {
    return "info";
  }
  if (val === "abandoned" || val === "no_answer" || val === "voicemail") {
    return "warn";
  }
  return undefined;
}
