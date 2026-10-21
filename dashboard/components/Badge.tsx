import type { Tone } from "@/lib/format";

export function Badge({
  tone,
  children,
}: {
  tone?: Tone;
  children: React.ReactNode;
}) {
  return (
    <span className="badge" data-tone={tone}>
      {children}
    </span>
  );
}
