export function StatCard({
  label,
  value,
  note,
  noteTone,
}: {
  label: string;
  value: React.ReactNode;
  note?: string;
  noteTone?: "ok" | "danger";
}) {
  return (
    <div className="card stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note ? (
        <div className="stat-note" data-tone={noteTone}>
          {note}
        </div>
      ) : null}
    </div>
  );
}
