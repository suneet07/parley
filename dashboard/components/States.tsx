export function Loading() {
  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="skeleton h-6 w-full" />
      <div className="skeleton h-6 w-4/5" />
      <div className="skeleton h-6 w-2/3" />
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="error-box">{message}</div>;
}

export function Empty({ children }: { children?: React.ReactNode }) {
  return <div className="empty">{children ?? "No data yet."}</div>;
}
