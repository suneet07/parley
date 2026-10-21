import useSWR from "swr";
import type { CallDetail, CallRow, ConfigView, EvalReport, Metrics } from "./types";

export async function fetcher<T>(url: string): Promise<T | null> {
  const res = await fetch(url);
  if (res.status === 404) {
    return null;
  }
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function useCalls(limit = 50, offset = 0) {
  return useSWR<CallRow[] | null>(`/api/calls?limit=${limit}&offset=${offset}`, fetcher);
}

export function useCall(id: string | null | undefined) {
  return useSWR<CallDetail | null>(id ? `/api/calls/${id}` : null, fetcher);
}

export function useMetrics(days = 7) {
  return useSWR<Metrics | null>(`/api/metrics?days=${days}`, fetcher);
}

export function useEval(suite = "smoke") {
  return useSWR<EvalReport | null>(suite ? `/api/evals/latest?suite=${suite}` : null, fetcher);
}

export function useConfig() {
  return useSWR<ConfigView | null>("/api/config", fetcher);
}
