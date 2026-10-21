import { type NextRequest, NextResponse } from "next/server";

const API_URL = process.env.PARLEY_API_URL ?? "http://localhost:8000";
const ALLOWED_GET_PATH = /^(calls|metrics|evals|config|live|outbound)(\/|$)/;
const ALLOWED_POST_PATH = /^live\/[^/]+\/takeover$/;

function buildAuthHeaders(): HeadersInit {
  const headers: Record<string, string> = {};
  if (process.env.PARLEY_API_KEY) {
    headers["Authorization"] = `Bearer ${process.env.PARLEY_API_KEY}`;
  }
  return headers;
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
): Promise<Response> {
  const { path } = await context.params;
  const joined = path.join("/");
  if (!ALLOWED_GET_PATH.test(joined)) {
    return NextResponse.json({ detail: "Not found" }, { status: 404 });
  }

  const search = request.nextUrl.search;
  const targetUrl = `${API_URL.replace(/\/$/, "")}/api/${joined}${search}`;
  const upstream = await fetch(targetUrl, {
    method: "GET",
    headers: buildAuthHeaders(),
    cache: "no-store",
  });
  const body = await upstream.text();
  return new Response(body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
): Promise<Response> {
  const { path } = await context.params;
  const joined = path.join("/");
  if (!ALLOWED_POST_PATH.test(joined)) {
    return NextResponse.json({ detail: "Not found" }, { status: 404 });
  }

  const search = request.nextUrl.search;
  const targetUrl = `${API_URL.replace(/\/$/, "")}/api/${joined}${search}`;
  const upstream = await fetch(targetUrl, {
    method: "POST",
    headers: buildAuthHeaders(),
    cache: "no-store",
  });
  const body = await upstream.text();
  return new Response(body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
