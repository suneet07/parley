import type { NextConfig } from "next";

// The dashboard talks to the parley API (FastAPI). In dev it runs on :8000.
const API_URL = process.env.PARLEY_API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
