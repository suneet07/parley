"use client";

import { Activity, Gauge, LayoutDashboard, Lightbulb, PhoneCall, Settings2, FlaskConical } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/calls", label: "Calls", icon: PhoneCall },
  { href: "/insights", label: "Insights", icon: Lightbulb },
  { href: "/latency", label: "Latency", icon: Gauge },
  { href: "/evals", label: "Evals", icon: FlaskConical },
  { href: "/config", label: "Config", icon: Settings2 },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <nav className="sidebar" aria-label="Main">
      <div className="brand">
        <span className="brand-mark">P</span>
        parley
      </div>
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        return (
          <Link key={href} href={href} className="nav-item" data-active={String(active)}>
            <Icon aria-hidden />
            {label}
          </Link>
        );
      })}
      <div className="sidebar-foot">
        <Activity aria-hidden className="inline h-3 w-3" /> v0.1
      </div>
    </nav>
  );
}
