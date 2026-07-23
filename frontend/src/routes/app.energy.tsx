import { createFileRoute } from "@tanstack/react-router";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Leaf, TrendingDown, Zap } from "lucide-react";

import { AppTopbar } from "@/components/app-topbar";

export const Route = createFileRoute("/app/energy")({
  head: () => ({
    meta: [
      { title: "Energy — Iron Sight" },
      { name: "description", content: "Track, optimize and forecast plant energy consumption." },
      { property: "og:title", content: "Energy — Iron Sight" },
      { property: "og:description", content: "Track, optimize and forecast plant energy consumption." },
    ],
  }),
  component: EnergyPage,
});

const data = Array.from({ length: 30 }).map((_, i) => ({
  d: `D${i + 1}`,
  kwh: 320 + Math.round(Math.sin(i / 3) * 45 + Math.random() * 20),
  saved: 12 + Math.round(Math.cos(i / 4) * 8 + i * 0.4),
}));

function EnergyPage() {
  return (
    <>
      <AppTopbar title="Energy Monitoring" breadcrumbs={["Resources"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        <div className="grid gap-3 md:grid-cols-3">
          {[
            { l: "Today", v: "412 kWh", d: "-6.4%", i: Zap },
            { l: "Saved MTD", v: "1.28 MWh", d: "$184", i: TrendingDown },
            { l: "CO₂e avoided", v: "612 kg", d: "This month", i: Leaf },
          ].map((k) => (
            <div key={k.l} className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
              <div className="mb-3 grid size-10 place-items-center rounded-lg bg-primary/15">
                <k.i className="size-5 text-primary" />
              </div>
              <div className="text-xs text-muted-foreground">{k.l}</div>
              <div className="text-2xl font-semibold tracking-tight">{k.v}</div>
              <div className="text-xs text-emerald-400">{k.d}</div>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
          <div className="mb-3">
            <div className="text-sm font-semibold">Consumption vs. AI-optimized</div>
            <div className="text-xs text-muted-foreground">Last 30 days · kWh/day</div>
          </div>
          <div className="h-80">
            <ResponsiveContainer>
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="e1" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0" stopColor="oklch(0.68 0.19 250)" stopOpacity="0.5" />
                    <stop offset="1" stopColor="oklch(0.68 0.19 250)" stopOpacity="0" />
                  </linearGradient>
                  <linearGradient id="e2" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0" stopColor="oklch(0.74 0.18 160)" stopOpacity="0.5" />
                    <stop offset="1" stopColor="oklch(0.74 0.18 160)" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
                <XAxis dataKey="d" stroke="oklch(0.66 0.02 260)" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="oklch(0.66 0.02 260)" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "oklch(0.19 0.025 265)", border: "1px solid oklch(1 0 0 / 10%)", borderRadius: 8, fontSize: 12 }} />
                <Area type="monotone" dataKey="kwh" stroke="oklch(0.75 0.15 210)" strokeWidth={2} fill="url(#e1)" />
                <Area type="monotone" dataKey="saved" stroke="oklch(0.74 0.18 160)" strokeWidth={2} fill="url(#e2)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </>
  );
}
