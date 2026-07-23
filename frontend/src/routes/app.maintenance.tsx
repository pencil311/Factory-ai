import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { AlertTriangle, Calendar, TrendingDown, Wrench } from "lucide-react";

import { AppTopbar } from "@/components/app-topbar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/app/maintenance")({
  head: () => ({
    meta: [
      { title: "Predictive Maintenance — Iron Sight" },
      { name: "description", content: "AI-driven remaining-useful-life predictions for every asset." },
      { property: "og:title", content: "Predictive Maintenance — Iron Sight" },
      { property: "og:description", content: "AI-driven remaining-useful-life predictions for every asset." },
    ],
  }),
  component: MaintenancePage,
});

const cards = [
  { id: "COM-004", name: "Compressor C-04", risk: 87, rul: "6 days", trend: "Rising", severity: "critical", rec: "Replace pressure valve, inspect bearing shaft" },
  { id: "ROB-022", name: "Robotic Arm R-22", risk: 62, rul: "18 days", trend: "Rising", severity: "warning", rec: "Recalibrate torque sensor, lubricate joint 4" },
  { id: "PMP-031", name: "Hydraulic Pump P-31", risk: 48, rul: "26 days", trend: "Stable", severity: "warning", rec: "Filter change during next planned window" },
  { id: "CNC-014", name: "5-Axis CNC #014", risk: 21, rul: "94 days", trend: "Falling", severity: "healthy", rec: "No action recommended" },
  { id: "CNV-018", name: "Conveyor CV-18", risk: 18, rul: "112 days", trend: "Stable", severity: "healthy", rec: "Routine inspection in 30 days" },
  { id: "WLD-007", name: "Welder W-07", risk: 34, rul: "58 days", trend: "Stable", severity: "warning", rec: "Electrode replacement scheduled" },
];

const severityStyles: Record<string, string> = {
  critical: "border-red-500/40 bg-red-500/10",
  warning: "border-yellow-500/40 bg-yellow-500/10",
  healthy: "border-emerald-500/30 bg-emerald-500/10",
};

const sparkData = Array.from({ length: 20 }).map((_, i) => ({ v: 20 + Math.sin(i / 2) * 8 + i * 2 }));

function MaintenancePage() {
  return (
    <>
      <AppTopbar title="Predictive Maintenance" breadcrumbs={["Operations"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Predictive Maintenance</h2>
            <p className="text-sm text-muted-foreground">
              6 assets require attention · 412 hours of downtime avoided this month
            </p>
          </div>
          <Button size="sm" className="shadow-glow">
            <Calendar className="mr-1.5 size-3.5" /> Schedule window
          </Button>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {cards.map((c, i) => (
            <motion.div
              key={c.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className={cn(
                "relative overflow-hidden rounded-2xl border p-5 backdrop-blur",
                severityStyles[c.severity],
              )}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-xs text-muted-foreground">{c.id}</div>
                  <div className="text-base font-semibold tracking-tight">{c.name}</div>
                </div>
                <Badge
                  className={cn(
                    "border-transparent",
                    c.severity === "critical" && "bg-red-500/20 text-red-300",
                    c.severity === "warning" && "bg-yellow-500/20 text-yellow-300",
                    c.severity === "healthy" && "bg-emerald-500/20 text-emerald-300",
                  )}
                >
                  {c.severity}
                </Badge>
              </div>

              <div className="mt-5 flex items-end justify-between">
                <div>
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
                    Failure probability
                  </div>
                  <div className="text-3xl font-semibold tracking-tight">{c.risk}%</div>
                </div>
                <div className="h-14 w-28 opacity-80">
                  <ResponsiveContainer>
                    <AreaChart data={sparkData}>
                      <defs>
                        <linearGradient id={`sp-${c.id}`} x1="0" x2="0" y1="0" y2="1">
                          <stop
                            offset="0"
                            stopColor={c.severity === "critical" ? "oklch(0.65 0.23 22)" : c.severity === "warning" ? "oklch(0.8 0.17 72)" : "oklch(0.74 0.18 160)"}
                            stopOpacity="0.6"
                          />
                          <stop offset="1" stopColor="transparent" />
                        </linearGradient>
                      </defs>
                      <Area
                        type="monotone"
                        dataKey="v"
                        stroke={c.severity === "critical" ? "oklch(0.65 0.23 22)" : c.severity === "warning" ? "oklch(0.8 0.17 72)" : "oklch(0.74 0.18 160)"}
                        strokeWidth={1.5}
                        fill={`url(#sp-${c.id})`}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border border-white/10 bg-background/40 p-2">
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">RUL</div>
                  <div className="mt-0.5 font-mono">{c.rul}</div>
                </div>
                <div className="rounded-lg border border-white/10 bg-background/40 p-2">
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Trend</div>
                  <div className="mt-0.5 flex items-center gap-1">
                    {c.trend === "Falling" ? <TrendingDown className="size-3 text-emerald-400" /> : <AlertTriangle className="size-3 text-warning" />}
                    {c.trend}
                  </div>
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-white/10 bg-background/40 p-3 text-xs leading-relaxed">
                <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-muted-foreground">
                  <Wrench className="size-3" /> Recommendation
                </div>
                {c.rec}
              </div>
            </motion.div>
          ))}
        </div>
      </main>
    </>
  );
}
