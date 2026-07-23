import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { AlertOctagon, AlertTriangle, Info, Search } from "lucide-react";
import { useState } from "react";

import { AppTopbar } from "@/components/app-topbar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/app/alerts")({
  head: () => ({
    meta: [
      { title: "Alert Center — Iron Sight" },
      { name: "description", content: "Every plant signal, triaged and correlated by AI." },
      { property: "og:title", content: "Alert Center — Iron Sight" },
      { property: "og:description", content: "Every plant signal, triaged and correlated by AI." },
    ],
  }),
  component: AlertsPage,
});

type Sev = "critical" | "warning" | "info";
const data: { s: Sev; title: string; asset: string; time: string; owner: string; status: string }[] = [
  { s: "critical", title: "Compressor C-04 pressure over threshold (12.4 bar)", asset: "Rotterdam · Line 3", time: "2m", owner: "Reliability A", status: "Open" },
  { s: "critical", title: "Line A emergency stop pressed by operator", asset: "Nordic 01 · Line A", time: "8m", owner: "Shift lead", status: "Acknowledged" },
  { s: "warning", title: "Bearing B-1147 vibration trending up", asset: "Nordic 01 · Line A", time: "14m", owner: "Reliability B", status: "Open" },
  { s: "warning", title: "Robot arm R-22 torque variance ±8%", asset: "Osaka 04 · Cell 12", time: "1h", owner: "Automation", status: "In progress" },
  { s: "info", title: "Predictive model retrained on 24h data", asset: "All plants", time: "1h", owner: "Elena", status: "Resolved" },
  { s: "warning", title: "Cooling tower T-2 water level below target", asset: "Rotterdam · Utilities", time: "2h", owner: "Facilities", status: "Open" },
  { s: "info", title: "Weekly reliability report ready", asset: "All plants", time: "3h", owner: "Elena", status: "Resolved" },
];

const sevIcon = { critical: AlertOctagon, warning: AlertTriangle, info: Info } as const;
const sevColor = {
  critical: "text-red-300 bg-red-500/15",
  warning: "text-yellow-300 bg-yellow-500/15",
  info: "text-sky-300 bg-sky-500/15",
} as const;

function AlertsPage() {
  const [filter, setFilter] = useState<Sev | "all">("all");
  const list = filter === "all" ? data : data.filter((d) => d.s === filter);

  return (
    <>
      <AppTopbar title="Alert Center" breadcrumbs={["Operations"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        <div className="grid gap-3 md:grid-cols-3">
          {[
            { l: "Critical", v: 3, c: "text-red-300", i: AlertOctagon },
            { l: "Warning", v: 12, c: "text-yellow-300", i: AlertTriangle },
            { l: "Info", v: 24, c: "text-sky-300", i: Info },
          ].map((s) => (
            <div key={s.l} className="flex items-center gap-4 rounded-xl border border-white/10 bg-white/[0.03] p-4">
              <div className={cn("grid size-11 place-items-center rounded-lg bg-white/5", s.c)}>
                <s.i className="size-5" />
              </div>
              <div>
                <div className="text-xs text-muted-foreground">{s.l}</div>
                <div className={cn("text-2xl font-semibold", s.c)}>{s.v}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-white/10 bg-white/[0.03]">
          <div className="flex items-center gap-2 border-b border-white/10 p-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input placeholder="Search alerts…" className="h-9 border-white/10 bg-background/40 pl-9" />
            </div>
            {(["all","critical","warning","info"] as const).map((f) => (
              <Button
                key={f}
                variant={filter === f ? "default" : "ghost"}
                size="sm"
                onClick={() => setFilter(f)}
                className={cn("capitalize", filter === f && "shadow-glow")}
              >
                {f}
              </Button>
            ))}
          </div>

          <ul className="divide-y divide-white/5">
            {list.map((a, i) => {
              const Icon = sevIcon[a.s];
              return (
                <motion.li
                  key={a.title}
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.02 }}
                  className="flex items-start gap-4 p-4 transition hover:bg-white/[0.02]"
                >
                  <div className={cn("grid size-9 shrink-0 place-items-center rounded-lg", sevColor[a.s])}>
                    <Icon className="size-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <div className="truncate text-sm font-medium">{a.title}</div>
                      <Badge variant="outline" className="border-white/10 bg-white/[0.03] text-[10px]">
                        {a.status}
                      </Badge>
                    </div>
                    <div className="mt-0.5 text-xs text-muted-foreground">
                      {a.asset} · {a.time} ago · Owner {a.owner}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" variant="ghost">Snooze</Button>
                    <Button size="sm" variant="outline" className="border-white/10 bg-white/[0.03]">
                      Acknowledge
                    </Button>
                  </div>
                </motion.li>
              );
            })}
          </ul>
        </div>
      </main>
    </>
  );
}
