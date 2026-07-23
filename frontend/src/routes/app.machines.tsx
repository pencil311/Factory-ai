import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Activity, Filter, MoreHorizontal, Plus, Search } from "lucide-react";

import { AppTopbar } from "@/components/app-topbar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/app/machines")({
  head: () => ({
    meta: [
      { title: "Machines — Iron Sight" },
      { name: "description", content: "Every asset, sensor, and cell — one live inventory." },
      { property: "og:title", content: "Machines — Iron Sight" },
      { property: "og:description", content: "Every asset, sensor, and cell — one live inventory." },
    ],
  }),
  component: MachinesPage,
});

const machines = [
  { id: "CNC-014", name: "5-Axis CNC #014", line: "Line A", status: "Running", health: 96, temp: "42°C", vib: "0.4g", oee: "91%" },
  { id: "ROB-022", name: "Robotic Arm R-22", line: "Cell 12", status: "Warning", health: 78, temp: "48°C", vib: "1.2g", oee: "84%" },
  { id: "COM-004", name: "Compressor C-04", line: "Line 3", status: "Critical", health: 41, temp: "72°C", vib: "3.1g", oee: "62%" },
  { id: "CNV-018", name: "Conveyor CV-18", line: "Line B", status: "Running", health: 88, temp: "31°C", vib: "0.2g", oee: "95%" },
  { id: "WLD-007", name: "Welder W-07", line: "Line A", status: "Idle", health: 92, temp: "22°C", vib: "0.0g", oee: "0%" },
  { id: "PMP-031", name: "Hydraulic Pump P-31", line: "Utilities", status: "Maintenance", health: 68, temp: "38°C", vib: "0.9g", oee: "—" },
  { id: "CNC-021", name: "5-Axis CNC #021", line: "Line A", status: "Running", health: 94, temp: "40°C", vib: "0.5g", oee: "89%" },
  { id: "ROB-018", name: "Robotic Arm R-18", line: "Cell 12", status: "Running", health: 90, temp: "45°C", vib: "0.6g", oee: "87%" },
];

const statusStyles: Record<string, string> = {
  Running: "bg-emerald-500/15 text-emerald-300",
  Warning: "bg-yellow-500/15 text-yellow-300",
  Critical: "bg-red-500/15 text-red-300",
  Idle: "bg-slate-500/20 text-slate-300",
  Maintenance: "bg-sky-500/15 text-sky-300",
};

function MachinesPage() {
  return (
    <>
      <AppTopbar title="Machines" breadcrumbs={["Operations"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Machines</h2>
            <p className="text-sm text-muted-foreground">
              128 assets · 47,281 sensors · updated 3s ago
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="border-white/10 bg-white/[0.03]">
              <Filter className="mr-1.5 size-3.5" /> Filter
            </Button>
            <Button size="sm" className="shadow-glow">
              <Plus className="mr-1.5 size-3.5" /> Add asset
            </Button>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-4">
          {[
            { l: "Running", v: 96, c: "text-emerald-300" },
            { l: "Warning", v: 18, c: "text-yellow-300" },
            { l: "Critical", v: 3, c: "text-red-300" },
            { l: "Maintenance", v: 11, c: "text-sky-300" },
          ].map((s) => (
            <div key={s.l} className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
              <div className="text-xs text-muted-foreground">{s.l}</div>
              <div className={cn("mt-1 text-2xl font-semibold", s.c)}>{s.v}</div>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-white/10 bg-white/[0.03]">
          <div className="flex items-center gap-2 border-b border-white/10 p-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input placeholder="Search 128 assets…" className="h-9 border-white/10 bg-background/40 pl-9" />
            </div>
            <Button variant="ghost" size="sm">All lines</Button>
            <Button variant="ghost" size="sm">All plants</Button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted-foreground">
                <tr className="border-b border-white/5">
                  <th className="px-4 py-3 font-medium">Asset</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Health</th>
                  <th className="px-4 py-3 font-medium">Temp</th>
                  <th className="px-4 py-3 font-medium">Vibration</th>
                  <th className="px-4 py-3 font-medium">OEE</th>
                  <th className="w-8"></th>
                </tr>
              </thead>
              <tbody>
                {machines.map((m, i) => (
                  <motion.tr
                    key={m.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.02 }}
                    className="border-b border-white/5 last:border-0 hover:bg-white/[0.02]"
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <div className="grid size-8 place-items-center rounded-md bg-white/5 ring-1 ring-white/10">
                          <Activity className="size-3.5 text-primary" />
                        </div>
                        <div>
                          <div className="font-medium">{m.name}</div>
                          <div className="text-xs text-muted-foreground">{m.id} · {m.line}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <Badge className={cn("border-transparent", statusStyles[m.status])}>{m.status}</Badge>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-24 overflow-hidden rounded-full bg-white/10">
                          <div
                            className={cn(
                              "h-full rounded-full",
                              m.health > 80 ? "bg-emerald-400" : m.health > 60 ? "bg-yellow-400" : "bg-red-400",
                            )}
                            style={{ width: `${m.health}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs">{m.health}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs">{m.temp}</td>
                    <td className="px-4 py-3 font-mono text-xs">{m.vib}</td>
                    <td className="px-4 py-3 font-mono text-xs">{m.oee}</td>
                    <td className="px-4 py-3">
                      <button className="grid size-7 place-items-center rounded-md text-muted-foreground hover:bg-white/5">
                        <MoreHorizontal className="size-4" />
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </>
  );
}
