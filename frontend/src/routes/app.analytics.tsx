import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AppTopbar } from "@/components/app-topbar";
import { Badge } from "@/components/ui/badge";

export const Route = createFileRoute("/app/analytics")({
  head: () => ({
    meta: [
      { title: "Analytics — Iron Sight" },
      { name: "description", content: "OEE, throughput, quality and energy — one command dashboard." },
      { property: "og:title", content: "Analytics — Iron Sight" },
      { property: "og:description", content: "OEE, throughput, quality and energy — one command dashboard." },
    ],
  }),
  component: AnalyticsPage,
});

const throughput = Array.from({ length: 14 }).map((_, i) => ({
  d: `D${i + 1}`,
  actual: 1000 + Math.round(Math.sin(i) * 100 + i * 20),
  target: 1200,
}));

const quality = Array.from({ length: 14 }).map((_, i) => ({
  d: `D${i + 1}`,
  pass: 92 + Math.round(Math.sin(i / 2) * 3),
  reject: 8 - Math.round(Math.sin(i / 2) * 2),
}));

const rings = [
  { name: "Availability", value: 92, fill: "oklch(0.74 0.18 160)" },
  { name: "Performance", value: 88, fill: "oklch(0.75 0.15 210)" },
  { name: "Quality", value: 97, fill: "oklch(0.68 0.19 250)" },
];

const heatmap = Array.from({ length: 7 }).map(() =>
  Array.from({ length: 24 }).map(() => Math.random()),
);

function AnalyticsPage() {
  return (
    <>
      <AppTopbar title="Analytics" breadcrumbs={["Overview"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Plant analytics</h2>
            <p className="text-sm text-muted-foreground">
              Last 14 days · All lines
            </p>
          </div>
          <Badge variant="outline" className="border-white/10 bg-white/[0.03]">Live sync</Badge>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <Panel title="OEE decomposition">
            <div className="h-64">
              <ResponsiveContainer>
                <RadialBarChart innerRadius="30%" outerRadius="100%" data={rings} startAngle={90} endAngle={-270}>
                  <RadialBar dataKey="value" cornerRadius={12} background={{ fill: "oklch(1 0 0 / 5%)" }} />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="Throughput" subtitle="Units/day vs. target">
            <div className="h-64">
              <ResponsiveContainer>
                <LineChart data={throughput}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
                  <XAxis dataKey="d" stroke="oklch(0.66 0.02 260)" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="oklch(0.66 0.02 260)" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: "oklch(0.19 0.025 265)", border: "1px solid oklch(1 0 0 / 10%)", borderRadius: 8, fontSize: 12 }} />
                  <Line type="monotone" dataKey="actual" stroke="oklch(0.75 0.15 210)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="target" stroke="oklch(0.66 0.02 260)" strokeDasharray="4 4" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="Quality yield">
            <div className="h-64">
              <ResponsiveContainer>
                <BarChart data={quality}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
                  <XAxis dataKey="d" stroke="oklch(0.66 0.02 260)" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="oklch(0.66 0.02 260)" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: "oklch(0.19 0.025 265)", border: "1px solid oklch(1 0 0 / 10%)", borderRadius: 8, fontSize: 12 }} />
                  <Bar dataKey="pass" stackId="q" fill="oklch(0.74 0.18 160)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="reject" stackId="q" fill="oklch(0.65 0.23 22)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        <Panel title="Downtime heatmap" subtitle="Hours per day · past week">
          <div className="overflow-x-auto">
            <div className="min-w-[600px]">
              <div className="mb-1 grid grid-cols-[60px_repeat(24,1fr)] gap-0.5 text-[10px] text-muted-foreground">
                <div />
                {Array.from({ length: 24 }).map((_, i) => (
                  <div key={i} className="text-center">{i}</div>
                ))}
              </div>
              {heatmap.map((row, i) => (
                <div key={i} className="mb-0.5 grid grid-cols-[60px_repeat(24,1fr)] gap-0.5">
                  <div className="text-[11px] text-muted-foreground">{["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i]}</div>
                  {row.map((v, j) => (
                    <motion.div
                      key={j}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: (i * 24 + j) * 0.003 }}
                      className="h-6 rounded"
                      style={{
                        background: `oklch(${0.3 + v * 0.35} ${0.05 + v * 0.18} ${255 - v * 60})`,
                      }}
                      title={`${v.toFixed(2)}`}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </main>
    </>
  );
}

function Panel({ title, subtitle, children }: any) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="mb-3">
        <div className="text-sm font-semibold tracking-tight">{title}</div>
        {subtitle && <div className="text-xs text-muted-foreground">{subtitle}</div>}
      </div>
      {children}
    </div>
  );
}
